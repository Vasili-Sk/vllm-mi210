# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static expert residency for the Triton WNA16 MoE path."""

from __future__ import annotations

import gc
import json
import re
from typing import Any

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_wna16_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
)
from vllm.triton_utils import tl

logger = init_logger(__name__)

_RANKINGS: dict[int, list[int]] = {}
_BUILT: set[int] = set()
_DEFERRED: list[tuple[Any, torch.nn.Module, int]] = []


def tier_size() -> int:
    return envs.VLLM_WNA16_HOT_TIER_SIZE


def rankings_path() -> str | None:
    return envs.VLLM_WNA16_HOT_TIER_FILE


def _load_rankings() -> dict[int, list[int]]:
    if _RANKINGS:
        return _RANKINGS
    path = rankings_path()
    if not path:
        return _RANKINGS
    with open(path) as rankings_file:
        raw = json.load(rankings_file)
    _RANKINGS.update({int(layer): list(order) for layer, order in raw.items()})
    return _RANKINGS


def _layer_index(layer_name: str) -> int | None:
    match = re.search(r"layers\.(\d+)\.", layer_name)
    return int(match.group(1)) if match else None


def _num_experts(layer: torch.nn.Module) -> int | None:
    for name in ("w13_weight_packed", "w13_qweight", "w13_weight"):
        tensor = getattr(layer, name, None)
        if tensor is not None:
            return tensor.shape[0]
    return None


def _uva_view(cpu_tensor: torch.Tensor) -> torch.Tensor:
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    return get_accelerator_view_from_cpu_tensor(cpu_tensor)


def _find_experts(obj: Any, depth: int = 0) -> Any:
    if obj is None or depth > 5:
        return None
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
        TritonWNA16Experts,
    )

    if isinstance(obj, TritonWNA16Experts):
        return obj
    for value in vars(obj).values():
        if isinstance(value, TritonWNA16Experts):
            return value
        result = _find_experts(value, depth + 1)
        if result is not None:
            return result
    return None


def build_and_attach(
    moe_method: Any,
    layer: torch.nn.Module,
    forced_index: int | None = None,
) -> bool:
    """Split one expert layer into resident and host-backed rows."""
    capacity = tier_size()
    rankings = _load_rankings()
    if capacity <= 0 or not rankings:
        return False

    layer_index = forced_index
    if layer_index is None:
        layer_index = _layer_index(getattr(moe_method, "layer_name", "") or "")
    if layer_index is None:
        layer_index = len(_BUILT)
    if layer_index not in rankings:
        logger.info("hot tier: leave unranked MoE layer %d unchanged", layer_index)
        return False
    if layer_index in _BUILT:
        return True

    num_experts = _num_experts(layer)
    order = rankings[layer_index]
    if num_experts is None or len(order) != num_experts:
        raise ValueError(
            f"hot tier layer {layer_index} has {num_experts} experts but "
            f"its ranking has {len(order)} entries"
        )
    if not 0 < capacity < num_experts:
        raise ValueError(
            f"hot tier capacity must be in 1..{num_experts - 1}, got {capacity}"
        )
    if sorted(order) != list(range(num_experts)):
        raise ValueError(f"hot tier layer {layer_index} ranking is not a permutation")

    hot_ids = sorted(order[:capacity])
    cold_ids = sorted(order[capacity:])
    device = torch.device("cuda")
    cold_cpu: dict[str, torch.Tensor] = {}

    def find(*names: str) -> tuple[str | None, torch.Tensor | None, bool]:
        for name in names:
            value = getattr(layer, name, None)
            if value is not None:
                tensor = value.data if isinstance(value, torch.nn.Parameter) else value
                return (
                    name,
                    tensor,
                    bool(getattr(value, "_vllm_is_uva_offloaded", False)),
                )
        return None, None, False

    names = {
        "w1": find("w13_weight_packed", "w13_qweight", "w13_weight"),
        "w2": find("w2_weight_packed", "w2_qweight", "w2_weight"),
        "s1": find("w13_weight_scale", "w13_scales"),
        "s2": find("w2_weight_scale", "w2_scales"),
        "z1": find("w13_weight_zero_point", "w13_qzeros"),
        "z2": find("w2_weight_zero_point", "w2_qzeros"),
    }
    if names["w1"][0] is None or names["w2"][0] is None:
        raise ValueError(f"hot tier layer {layer_index} has no expert weights")
    source = {key: names[key][1] for key in names}
    compact_cold = not names["w1"][2] or envs.VLLM_WNA16_HOT_TIER_COMPACT_UVA

    def take(
        tensor: torch.Tensor | None,
        ids: list[int],
        location: str,
        key: str,
    ) -> torch.Tensor | None:
        if tensor is None or tensor.untyped_storage().nbytes() == 0:
            return None
        selected = tensor.index_select(
            0, torch.tensor(ids, dtype=torch.long, device=tensor.device)
        )
        if location == "device":
            return selected.to(device=device, copy=True)
        cpu_tensor = selected.to(device="cpu").contiguous().pin_memory()
        cold_cpu[key] = cpu_tensor
        return _uva_view(cpu_tensor)

    hot = {key: take(tensor, hot_ids, "device", key) for key, tensor in source.items()}
    cold = (
        {key: take(tensor, cold_ids, "host", key) for key, tensor in source.items()}
        if compact_cold
        else source
    )

    hot_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    cold_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    for local_index, expert in enumerate(hot_ids):
        hot_map[expert] = local_index
    for local_index, expert in enumerate(cold_ids):
        cold_map[expert] = local_index if compact_cold else expert

    if compact_cold:
        for key in ("w1", "w2", "s1", "s2", "z1", "z2"):
            name = names[key][0]
            value = cold[key]
            if name is None or value is None:
                continue
            old = getattr(layer, name, None)
            if isinstance(old, torch.nn.Parameter):
                old.data = value
            else:
                setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))
        for alias, key in (("w13_weight", "w1"), ("w2_weight", "w2")):
            value = getattr(layer, alias, None)
            if isinstance(value, torch.nn.Parameter) and value.shape[0] == num_experts:
                value.data = cold[key]

        quant_config = getattr(moe_method, "moe_quant_config", None)
        for side, scale, zero_point in (
            ("_w1", cold["s1"], cold["z1"]),
            ("_w2", cold["s2"], cold["z2"]),
        ):
            config = getattr(quant_config, side, None)
            if config is None:
                continue
            if scale is not None:
                config.scale = scale
            if zero_point is not None and getattr(config, "zp", None) is not None:
                config.zp = zero_point

    tier = {
        "hot_w1": hot["w1"],
        "hot_w2": hot["w2"],
        "hot_s1": hot["s1"],
        "hot_s2": hot["s2"],
        "hot_z1": hot["z1"],
        "hot_z2": hot["z2"],
        "hot_map": hot_map,
        "cold_map": cold_map,
        "global_num_experts": num_experts,
        "layer_index": layer_index,
        "cold_cpu": cold_cpu,
    }
    experts = _find_experts(getattr(moe_method, "moe_kernel", None))
    if experts is None:
        raise RuntimeError("hot tier: Triton WNA16 experts instance is missing")
    experts._tier = tier
    _BUILT.add(layer_index)
    logger.info(
        "hot tier layer %d: %d resident experts and %d host-backed experts",
        layer_index,
        len(hot_ids),
        len(cold_ids),
    )
    return True


def defer_layer(moe_method: Any, layer: torch.nn.Module) -> bool:
    """Place all ranked layers in reverse order to limit peak device memory."""
    rankings = _load_rankings()
    if tier_size() <= 0 or not rankings:
        return False
    index = len(_DEFERRED)
    if index >= len(rankings):
        logger.info("hot tier: leave additional MoE layer %d unchanged", index)
        return False
    _DEFERRED.append((moe_method, layer, index))
    if len(_DEFERRED) < len(rankings):
        return False

    logger.info("hot tier: place %d layers in reverse order", len(_DEFERRED))
    for method, target, target_index in reversed(_DEFERRED):
        build_and_attach(method, target, forced_index=target_index)
        if target_index % 4 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    return True


def _apply_once(
    experts: Any,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    s1: torch.Tensor,
    s2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    apply_router_weight_on_input: bool,
    *,
    cold_w1: torch.Tensor,
    cold_w2: torch.Tensor,
    cold_s1: torch.Tensor,
    cold_s2: torch.Tensor,
    hot_map: torch.Tensor,
    cold_map: torch.Tensor,
) -> None:
    num_tokens = hidden_states.size(0)
    top_k = topk_ids.size(1)
    _, _, intermediate_size, hidden_size, _ = experts.moe_problem_size(
        hidden_states, w1, w2, topk_ids
    )
    config = try_get_optimal_moe_config(
        w1.size(),
        w2.size(),
        top_k,
        experts.quant_config.config_name(hidden_states.dtype),
        num_tokens,
        block_shape=experts.block_shape,
    )
    cache1 = _resize_cache(workspace2, (num_tokens, top_k, intermediate_size))
    activation_size = experts.adjust_N_for_activation(intermediate_size, activation)
    cache2 = _resize_cache(workspace13, (num_tokens * top_k, activation_size))
    cache3 = _resize_cache(workspace2, (num_tokens, top_k, hidden_size))
    sorted_ids, expert_ids, padded_tokens = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], global_num_experts, expert_map
    )

    invoke_fused_moe_wna16_triton_kernel(
        hidden_states,
        w1,
        cache1,
        s1,
        None,
        None,
        sorted_ids,
        expert_ids,
        padded_tokens,
        False,
        top_k,
        config,
        compute_type=tl.bfloat16,
        use_int8_w8a16=experts.quant_config.use_int8_w8a16,
        use_int4_w4a16=experts.quant_config.use_int4_w4a16,
        block_shape=experts.block_shape,
        B_cold=cold_w1,
        B_scale_cold=cold_s1,
        expert_hot_map=hot_map,
        expert_cold_map=cold_map,
    )
    experts.activation(activation, cache2, cache1.view(-1, intermediate_size))
    quantized_cache2, _ = moe_kernel_quantize_input(
        cache2,
        a2_scale,
        experts.quant_dtype,
        experts.per_act_token_quant,
        experts.block_shape,
    )
    invoke_fused_moe_wna16_triton_kernel(
        quantized_cache2,
        w2,
        cache3,
        s2,
        None,
        topk_weights,
        sorted_ids,
        expert_ids,
        padded_tokens,
        not apply_router_weight_on_input,
        1,
        config,
        compute_type=tl.bfloat16,
        use_int8_w8a16=experts.quant_config.use_int8_w8a16,
        use_int4_w4a16=experts.quant_config.use_int4_w4a16,
        block_shape=experts.block_shape,
        B_cold=cold_w2,
        B_scale_cold=cold_s2,
        expert_hot_map=hot_map,
        expert_cold_map=cold_map,
    )
    experts.moe_sum(cache3, output)


def apply_split(
    experts: Any,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    global_num_experts: int,
    a1q_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    apply_router_weight_on_input: bool,
) -> None:
    """Run all selected experts through one exact hot/cold kernel."""
    del global_num_experts, a1q_scale
    tier = experts._tier
    if experts.quant_config.w1_zp is not None or experts.quant_config.w2_zp is not None:
        raise RuntimeError("static hot tier requires symmetric WNA16 weights")
    _apply_once(
        experts,
        output,
        hidden_states,
        tier["hot_w1"],
        tier["hot_w2"],
        tier["hot_s1"],
        tier["hot_s2"],
        topk_weights,
        topk_ids,
        activation,
        tier["global_num_experts"],
        None,
        a2_scale,
        workspace13,
        workspace2,
        apply_router_weight_on_input,
        cold_w1=w1,
        cold_w2=w2,
        cold_s1=experts.w1_scale,
        cold_s2=experts.w2_scale,
        hot_map=tier["hot_map"],
        cold_map=tier["cold_map"],
    )
