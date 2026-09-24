# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import torch
from safetensors.torch import save_file

from vllm.model_executor.layers.fused_moe.experts.expert_mmap_backing import (
    ExpertMmapBacking,
)


def test_expert_mmap_backing_repacks_checkpoint_rows(tmp_path) -> None:
    prefix = "model.language_model.layers.0.mlp.experts.0"
    tensors = {
        f"{prefix}.gate_proj.qweight": torch.arange(6, dtype=torch.int32).reshape(2, 3),
        f"{prefix}.up_proj.qweight": torch.arange(6, 12, dtype=torch.int32).reshape(
            2, 3
        ),
        f"{prefix}.down_proj.qweight": torch.arange(12, 20, dtype=torch.int32).reshape(
            4, 2
        ),
        f"{prefix}.gate_proj.scales": torch.arange(3, dtype=torch.bfloat16).reshape(
            1, 3
        ),
        f"{prefix}.up_proj.scales": torch.arange(3, 6, dtype=torch.bfloat16).reshape(
            1, 3
        ),
        f"{prefix}.down_proj.scales": torch.arange(2, dtype=torch.bfloat16).reshape(
            1, 2
        ),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}})
    )

    backing = ExpertMmapBacking(tmp_path, cache_mib=0)
    w1 = torch.empty((6, 8), dtype=torch.uint8)
    w2 = torch.empty((2, 16), dtype=torch.uint8)
    s1 = torch.empty((6, 1), dtype=torch.bfloat16)
    s2 = torch.empty((2, 1), dtype=torch.bfloat16)
    backing.refill(0, 0, w1=w1, w2=w2, s1=s1, s2=s2)

    torch.testing.assert_close(
        w1.view(torch.int32)[:3], tensors[f"{prefix}.gate_proj.qweight"].T
    )
    torch.testing.assert_close(
        w1.view(torch.int32)[3:], tensors[f"{prefix}.up_proj.qweight"].T
    )
    torch.testing.assert_close(
        w2.view(torch.int32), tensors[f"{prefix}.down_proj.qweight"].T
    )
    torch.testing.assert_close(s1[:3], tensors[f"{prefix}.gate_proj.scales"].T)
    torch.testing.assert_close(s1[3:], tensors[f"{prefix}.up_proj.scales"].T)
    torch.testing.assert_close(s2, tensors[f"{prefix}.down_proj.scales"].T)
    assert backing.cached_bytes == 0
    assert not backing.lru

    backing.close()
    assert not backing.handles
    assert not backing.raw_files
