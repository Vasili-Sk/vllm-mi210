# vLLM MI210 Flash-Next

This repository is an experimental vLLM fork for Qwen Flash-Next on AMD Instinct MI210.

The `flash-next-offloaded` branch adds disk-backed PLE, ROCm QSA fixes, AITER attention fallback, WNA16 safety fixes, and static expert residency.

## Recommended MI210 settings

| Setting | Recommended value | Purpose |
|---|---:|---|
| `ROCM_PATH` | `/opt/rocm` | Select the ROCm installation. |
| `VLLM_TARGET_DEVICE` | `rocm` | Build and run the ROCm backend. |
| `VLLM_ROCM_USE_AITER` | `1` | Enable supported AITER ROCm kernels. |
| `VLLM_ROCM_USE_AITER_MHA` | `0` | Do not replace head-dimension-256 full attention. |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:128` | Reduce large inactive allocator blocks. |
| `GPU_PINNED_MIN_XFER_SIZE` | `67108864` | Use pinned transfers for large host copies. |
| `HSA_NO_SCRATCH_RECLAIM` | `1` | Keep ROCm scratch memory stable. |
| `HIP_FORCE_DEV_KERNARG` | `1` | Use the tested HIP kernel-argument path. |
| `VLLM_PLE_MMAP` | `1` | Keep the large PLE table in model files. |
| `VLLM_QSA_SORT_BLOCKS` | `1` | Keep multirow QSA block order stable. |
| `VLLM_WNA16_HOT_TIER_SIZE` | `424` | Keep 424 of 512 experts per layer in VRAM. |
| `VLLM_WNA16_HOT_TIER_FILE` | Ranking JSON path | Select the expert order for each layer. |
| `VLLM_WNA16_HOT_TIER_COMPACT_UVA` | `0` | Reuse full UVA tensors when host RAM permits. |
| `--tensor-parallel-size` | `1` | Use one MI210. |
| `--kv-cache-dtype` | `bfloat16` | Use the tested KV-cache format. |
| `--max-model-len` | `8256` | Support an 8,192-token prompt plus output. |
| `--max-num-batched-tokens` | `4096` | Split 8K prefill into two chunks. |
| `--max-num-seqs` | `1` | Limit graph and KV memory use. |
| `--gpu-memory-utilization` | `0.99` | Reserve most MI210 memory for vLLM. |
| `--safetensors-load-strategy` | `lazy` | Load local safetensors when needed. |
| `--offload-backend` | `uva` | Let the GPU address pinned host tensors. |
| `--cpu-offload-gb` | `10` | Set the expert offload budget. |
| `--cpu-offload-params` | `experts` | Offload expert tensors only. |
| `--compilation-config` | `FULL_DECODE_ONLY` | Capture full decode graphs. |
| Prefix caching | Disabled | Make validation run the full prefill path. |

## Start script

Save this file as `serve-flash-next-mi210.sh`.

Set `MODEL` and `RANKINGS` before you run it.

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to the local Flash-Next model directory}"
: "${RANKINGS:?Set RANKINGS to the expert-ranking JSON file}"

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export VLLM_TARGET_DEVICE=rocm

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=0

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export GPU_PINNED_MIN_XFER_SIZE=67108864
export HSA_NO_SCRATCH_RECLAIM=1
export HIP_FORCE_DEV_KERNARG=1

export VLLM_PLE_MMAP=1
unset VLLM_PLE_CPU_OFFLOAD

export VLLM_QSA_SORT_BLOCKS=1

export VLLM_WNA16_HOT_TIER_SIZE=424
export VLLM_WNA16_HOT_TIER_FILE="$RANKINGS"
export VLLM_WNA16_HOT_TIER_COMPACT_UVA=0

template_args=()
if [[ -f "$MODEL/chat_template.jinja" ]]; then
    template_args=(--chat-template "$MODEL/chat_template.jinja")
fi

exec vllm serve "$MODEL" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --served-model-name "${SERVED_MODEL_NAME:-qwen-flash-next}" \
    --tensor-parallel-size 1 \
    --kv-cache-dtype bfloat16 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --max-model-len 8256 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.99 \
    --safetensors-load-strategy lazy \
    --offload-backend uva \
    --cpu-offload-gb 10 \
    --cpu-offload-params experts \
    --no-enable-prefix-caching \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    "${template_args[@]}"
```

Run it with:

```bash
MODEL=/path/to/model \
RANKINGS=/path/to/expert-rankings.json \
./serve-flash-next-mi210.sh
```

The script does not select a GPU. Set your normal ROCm device variables outside the script when the system has more than one GPU.

## What this fork adds

### Disk-backed PLE

`VLLM_PLE_MMAP=1` keeps the PLE table in read-only safetensors mappings.

The runtime gathers the required rows into fixed host and GPU staging buffers.

The old `VLLM_PLE_CPU_OFFLOAD=1` mode keeps the full PLE table in pinned host memory. It remains available for compatibility.

### ROCm QSA

The ROCm QSA path uses AITER Triton attention when ROCm `flash-attn` is not available.

`VLLM_QSA_SORT_BLOCKS=1` sorts multirow QSA selections. This fixes unstable prefill ordering.

The sparse-attention split-K path also keeps one reduction layout through eight rows.

### WNA16 safety

The ROCm Triton WNA16 path adds a zero W2 guard row for 512-expert layers.

The logical expert count stays 512.

The gfx9 packed layout is limited to validated group-128 weights.

### Static expert residency

The hot tier keeps ranked experts in MI210 memory.

Cold experts stay in pinned host memory.

The Triton kernel runs every selected expert. It does not drop cold routes.

The ranking file must contain one complete expert permutation for each managed layer.

Example format:

```json
{
  "0": [0, 1, 2, 3],
  "1": [3, 1, 0, 2]
}
```

A 512-expert model needs every ID from `0` through `511` exactly once in each layer entry.

### Immutable expert refill

The repository contains a safetensors refill helper.

It can rebuild one GPTQ expert in the Triton runtime layout.

Adaptive expert migration is not connected yet.

## Model requirements

The current runtime expects:

- A local Qwen Flash-Next checkpoint.
- A local `model.safetensors.index.json`.
- All referenced safetensors shards.
- W4A16 routed experts.
- Symmetric WNA16 weights for the hot tier.
- FP8 or supported PLE tensors.
- A complete expert-ranking JSON file.
- A compatible external AITER installation.

## Tested configuration

The current end-to-end test used:

- One AMD Instinct MI210.
- ROCm 7.2.
- 48 managed MoE layers.
- 512 experts per layer.
- 424 resident experts per layer.
- 88 host-backed experts per layer.
- Disk-backed PLE.
- Full decode graphs.
- An 8,256-token model limit.
- A 4,096-token prefill chunk size.

The server completed model loading, graph capture, health checks, and text generation.

## Branches

| Branch | Purpose |
|---|---|
| `main` | Clean vLLM upstream line. |
| `vendor/davetha-mi210.7` | Imported davetha MI210 base. |
| `mi210/main` | MI210 base branch. |
| `flash-next-offloaded` | Flash-Next offload development branch. |

## Important source work

- [vLLM PR #54371](https://github.com/vllm-project/vllm/pull/54371): UVA PLE offload and Engram sharding.
- [vLLM PR #57497](https://github.com/vllm-project/vllm/pull/57497): ROCm PLE host offload.
- [vLLM PR #54129](https://github.com/vllm-project/vllm/pull/54129): mmap-backed PLE design.

PR #54129 was adapted to the shared PLE code and the AMD model path.

## Known limits

- This is not an official vLLM release.
- The tested target is one MI210.
- Multi-GPU Flash-Next offload is not validated.
- Adaptive expert migration is not connected.
- MTP is not part of the current public runtime path.
- AITER is external and must be installed separately.
- Memory values must be retested for other models and GPUs.

## Upstream

This fork is based on [vLLM](https://github.com/vllm-project/vllm).

Use the [official vLLM documentation](https://docs.vllm.ai) for general installation and API information.

The repository keeps the upstream vLLM license and source notices.
