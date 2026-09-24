#!/usr/bin/env bash
# Build a small MI210 binary kit for this vLLM source revision.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  tools/build-mi210-binary-kit.sh \
    --build-python /path/to/python3.12 \
    --aiter-wheel /path/to/amd_aiter-0.1.21-cp312-cp312-linux_x86_64.whl \
    --aiter-site-packages /path/to/site-packages \
    --ranking /path/to/expert-ranking.json \
    [--triton-kernels-src /path/to/triton_kernels] \
    [--vllm-wheel /path/to/vllm-wheel.whl \
     --wheel-source-commit COMMIT] \
    [--output-dir ./bins] [--scratch-dir /path/outside/repository] \
    [--rocm-path /opt/rocm-7.2.1] [--jobs 16] \
    [--version 0.28.1rc0+mi210.flashnext]

The AITER site-packages directory must contain aiter/jit/*.so and
aiter/ops/triton/configs/gfx90a. The script copies only these tested runtime
files. It does not copy AITER build caches or source trees.

Set MI210_LOCK to a command that wraps the wheel build when this machine needs
a GPU mutex. Example: MI210_LOCK="$HOME/mi210_lock --note release-wheel".
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
BUILD_PYTHON=
AITER_WHEEL=
AITER_SITE=
RANKING=
TRITON_KERNELS_SRC_DIR=${TRITON_KERNELS_SRC_DIR:-}
VLLM_WHEEL=
WHEEL_SOURCE_COMMIT=
OUTPUT_DIR="$REPO/bins"
SCRATCH_DIR=
ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.1}
JOBS=${MAX_JOBS:-16}
PACKAGE_VERSION=${VLLM_VERSION_OVERRIDE:-0.28.1rc0+mi210.flashnext}

while (($#)); do
    case "$1" in
        --build-python) BUILD_PYTHON=$2; shift 2 ;;
        --aiter-wheel) AITER_WHEEL=$2; shift 2 ;;
        --aiter-site-packages) AITER_SITE=$2; shift 2 ;;
        --ranking) RANKING=$2; shift 2 ;;
        --triton-kernels-src) TRITON_KERNELS_SRC_DIR=$2; shift 2 ;;
        --vllm-wheel) VLLM_WHEEL=$2; shift 2 ;;
        --wheel-source-commit) WHEEL_SOURCE_COMMIT=$2; shift 2 ;;
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --scratch-dir) SCRATCH_DIR=$2; shift 2 ;;
        --rocm-path) ROCM_PATH=$2; shift 2 ;;
        --jobs) JOBS=$2; shift 2 ;;
        --version) PACKAGE_VERSION=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ -x "$BUILD_PYTHON" ]] || die "--build-python must name an executable Python."
[[ -f "$AITER_WHEEL" ]] || die "--aiter-wheel does not exist: $AITER_WHEEL"
[[ -d "$AITER_SITE/aiter" ]] || die "--aiter-site-packages has no aiter package."
[[ -f "$RANKING" ]] || die "--ranking does not exist: $RANKING"
[[ -d "$ROCM_PATH" ]] || die "ROCm does not exist: $ROCM_PATH"
[[ $(uname -m) == x86_64 ]] || die "The build host must use x86-64."
[[ $($BUILD_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")') == 3.12 ]] || \
    die "The build Python must be Python 3.12."
[[ -d "$AITER_SITE/aiter/ops/triton/configs/gfx90a" ]] || \
    die "The AITER installation has no gfx90a configuration directory."
compgen -G "$AITER_SITE/aiter/jit/*.so" >/dev/null || \
    die "The AITER installation has no prebuilt JIT modules."

git -C "$REPO" diff --quiet --ignore-submodules -- || \
    die "Commit tracked source changes before packaging."
git -C "$REPO" diff --cached --quiet --ignore-submodules -- || \
    die "Commit staged source changes before packaging."

BUILDER_COMMIT=$(git -C "$REPO" rev-parse HEAD)
if [[ -n "$VLLM_WHEEL" ]]; then
    [[ -n "$WHEEL_SOURCE_COMMIT" ]] || \
        die "--wheel-source-commit is required with --vllm-wheel."
    COMMIT=$(git -C "$REPO" rev-parse "$WHEEL_SOURCE_COMMIT^{commit}") || \
        die "The wheel source commit is not in this repository."
else
    COMMIT=$BUILDER_COMMIT
fi
SHORT_COMMIT=${COMMIT:0:10}
KIT_NAME="vllm-mi210-flash-next-${SHORT_COMMIT}"
if [[ -z "$SCRATCH_DIR" ]]; then
    SCRATCH_DIR="${TMPDIR:-/tmp}/$KIT_NAME-build"
fi
case "$(realpath -m "$SCRATCH_DIR")/" in
    "$(realpath "$REPO")/"*) die "Scratch data must stay outside the Git repository." ;;
esac
mkdir -p "$OUTPUT_DIR"
rm -rf "$SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"/{source,wheels,stage}

if [[ -z "$VLLM_WHEEL" ]]; then
    [[ -d "$TRITON_KERNELS_SRC_DIR" ]] || \
        die "Use --triton-kernels-src with a local ROCm Triton triton_kernels directory."
    printf 'Create a clean source tree for %s.\n' "$COMMIT"
    git -C "$REPO" archive "$COMMIT" | tar -x -C "$SCRATCH_DIR/source"

    build_command=(env
        "PATH=$ROCM_PATH/bin:$PATH"
        "ROCM_PATH=$ROCM_PATH"
        "ROCM_HOME=$ROCM_PATH"
        "HIP_PATH=$ROCM_PATH"
        "CMAKE_PREFIX_PATH=$ROCM_PATH"
        VLLM_TARGET_DEVICE=rocm
        PYTORCH_ROCM_ARCH=gfx90a
        CMAKE_BUILD_TYPE=Release
        CARGO_PROFILE_RELEASE_STRIP=symbols
        "VLLM_VERSION_OVERRIDE=$PACKAGE_VERSION"
        "MAX_JOBS=$JOBS"
        "TRITON_KERNELS_SRC_DIR=$TRITON_KERNELS_SRC_DIR"
        "$BUILD_PYTHON" -m pip wheel --no-build-isolation --no-deps
        --wheel-dir "$SCRATCH_DIR/wheels"
        "$SCRATCH_DIR/source")

    printf 'Build the gfx90a vLLM wheel. Output remains visible.\n'
    if [[ -n ${MI210_LOCK:-} ]]; then
        # MI210_LOCK is intentionally split into words. Do not place shell syntax in it.
        read -r -a lock_command <<<"$MI210_LOCK"
        "${lock_command[@]}" "${build_command[@]}" 2>&1 | tee "$SCRATCH_DIR/build-vllm-wheel.log"
    else
        "${build_command[@]}" 2>&1 | tee "$SCRATCH_DIR/build-vllm-wheel.log"
    fi
    VLLM_WHEEL=$(find "$SCRATCH_DIR/wheels" -maxdepth 1 -type f -name 'vllm-*.whl' -print -quit)
else
    [[ -f "$VLLM_WHEEL" ]] || die "--vllm-wheel does not exist: $VLLM_WHEEL"
fi
[[ -n "$VLLM_WHEEL" && -f "$VLLM_WHEEL" ]] || die "The vLLM wheel build produced no wheel."

STAGE="$SCRATCH_DIR/stage/$KIT_NAME"
mkdir -p "$STAGE"/{wheels,runtime/aiter/jit,runtime/aiter/ops/triton/configs,rankings,licenses}
STRIP_TOOL="$ROCM_PATH/llvm/bin/llvm-strip"
[[ -x "$STRIP_TOOL" ]] || die "ROCm llvm-strip does not exist: $STRIP_TOOL"
STRIPPED_VLLM_WHEEL="$STAGE/wheels/$(basename "$VLLM_WHEEL")"
"$BUILD_PYTHON" - "$VLLM_WHEEL" "$STRIPPED_VLLM_WHEEL" "$STRIP_TOOL" <<'PY'
from base64 import urlsafe_b64encode
import csv
from hashlib import sha256
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

source, output, strip_tool = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
records = []
stripped = []
with ZipFile(source) as src, ZipFile(
    output, "w", compression=ZIP_DEFLATED, compresslevel=6
) as dst, tempfile.TemporaryDirectory(dir=output.parent) as temporary:
    record_names = [
        item.filename
        for item in src.infolist()
        if item.filename.endswith(".dist-info/RECORD")
    ]
    if len(record_names) != 1:
        raise SystemExit("The vLLM wheel must contain one RECORD file.")
    record_name = record_names[0]
    for index, item in enumerate(src.infolist()):
        name = item.filename
        if name == record_name:
            continue
        data = src.read(name)
        if data.startswith(b"\x7fELF"):
            path = Path(temporary) / str(index)
            path.write_bytes(data)
            path.chmod(0o755)
            subprocess.run(
                [strip_tool, "--strip-debug", str(path)], check=True
            )
            data = path.read_bytes()
            stripped.append(name)
        dst.writestr(item, data)
        if not name.endswith("/"):
            digest = urlsafe_b64encode(sha256(data).digest()).rstrip(b"=").decode()
            records.append((name, f"sha256={digest}", str(len(data))))
    rows = StringIO(newline="")
    writer = csv.writer(rows, lineterminator="\n")
    writer.writerows(records)
    writer.writerow((record_name, "", ""))
    dst.writestr(record_name, rows.getvalue().encode())
print(f"Removed debug sections from {len(stripped)} vLLM ELF files.")
PY
SLIM_AITER_WHEEL="$STAGE/wheels/$(basename "$AITER_WHEEL")"
"$BUILD_PYTHON" - "$AITER_WHEEL" "$SLIM_AITER_WHEEL" <<'PY'
from base64 import urlsafe_b64encode
import csv
from hashlib import sha256
from io import StringIO
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

source, output = map(Path, sys.argv[1:])
records = []
with ZipFile(source) as src, ZipFile(
    output, "w", compression=ZIP_DEFLATED, compresslevel=6
) as dst:
    record_names = [
        item.filename
        for item in src.infolist()
        if item.filename.endswith(".dist-info/RECORD")
    ]
    if len(record_names) != 1:
        raise SystemExit("The AITER wheel must contain one RECORD file.")
    record_name = record_names[0]
    for item in src.infolist():
        name = item.filename
        required_meta = (
            name.startswith("aiter_meta/csrc/")
            and not name.endswith((".co", ".png", ".md", ".MD"))
        )
        if (name.startswith("aiter_meta/") and not required_meta) or name == record_name:
            continue
        data = src.read(name)
        dst.writestr(item, data)
        if not name.endswith("/"):
            digest = urlsafe_b64encode(sha256(data).digest()).rstrip(b"=").decode()
            records.append((name, f"sha256={digest}", str(len(data))))
    rows = StringIO(newline="")
    writer = csv.writer(rows, lineterminator="\n")
    writer.writerows(records)
    writer.writerow((record_name, "", ""))
    dst.writestr(record_name, rows.getvalue().encode())
PY
cp -a "$AITER_SITE/aiter/jit/"*.so "$STAGE/runtime/aiter/jit/"
for module in "$STAGE/runtime/aiter/jit/"*.so; do
    "$STRIP_TOOL" --strip-debug "$module"
done
cp -a "$AITER_SITE/aiter/ops/triton/configs/gfx90a" \
    "$STAGE/runtime/aiter/ops/triton/configs/"
cp -a "$RANKING" "$STAGE/rankings/expert-ranking.json"
cp -a "$REPO/LICENSE" "$STAGE/licenses/vLLM-LICENSE"

"$BUILD_PYTHON" - "$AITER_WHEEL" "$STAGE/licenses/AITER-LICENSE" <<'PY'
from pathlib import Path
import sys, zipfile
wheel, output = map(Path, sys.argv[1:])
with zipfile.ZipFile(wheel) as archive:
    names = [n for n in archive.namelist()
             if n.endswith('.dist-info/licenses/LICENSE')]
    if not names:
        raise SystemExit('The AITER wheel has no distribution license.')
    output.write_bytes(archive.read(names[0]))
PY

cat >"$STAGE/install.sh" <<'INSTALL'
#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python3.12}
"$PYTHON" - <<'PY'
import platform, sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit('Python 3.12 is required.')
if platform.machine() != 'x86_64':
    raise SystemExit('Linux x86-64 is required.')
try:
    import torch
except ImportError:
    raise SystemExit('Install PyTorch for ROCm 7.2 before you run this installer.')
if not torch.version.hip or not torch.version.hip.startswith('7.2'):
    raise SystemExit(f'PyTorch must use ROCm 7.2. Found HIP {torch.version.hip!r}.')
PY
mapfile -t wheels < <(find "$ROOT/wheels" -maxdepth 1 -type f -name '*.whl' | sort)
((${#wheels[@]} == 2)) || { echo 'Expected two wheels.' >&2; exit 1; }
pip_options=()
[[ ${INSTALL_NO_DEPS:-0} != 1 ]] || pip_options+=(--no-deps)
"$PYTHON" -m pip install --force-reinstall "${pip_options[@]}" "${wheels[@]}"
SITE=$($PYTHON - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
cp -a "$ROOT/runtime/aiter/." "$SITE/aiter/"
"$PYTHON" - <<'PY'
import aiter, torch, vllm
import vllm._C, vllm._rocm_C
print('vLLM:', vllm.__version__)
print('PyTorch:', torch.__version__)
print('PyTorch HIP:', torch.version.hip)
print('AITER:', getattr(aiter, '__version__', '0.1.21'))
print('MI210 binary kit installation passed.')
PY
INSTALL
chmod 0755 "$STAGE/install.sh"

cat >"$STAGE/serve-flash-next-mi210.sh" <<'SERVE'
#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${MODEL:?Set MODEL to the local Flash-Next model directory}"
RANKINGS=${RANKINGS:-$ROOT/rankings/expert-ranking.json}
[[ -f "$RANKINGS" ]] || { echo "Ranking file not found: $RANKINGS" >&2; exit 1; }
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.1}
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
export VLLM_WNA16_HOT_TIER_SIZE=${VLLM_WNA16_HOT_TIER_SIZE:-424}
export VLLM_WNA16_HOT_TIER_FILE=$RANKINGS
export VLLM_WNA16_HOT_TIER_COMPACT_UVA=0
template_args=()
[[ ! -f "$MODEL/chat_template.jinja" ]] || template_args=(--chat-template "$MODEL/chat_template.jinja")
exec vllm serve "$MODEL" \
    --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" \
    --served-model-name "${SERVED_MODEL_NAME:-qwen-flash-next}" \
    --tensor-parallel-size 1 --kv-cache-dtype bfloat16 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --max-model-len "${MAX_MODEL_LEN:-8256}" \
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-4096}" \
    --max-num-seqs "${MAX_NUM_SEQS:-1}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.99}" \
    --safetensors-load-strategy lazy --offload-backend uva \
    --cpu-offload-gb "${CPU_OFFLOAD_GB:-10}" --cpu-offload-params experts \
    --no-enable-prefix-caching --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    "${template_args[@]}" "$@"
SERVE
chmod 0755 "$STAGE/serve-flash-next-mi210.sh"

VLLM_FILE=$(basename "$VLLM_WHEEL")
AITER_FILE=$(basename "$AITER_WHEEL")
cat >"$STAGE/README.md" <<EOF
# vLLM MI210 Flash-Next binary kit

This kit targets Ubuntu 24.04, Linux x86-64, Python 3.12, ROCm 7.2.1, and one AMD Instinct MI210.

Source commit: \`$COMMIT\`

Included files:

- \`wheels/$VLLM_FILE\`: vLLM wheel compiled for gfx90a. ELF debug sections are removed.
- \`wheels/$AITER_FILE\`: reduced AITER runtime. It keeps the import source and templates that AITER reads. It excludes bundled code objects, third-party build trees, heuristic models, and non-gfx90a data.
- \`runtime/aiter\`: tested gfx90a AITER modules and configuration files.
- \`rankings/expert-ranking.json\`: tested static expert order.
- \`install.sh\`: installer for an active Python 3.12 environment.
- \`serve-flash-next-mi210.sh\`: generic one-MI210 server launcher.

## Install

Install ROCm 7.2.1 and a ROCm 7.2 PyTorch build first. Create and activate a Python 3.12 virtual environment. Then run:

\`\`\`bash
./install.sh
\`\`\`

The installer downloads normal Python dependencies. It force-installs the two bundled wheels. It does not download another vLLM or AITER wheel.

Set \`INSTALL_NO_DEPS=1\` only when the active environment already has all vLLM and AITER dependencies.

## Start

\`\`\`bash
MODEL=/path/to/qwen-flash-next ./serve-flash-next-mi210.sh
\`\`\`

The launcher does not select a GPU. Set the normal ROCm device variables before the command when the host has more than one GPU.

The model files and ROCm/PyTorch runtime are not in this small kit.
EOF

cat >"$STAGE/MANIFEST.txt" <<EOF
Kit: $KIT_NAME
Runtime source commit: $COMMIT
Kit builder commit: $BUILDER_COMMIT
Build target: Ubuntu 24.04, Linux x86-64, Python 3.12, ROCm 7.2.1, gfx90a
Build type: Release; packaged ELF debug sections removed
Requested package version: $PACKAGE_VERSION
vLLM wheel: wheels/$VLLM_FILE
AITER wheel: wheels/$AITER_FILE
AITER wheel: bundled code objects, third-party build trees, heuristics, and unused aiter_meta removed
AITER runtime: prebuilt top-level JIT modules and gfx90a Triton configurations
Expert ranking: rankings/expert-ranking.json
EOF

(
    cd "$STAGE"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
)
ZIP="$OUTPUT_DIR/$KIT_NAME.zip"
rm -f "$ZIP"
"$BUILD_PYTHON" - "$STAGE" "$ZIP" <<'PY'
from pathlib import Path
import sys, zipfile
source, output = map(Path, sys.argv[1:])
with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED,
                     compresslevel=6) as archive:
    for path in sorted(source.rglob('*')):
        if path.is_file():
            archive.write(path, path.relative_to(source.parent))
PY
sha256sum "$ZIP" >"$ZIP.sha256"
printf '\nCreated:\n'
ls -lh "$ZIP" "$ZIP.sha256"
cat "$ZIP.sha256"
