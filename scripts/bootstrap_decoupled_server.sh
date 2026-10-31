#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_PREFIX="$PROJECT_ROOT/.conda/fastwam"
DOWNLOAD_ASSETS=0
FASTWAM_COMMIT=7faa71108368fbb3b6885649f112af607427a2d4
LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
FASTWAM_PATCH="$PROJECT_ROOT/patches/FastWAM-exp-multigpu.patch"

while (($#)); do
  case "$1" in
    --env-prefix)
      ENV_PREFIX=$(realpath -m "$2")
      shift 2
      ;;
    --download-assets)
      DOWNLOAD_ASSETS=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--env-prefix PATH] [--download-assets]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v conda >/dev/null || { echo "conda is required" >&2; exit 1; }
mkdir -p "$PROJECT_ROOT/repos" "$PROJECT_ROOT/checkpoints"

if [[ ! -d "$PROJECT_ROOT/repos/FastWAM/.git" ]]; then
  git clone https://github.com/yuantianyuan01/FastWAM.git "$PROJECT_ROOT/repos/FastWAM"
  git -C "$PROJECT_ROOT/repos/FastWAM" checkout --detach "$FASTWAM_COMMIT"
fi

if git -C "$PROJECT_ROOT/repos/FastWAM" apply --reverse --check "$FASTWAM_PATCH" >/dev/null 2>&1; then
  echo "FastWAM patch is already applied"
else
  FASTWAM_HEAD=$(git -C "$PROJECT_ROOT/repos/FastWAM" rev-parse HEAD)
  if [[ "$FASTWAM_HEAD" != "$FASTWAM_COMMIT" ]]; then
    echo "FastWAM HEAD is $FASTWAM_HEAD; expected $FASTWAM_COMMIT before patching" >&2
    exit 1
  fi
  git -C "$PROJECT_ROOT/repos/FastWAM" apply --check "$FASTWAM_PATCH"
  git -C "$PROJECT_ROOT/repos/FastWAM" apply "$FASTWAM_PATCH"
fi

if [[ ! -d "$PROJECT_ROOT/repos/LIBERO/.git" ]]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$PROJECT_ROOT/repos/LIBERO"
  git -C "$PROJECT_ROOT/repos/LIBERO" checkout --detach "$LIBERO_COMMIT"
fi
LIBERO_HEAD=$(git -C "$PROJECT_ROOT/repos/LIBERO" rev-parse HEAD)
if [[ "$LIBERO_HEAD" != "$LIBERO_COMMIT" ]]; then
  echo "LIBERO HEAD is $LIBERO_HEAD; expected $LIBERO_COMMIT" >&2
  exit 1
fi

python3 "$PROJECT_ROOT/scripts/configure_libero_paths.py"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  CONDA_PKGS_DIRS="$PROJECT_ROOT/.conda/pkgs" conda create -p "$ENV_PREFIX" python=3.10 pip -y
fi

PYTHONNOUSERSITE=1 "$ENV_PREFIX/bin/python" -m pip install \
  -r "$PROJECT_ROOT/configs/baseline/fastwam-core-requirements.txt"
PYTHONNOUSERSITE=1 "$ENV_PREFIX/bin/python" -m pip install \
  -r "$PROJECT_ROOT/configs/decoupled-runtime-requirements.txt"
PYTHONNOUSERSITE=1 "$ENV_PREFIX/bin/python" -m pip install --no-deps -e "$PROJECT_ROOT/repos/FastWAM"
PYTHONNOUSERSITE=1 "$ENV_PREFIX/bin/python" -m pip install --no-deps -e "$PROJECT_ROOT/repos/LIBERO"
PYTHONNOUSERSITE=1 "$ENV_PREFIX/bin/python" -m pip check

if ((DOWNLOAD_ASSETS)); then
  PYTHONNOUSERSITE=1 "$ENV_PREFIX/bin/python" \
    "$PROJECT_ROOT/scripts/download_decoupled_assets.py"
fi

LIBERO_CONFIG_PATH="$PROJECT_ROOT/.runtime/libero" \
PYTHONPATH="$PROJECT_ROOT/repos/FastWAM/src:$PROJECT_ROOT/repos/LIBERO" \
PYTHONNOUSERSITE=1 "$ENV_PREFIX/bin/python" - <<'PY'
import fastwam
import libero.libero
import torch
print(f"environment ready: torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
PY

echo "Bootstrap complete. Run scripts/run_decoupled_experiment.sh when four GPUs are available."
