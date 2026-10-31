#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_PREFIX=${ADAPTIVE_WAM_ENV_PREFIX:-"$PROJECT_ROOT/.conda/fastwam"}
GPU_IDS=${1:-0,1,2,3}
MODEL="$PROJECT_ROOT/checkpoints/fastwam_release/libero_optional_idm_2cam224.pt"
STATS="$PROJECT_ROOT/checkpoints/fastwam_release/libero_optional_idm_2cam224_dataset_stats.json"
OUTPUT="$PROJECT_ROOT/results/oracle_decoupled_core_v1.csv"
ANALYSIS="$PROJECT_ROOT/results/oracle_decoupled_v1"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [[ ${#GPU_ARRAY[@]} -ne 4 ]]; then
  echo "Expected four comma-separated GPU IDs, got: $GPU_IDS" >&2
  exit 2
fi
if [[ $(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l) -ne 4 ]]; then
  echo "GPU IDs must be distinct: $GPU_IDS" >&2
  exit 2
fi
for gpu in "${GPU_ARRAY[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID: $gpu" >&2; exit 2; }
done

[[ -x "$ENV_PREFIX/bin/python" ]] || { echo "Missing environment: $ENV_PREFIX" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "Missing model: $MODEL" >&2; exit 1; }
[[ -f "$STATS" ]] || { echo "Missing dataset statistics: $STATS" >&2; exit 1; }
for path in "$PROJECT_ROOT"/snapshots_decoupled_core_v1/*.pkl; do
  [[ -e "$path" ]] || { echo "Broken snapshot link: $path" >&2; exit 1; }
done

"$ENV_PREFIX/bin/python" "$PROJECT_ROOT/scripts/configure_libero_paths.py"

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="$PROJECT_ROOT/checkpoints/diffsynth"
export LIBERO_CONFIG_PATH="$PROJECT_ROOT/.runtime/libero"
export PYTHONPATH="$PROJECT_ROOT/repos/FastWAM/src:$PROJECT_ROOT/repos/FastWAM/experiments/libero:$PROJECT_ROOT/repos/LIBERO${PYTHONPATH:+:$PYTHONPATH}"

"$ENV_PREFIX/bin/python" "$PROJECT_ROOT/oracle/counterfactual_oracle.py" \
  --snapshots-dir "$PROJECT_ROOT/snapshots_decoupled_core_v1" \
  --config-grid "$PROJECT_ROOT/configs/oracle_decoupled_grid.json" \
  --cost-table "$PROJECT_ROOT/configs/hardware_cost_table.json" \
  --model-path "$MODEL" \
  --eval-mode sustained \
  --output-csv "$OUTPUT" \
  --num-repeats 1 \
  --gpu-ids "$GPU_IDS" \
  --resume

"$ENV_PREFIX/bin/python" -m oracle.analyze_decoupled \
  --results-csv "$OUTPUT" \
  --output-dir "$ANALYSIS"

echo "Decoupled experiment complete: $OUTPUT"
