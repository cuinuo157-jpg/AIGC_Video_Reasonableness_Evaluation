#!/usr/bin/env bash
set -euo pipefail

# Run debug_dynamics.py on 4 GPUs in parallel.
# Example:
#   bash scripts/run_4gpu_batch.sh --input-dir data/videos --subject --offline --save-vis

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="${REPO_ROOT}/scripts/debug_dynamics.py"

INPUT_DIR=""
METHOD="raft"
SUBJECT_FLAG=""
OFFLINE_FLAG=""
SAVE_VIS_FLAG=""
MAX_FRAMES="60"
MAX_SIDE="512"
GPU_IDS=("0" "1" "2" "3")

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_4gpu_batch.sh --input-dir <dir> [options]

Required:
  --input-dir <dir>      Directory containing videos.

Options:
  --method <name>        Optical flow method: raft|farneback (default: raft)
  --subject              Enable subject segmentation (SAM2 + GroundingDINO)
  --offline              Enforce offline mode
  --save-vis             Save visualization images
  --max-frames <int>     Max frames per video (default: 60)
  --max-side <int>       Max image long side (default: 512)
  --gpus <ids>           Comma-separated GPU ids (default: 0,1,2,3)
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --method)
      METHOD="$2"
      shift 2
      ;;
    --subject)
      SUBJECT_FLAG="--subject"
      shift
      ;;
    --offline)
      OFFLINE_FLAG="--offline"
      shift
      ;;
    --save-vis)
      SAVE_VIS_FLAG="--save-vis"
      shift
      ;;
    --max-frames)
      MAX_FRAMES="$2"
      shift 2
      ;;
    --max-side)
      MAX_SIDE="$2"
      shift 2
      ;;
    --gpus)
      IFS=',' read -r -a GPU_IDS <<< "$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${INPUT_DIR}" ]]; then
  echo "Error: --input-dir is required."
  usage
  exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Error: input directory not found: ${INPUT_DIR}"
  exit 1
fi

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "Error: script not found: ${SCRIPT_PATH}"
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${REPO_ROOT}/outputs/dynamics_batch/${RUN_ID}"
mkdir -p "${RUN_DIR}"

VIDEO_LIST="${RUN_DIR}/videos.txt"
find "${INPUT_DIR}" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.webm" \) | sort > "${VIDEO_LIST}"

TOTAL_VIDEOS="$(wc -l < "${VIDEO_LIST}" | tr -d ' ')"
if [[ "${TOTAL_VIDEOS}" -eq 0 ]]; then
  echo "No videos found in: ${INPUT_DIR}"
  exit 1
fi

GPU_COUNT="${#GPU_IDS[@]}"
echo "Found ${TOTAL_VIDEOS} videos, using ${GPU_COUNT} workers on GPUs: ${GPU_IDS[*]}"
echo "Run logs: ${RUN_DIR}"

for ((i=0; i<GPU_COUNT; i++)); do
  : > "${RUN_DIR}/shard_${i}.txt"
done

idx=0
while IFS= read -r video; do
  shard=$((idx % GPU_COUNT))
  echo "${video}" >> "${RUN_DIR}/shard_${shard}.txt"
  idx=$((idx + 1))
done < "${VIDEO_LIST}"

worker() {
  local worker_idx="$1"
  local gpu_id="$2"
  local shard_file="${RUN_DIR}/shard_${worker_idx}.txt"
  local log_file="${RUN_DIR}/worker_${worker_idx}_gpu${gpu_id}.log"

  echo "[worker-${worker_idx}] GPU=${gpu_id} start" | tee -a "${log_file}"

  while IFS= read -r video; do
    [[ -z "${video}" ]] && continue
    echo "[worker-${worker_idx}] processing: ${video}" | tee -a "${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python "${SCRIPT_PATH}" \
      --input "${video}" \
      --device cuda \
      --method "${METHOD}" \
      ${SUBJECT_FLAG} \
      ${OFFLINE_FLAG} \
      ${SAVE_VIS_FLAG} \
      --max-frames "${MAX_FRAMES}" \
      --max-side "${MAX_SIDE}" \
      >> "${log_file}" 2>&1
  done < "${shard_file}"

  echo "[worker-${worker_idx}] done" | tee -a "${log_file}"
}

pids=()
for ((i=0; i<GPU_COUNT; i++)); do
  worker "${i}" "${GPU_IDS[$i]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "Batch run finished with failures. Check logs under: ${RUN_DIR}"
  exit 1
fi

echo "Batch run finished successfully."
echo "Logs and shard lists are under: ${RUN_DIR}"
