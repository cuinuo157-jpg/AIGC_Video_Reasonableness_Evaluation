#!/usr/bin/env bash
set -euo pipefail

# 统一调试命令清单（按需取消注释）
# 用法:
#   1) 先修改 VIDEO_PATH / DEVICE
#   2) 保持 1 条命令为激活状态，其余注释
#   3) 执行: bash scripts/debug_modules.sh

VIDEO_PATH="${VIDEO_PATH:-data/sample.mp4}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "VIDEO_PATH=${VIDEO_PATH}"
echo "DEVICE=${DEVICE}"

# ------------------ 当前激活命令（默认只保留 1 条） ------------------
# "${PYTHON_BIN}" scripts/debug_physics.py --input "${VIDEO_PATH}" --device "${DEVICE}" --enable-mllm

# ------------------ 其他模块命令（先注释，按需打开） ------------------
"${PYTHON_BIN}" scripts/debug_dynamics.py --input "${VIDEO_PATH}" --device "${DEVICE}" --enable-mllm
# "${PYTHON_BIN}" scripts/debug_temporal_coherence.py --input "${VIDEO_PATH}" --device "${DEVICE}" --enable-mllm
# "${PYTHON_BIN}" scripts/debug_bio_anomaly.py --input "${VIDEO_PATH}" --device "${DEVICE}"
# "${PYTHON_BIN}" scripts/debug_expression.py --input "${VIDEO_PATH}" --device "${DEVICE}"
# "${PYTHON_BIN}" scripts/debug_face_identity.py --input "${VIDEO_PATH}" --device "${DEVICE}"
# "${PYTHON_BIN}" scripts/debug_iris_tracking.py --input "${VIDEO_PATH}" --device "${DEVICE}" --save-vis
# "${PYTHON_BIN}" scripts/test_qwen_35_video.py
