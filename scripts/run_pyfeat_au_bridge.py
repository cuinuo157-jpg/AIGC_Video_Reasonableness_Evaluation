from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.expression_naturalness.au_extractor import run_local_au_sequence


def _load_frames(path: Path) -> list[np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        return [np.asarray(frame) for frame in payload.tolist()]
    return [np.asarray(frame) for frame in payload]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run py-feat AU extraction in an isolated Python environment")
    parser.add_argument("--frames-npy", required=True, help="Path to temporary .npy frames payload")
    parser.add_argument("--output-json", required=True, help="Path to write AU extraction JSON result")
    args = parser.parse_args()

    frames = _load_frames(Path(args.frames_npy))
    result = run_local_au_sequence(frames)
    Path(args.output_json).write_text(
        json.dumps(result, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
