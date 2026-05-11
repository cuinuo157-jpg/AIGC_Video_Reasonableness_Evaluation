from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .video_frames import load_video_frames

_face_analyzer = None


def _get_face_analyzer(device: str) -> Any:
    global _face_analyzer
    if _face_analyzer is None:
        from insightface.app import FaceAnalysis

        _insightface_root = str(Path(__file__).resolve().parent.parent.parent.parent / "third_party" / "insightface")
        ctx_id = 0 if "cuda" in device else -1
        _face_analyzer = FaceAnalysis(
            name="buffalo_l",
            root=_insightface_root,
            providers=(
                ["CUDAExecutionProvider"] if ctx_id >= 0 else ["CPUExecutionProvider"]
            ),
        )
        _face_analyzer.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return _face_analyzer


def extract_face_embeddings(
    video_path: str,
    device: str,
    hub: object | None = None,
) -> list[dict]:
    if hub is not None:
        frames = hub.get("video_frames")
    else:
        frames = load_video_frames(video_path)
    analyzer = _get_face_analyzer(device)
    results = []
    for frame in frames:
        faces_raw = analyzer.get(frame)
        faces = []
        for f in faces_raw:
            faces.append(
                {
                    "embedding": f.embedding / np.linalg.norm(f.embedding),
                    "bbox": f.bbox.tolist(),
                    "det_score": float(f.det_score),
                }
            )
        results.append({"faces": faces, "num_faces": len(faces)})
    return results
