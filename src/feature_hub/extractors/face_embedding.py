from __future__ import annotations

from typing import Any

import cv2
import numpy as np

_face_analyzer = None


def _load_frames(video_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _get_face_analyzer(device: str) -> Any:
    global _face_analyzer
    if _face_analyzer is None:
        from insightface.app import FaceAnalysis

        ctx_id = 0 if "cuda" in device else -1
        _face_analyzer = FaceAnalysis(
            name="buffalo_l",
            providers=(
                ["CUDAExecutionProvider"] if ctx_id >= 0 else ["CPUExecutionProvider"]
            ),
        )
        _face_analyzer.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return _face_analyzer


def extract_face_embeddings(video_path: str, device: str) -> list[dict]:
    frames = _load_frames(video_path)
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
