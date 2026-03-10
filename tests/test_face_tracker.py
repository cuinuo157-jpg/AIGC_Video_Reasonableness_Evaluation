import numpy as np

from src.face_identity.face_tracker import FaceTracker


def _make_frame_data(n_frames, n_faces=1):
    base_embs = []
    for _ in range(n_faces):
        e = np.random.rand(512).astype(np.float32)
        base_embs.append(e / np.linalg.norm(e))

    frames = []
    for _ in range(n_frames):
        faces = []
        for j in range(n_faces):
            e = base_embs[j] + np.random.randn(512).astype(np.float32) * 0.01
            e /= np.linalg.norm(e)
            faces.append(
                {"embedding": e, "bbox": [10 + j * 60, 10, 50 + j * 60, 50], "det_score": 0.95}
            )
        frames.append({"faces": faces, "num_faces": n_faces})
    return frames


def test_tracker_single_face():
    tracks = FaceTracker().track(_make_frame_data(10, 1))
    assert len(tracks) == 1
    assert len(tracks[0].embeddings) == 10


def test_tracker_multi_face():
    tracks = FaceTracker().track(_make_frame_data(10, 2))
    assert len(tracks) == 2


def test_tracker_no_faces():
    data = [{"faces": [], "num_faces": 0} for _ in range(5)]
    assert len(FaceTracker().track(data)) == 0
