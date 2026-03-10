from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class FaceTrack:
    track_id: int
    embeddings: list[np.ndarray] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)


class FaceTracker:
    def __init__(self, match_threshold: float = 0.4) -> None:
        self.match_threshold = match_threshold

    def track(self, frame_data: list[dict]) -> list[FaceTrack]:
        tracks: list[FaceTrack] = []
        next_id = 0
        active_tracks: list[int] = []

        for frame_idx, fd in enumerate(frame_data):
            faces = fd.get("faces", [])
            if not faces:
                active_tracks = []
                continue

            if not active_tracks:
                for f in faces:
                    tracks.append(
                        FaceTrack(
                            track_id=next_id,
                            embeddings=[f["embedding"]],
                            frame_indices=[frame_idx],
                        )
                    )
                    active_tracks.append(len(tracks) - 1)
                    next_id += 1
                continue

            cost = np.zeros((len(active_tracks), len(faces)))
            for i, ti in enumerate(active_tracks):
                last_emb = tracks[ti].embeddings[-1]
                for j, f in enumerate(faces):
                    cost[i, j] = 1.0 - float(np.dot(last_emb, f["embedding"]))

            row_ind, col_ind = linear_sum_assignment(cost)

            matched_faces = set()
            new_active = []
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < (1.0 - self.match_threshold):
                    ti = active_tracks[r]
                    tracks[ti].embeddings.append(faces[c]["embedding"])
                    tracks[ti].frame_indices.append(frame_idx)
                    new_active.append(ti)
                    matched_faces.add(c)

            for j, f in enumerate(faces):
                if j not in matched_faces:
                    tracks.append(
                        FaceTrack(
                            track_id=next_id,
                            embeddings=[f["embedding"]],
                            frame_indices=[frame_idx],
                        )
                    )
                    new_active.append(len(tracks) - 1)
                    next_id += 1

            active_tracks = new_active

        return tracks
