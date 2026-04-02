from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoryCurvatureDetail:
    score: float = 1.0
    trajectory_count: int = 0
    valid_trajectory_count: int = 0
    abnormal_event_count: int = 0
    sample_count: int = 0
    abnormal_ratio: float = 0.0


def _mad(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _trajectory_events(trajectory: np.ndarray) -> tuple[int, int]:
    """Count abnormal curvature-rate events for a trajectory.

    Args:
        trajectory: (T, 2) normalized coordinates with NaN for invisible steps.

    Returns:
        (abnormal_events, candidate_samples)
    """
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        return 0, 0

    valid = np.all(np.isfinite(trajectory), axis=1)
    if int(np.sum(valid)) < 5:
        return 0, 0

    points = trajectory[valid]
    velocity = np.diff(points, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    if speed.size < 4:
        return 0, 0

    # Curvature proxy: turning angle between consecutive velocity vectors.
    v1 = velocity[:-1]
    v2 = velocity[1:]
    denom = (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)) + 1e-8
    cos_theta = np.sum(v1 * v2, axis=1) / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    curvature = np.arccos(cos_theta)  # [0, pi]
    if curvature.size < 3:
        return 0, 0

    curvature_rate = np.abs(np.diff(curvature))
    speed_ref = speed[1:]  # align with curvature sequence
    speed_for_rate = speed_ref[1:]  # align with curvature_rate
    if curvature_rate.size != speed_for_rate.size:
        n = min(curvature_rate.size, speed_for_rate.size)
        curvature_rate = curvature_rate[:n]
        speed_for_rate = speed_for_rate[:n]

    if curvature_rate.size == 0:
        return 0, 0

    # Robust thresholds; "teleport-like" events require both sudden turn-rate
    # change and high instantaneous speed.
    curv_med = float(np.median(curvature_rate))
    curv_mad = _mad(curvature_rate)
    speed_med = float(np.median(speed_for_rate))
    speed_mad = _mad(speed_for_rate)

    curv_thr = curv_med + max(6.0 * curv_mad, 0.25)
    speed_thr = speed_med + max(6.0 * speed_mad, 0.01)

    abnormal = (curvature_rate > curv_thr) & (speed_for_rate > speed_thr)
    return int(np.sum(abnormal)), int(curvature_rate.size)


def compute_trajectory_curvature_smoothness(
    trajectories: list[np.ndarray],
) -> tuple[float, TrajectoryCurvatureDetail]:
    """Compute trajectory smoothness from curvature-rate stability.

    A high score means trajectories are temporally coherent without
    teleport-like abrupt turning and speed spikes.
    """
    detail = TrajectoryCurvatureDetail(trajectory_count=len(trajectories))
    if not trajectories:
        return 1.0, detail

    total_events = 0
    total_samples = 0
    valid_count = 0
    for traj in trajectories:
        events, samples = _trajectory_events(traj)
        if samples > 0:
            valid_count += 1
            total_events += events
            total_samples += samples

    detail.valid_trajectory_count = valid_count
    detail.abnormal_event_count = total_events
    detail.sample_count = total_samples

    if total_samples == 0:
        detail.score = 1.0
        return 1.0, detail

    abnormal_ratio = total_events / total_samples
    score = float(np.clip(1.0 - 3.0 * abnormal_ratio, 0, 1))

    detail.abnormal_ratio = float(abnormal_ratio)
    detail.score = score
    return score, detail
