from dataclasses import dataclass


@dataclass
class BiologicalAnomalyConfig:
    ear_blink_threshold: float = 0.21
    max_no_blink_frames: int = 90
    eye_symmetry_tolerance: float = 0.15
    finger_count_expected: int = 5
    joint_angle_range: tuple[float, float] = (0, 180)
    bone_length_ratio_tolerance: float = 0.15
