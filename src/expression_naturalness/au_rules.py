from __future__ import annotations

from dataclasses import dataclass

NATURAL_EXPRESSIONS = {
    "genuine_smile": {"required": ["AU06", "AU12"], "forbidden": []},
    "surprise": {
        "required": ["AU01", "AU02", "AU05", "AU26"],
        "forbidden": [],
    },
    "frown": {"required": ["AU04"], "forbidden": ["AU12"]},
    "fear": {
        "required": ["AU01", "AU02", "AU04", "AU20"],
        "forbidden": [],
    },
}

CONFLICT_PAIRS = [
    (["AU01", "AU02"], ["AU04"]),
    (["AU23"], ["AU26"]),
]

AU_ACTIVATION_THRESHOLD = 1.0


@dataclass
class Violation:
    violation_type: str
    description: str
    involved_aus: list[str]


def check_au_combination(aus: dict[str, float]) -> list[Violation]:
    violations = []
    active = {k for k, v in aus.items() if v >= AU_ACTIVATION_THRESHOLD}
    for group_a, group_b in CONFLICT_PAIRS:
        a_active = any(au in active for au in group_a)
        b_active = any(au in active for au in group_b)
        if a_active and b_active:
            violations.append(
                Violation(
                    violation_type="conflict",
                    description=f"Conflicting AUs: {group_a} vs {group_b}",
                    involved_aus=group_a + group_b,
                )
            )
    return violations
