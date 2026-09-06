"""A bounded search primitive, never a lateral one-metre navigation goal."""

import math
import numpy as np


def forward_search_arc(sign, *, length_m=0.30, radius_m=0.40):
    """Opt-in forward arc. Front depth alone does not certify side clearance."""
    if sign not in (-1, 1):
        raise ValueError("search sign must be -1 or 1")
    if not (math.isfinite(length_m) and math.isfinite(radius_m)
            and 0 < length_m <= 0.30 and radius_m >= 0.40):
        raise ValueError("search exceeds the bounded execution envelope")
    theta = np.linspace(0.0, length_m / radius_m, 12)
    return np.column_stack((radius_m * np.sin(theta),
                            sign * radius_m * (1.0 - np.cos(theta)),
                            np.zeros_like(theta))).astype(np.float32)
