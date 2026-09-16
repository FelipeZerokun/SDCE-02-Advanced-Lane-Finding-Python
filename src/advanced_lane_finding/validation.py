from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from advanced_lane_finding.config import (
    MeasurementConfig,
    PerspectiveConfig,
    ValidationConfig,
)
from advanced_lane_finding.detector import (
    LaneFit,
    evaluate_lane_fit,
)
from advanced_lane_finding.measurements import (
    calculate_pixel_scale,
)


class LaneValidationError(RuntimeError):
    """Raised when detected lane geometry is implausible."""


@dataclass(frozen=True, slots=True)
class LaneValidationResult:
    top_width_m: float
    middle_width_m: float
    bottom_width_m: float

    def __post_init__(self) -> None:
        widths = (
            self.top_width_m,
            self.middle_width_m,
            self.bottom_width_m,
        )

        if any(not isfinite(width) or width <= 0 for width in widths):
            raise LaneValidationError("Lane widths must be positive and finite")

    @property
    def minimum_width_m(self) -> float:
        return min(
            self.top_width_m,
            self.middle_width_m,
            self.bottom_width_m,
        )

    @property
    def maximum_width_m(self) -> float:
        return max(
            self.top_width_m,
            self.middle_width_m,
            self.bottom_width_m,
        )

    @property
    def width_variation_m(self) -> float:
        return self.maximum_width_m - self.minimum_width_m


def validate_lane_fit(
    fit: LaneFit,
    image_size: tuple[int, int],
    perspective: PerspectiveConfig,
    measurement: MeasurementConfig,
    validation: ValidationConfig,
) -> LaneValidationResult:
    """Validate lane width at the top, middle, and bottom."""

    _, height = image_size

    scale = calculate_pixel_scale(
        image_size,
        perspective,
        measurement,
    )

    y_values = np.asarray(
        [
            0.0,
            (height - 1) / 2.0,
            float(height - 1),
        ],
        dtype=np.float64,
    )

    left_x, right_x = evaluate_lane_fit(
        fit,
        y_values,
    )

    widths_m = (right_x - left_x) * scale.x_m_per_pixel

    if not np.all(np.isfinite(widths_m)):
        raise LaneValidationError("Lane width samples must be finite")

    if np.any(widths_m <= 0):
        raise LaneValidationError(
            "Right lane boundary must remain to the right of the left boundary"
        )

    result = LaneValidationResult(
        top_width_m=float(widths_m[0]),
        middle_width_m=float(widths_m[1]),
        bottom_width_m=float(widths_m[2]),
    )

    near_field_widths = (
        result.middle_width_m,
        result.bottom_width_m,
    )

    if min(near_field_widths) < validation.minimum_lane_width_m:
        raise LaneValidationError(
            "Lane width is below the configured minimum: "
            f"top={result.top_width_m:.3f} m, "
            f"middle={result.middle_width_m:.3f} m, "
            f"bottom={result.bottom_width_m:.3f} m "
            f"(limit={validation.minimum_lane_width_m:.3f} m)"
        )

    if max(near_field_widths) > validation.maximum_lane_width_m:
        raise LaneValidationError("Lane width is above the configured maximum")

    if result.width_variation_m > validation.maximum_width_variation_m:
        raise LaneValidationError("Lane width variation exceeds the configured maximum")

    return result
