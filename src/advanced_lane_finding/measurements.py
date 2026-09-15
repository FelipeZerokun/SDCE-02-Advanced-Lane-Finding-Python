from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from advanced_lane_finding.config import (
    MeasurementConfig,
    PerspectiveConfig,
)
from advanced_lane_finding.detector import FloatArray, LaneFit


class MeasurementError(ValueError):
    """Raised when physical lane measurements cannot be calculated."""


@dataclass(frozen=True, slots=True)
class PixelScale:
    x_m_per_pixel: float
    y_m_per_pixel: float

    def __post_init__(self) -> None:
        if not isfinite(self.x_m_per_pixel) or self.x_m_per_pixel <= 0:
            raise MeasurementError(
                "Horizontal meters-per-pixel scale must be positive and finite"
            )

        if not isfinite(self.y_m_per_pixel) or self.y_m_per_pixel <= 0:
            raise MeasurementError(
                "Vertical meters-per-pixel scale must be positive and finite"
            )


@dataclass(frozen=True, slots=True)
class LaneMeasurements:
    left_curvature_m: float
    right_curvature_m: float
    lane_width_m: float
    vehicle_offset_m: float

    def __post_init__(self) -> None:
        if self.left_curvature_m <= 0:
            raise MeasurementError("Left lane curvature must be positive")

        if self.right_curvature_m <= 0:
            raise MeasurementError("Right lane curvature must be positive")

        if not isfinite(self.lane_width_m) or self.lane_width_m <= 0:
            raise MeasurementError("Lane width must be positive and finite")

        if not isfinite(self.vehicle_offset_m):
            raise MeasurementError("Vehicle offset must be finite")


def calculate_pixel_scale(
    image_size: tuple[int, int],
    perspective: PerspectiveConfig,
    measurement: MeasurementConfig,
) -> PixelScale:
    """Calculate physical distance represented by one warped-image pixel."""

    width, height = image_size

    if width < 2 or height < 2:
        raise MeasurementError(
            "Measurement image dimensions must be at least two pixels"
        )

    bottom_left = perspective.destination[0]
    top_left = perspective.destination[1]
    top_right = perspective.destination[2]
    bottom_right = perspective.destination[3]

    horizontal_span = abs(bottom_right[0] - bottom_left[0])
    left_vertical_span = abs(bottom_left[1] - top_left[1])
    right_vertical_span = abs(bottom_right[1] - top_right[1])
    vertical_span = (left_vertical_span + right_vertical_span) / 2.0

    lane_width_pixels = horizontal_span * (width - 1)
    road_length_pixels = vertical_span * (height - 1)

    if lane_width_pixels <= 0:
        raise MeasurementError("Perspective destination lane width must be positive")

    if road_length_pixels <= 0:
        raise MeasurementError("Perspective destination road length must be positive")

    return PixelScale(
        x_m_per_pixel=(measurement.lane_width_m / lane_width_pixels),
        y_m_per_pixel=(measurement.visible_road_length_m / road_length_pixels),
    )


def curvature_radius_m(
    coefficients: FloatArray,
    evaluation_y_px: float,
    scale: PixelScale,
) -> float:
    """Calculate polynomial curvature radius in meters."""

    if coefficients.shape != (3,):
        raise MeasurementError("Lane polynomial must contain three coefficients")

    if not np.all(np.isfinite(coefficients)):
        raise MeasurementError("Lane polynomial must contain only finite values")

    if not isfinite(evaluation_y_px) or evaluation_y_px < 0:
        raise MeasurementError(
            "Curvature evaluation position must be nonnegative and finite"
        )

    a_px, b_px, _ = coefficients

    a_m = scale.x_m_per_pixel * float(a_px) / scale.y_m_per_pixel**2
    b_m = scale.x_m_per_pixel * float(b_px) / scale.y_m_per_pixel
    evaluation_y_m = evaluation_y_px * scale.y_m_per_pixel

    denominator = abs(2.0 * a_m)

    if denominator == 0.0:
        return float("inf")

    numerator = (1.0 + (2.0 * a_m * evaluation_y_m + b_m) ** 2) ** 1.5

    return numerator / denominator


def calculate_lane_measurements(
    fit: LaneFit,
    image_size: tuple[int, int],
    perspective: PerspectiveConfig,
    measurement: MeasurementConfig,
) -> LaneMeasurements:
    """Calculate curvature, lane width, and vehicle offset."""

    width, height = image_size
    scale = calculate_pixel_scale(
        image_size,
        perspective,
        measurement,
    )

    evaluation_y_px = float(height - 1)

    left_x_px = float(
        np.polyval(
            fit.left_coefficients,
            evaluation_y_px,
        )
    )
    right_x_px = float(
        np.polyval(
            fit.right_coefficients,
            evaluation_y_px,
        )
    )

    lane_width_px = right_x_px - left_x_px

    if lane_width_px <= 0:
        raise MeasurementError(
            "Right lane boundary must be to the right of the left boundary"
        )

    lane_width_m = lane_width_px * scale.x_m_per_pixel

    lane_center_px = (left_x_px + right_x_px) / 2.0
    vehicle_center_px = (width - 1) / 2.0

    vehicle_offset_m = (vehicle_center_px - lane_center_px) * scale.x_m_per_pixel

    left_curvature_m = curvature_radius_m(
        fit.left_coefficients,
        evaluation_y_px,
        scale,
    )
    right_curvature_m = curvature_radius_m(
        fit.right_coefficients,
        evaluation_y_px,
        scale,
    )

    return LaneMeasurements(
        left_curvature_m=left_curvature_m,
        right_curvature_m=right_curvature_m,
        lane_width_m=lane_width_m,
        vehicle_offset_m=vehicle_offset_m,
    )
