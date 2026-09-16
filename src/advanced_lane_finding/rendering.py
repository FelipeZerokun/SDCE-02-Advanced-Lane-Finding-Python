from __future__ import annotations

from math import isinf

import cv2
import numpy as np
from numpy.typing import NDArray

from advanced_lane_finding.config import RenderConfig
from advanced_lane_finding.detector import LaneFit, evaluate_lane_fit
from advanced_lane_finding.measurements import LaneMeasurements
from advanced_lane_finding.perspective import (
    PerspectiveTransform,
    unwarp_color_image,
)

ColorImage = NDArray[np.uint8]


class RenderingError(ValueError):
    """Raised when a lane visualization cannot be rendered."""


def create_lane_polygon(
    fit: LaneFit,
    image_size: tuple[int, int],
    config: RenderConfig,
) -> ColorImage:
    """Create a filled lane area in bird's-eye coordinates."""
    width, height = image_size

    if width < 2 or height < 2:
        raise RenderingError("Rendering image dimensions must be at least two pixels")

    y_values = np.arange(height, dtype=np.float64)
    left_x, right_x = evaluate_lane_fit(fit, y_values)

    if not np.all(np.isfinite(left_x)) or not np.all(np.isfinite(right_x)):
        raise RenderingError("Lane polygon coordinates must be finite")

    if np.any(right_x <= left_x):
        raise RenderingError("Right lane boundary must remain right of the left")

    left_points = np.column_stack((left_x, y_values))
    right_points = np.column_stack((right_x, y_values))[::-1]

    polygon = np.asarray(
        np.rint(np.vstack((left_points, right_points))),
        dtype=np.int32,
    )

    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.fillPoly(overlay, [polygon], config.lane_color_bgr)

    return overlay


def project_lane_polygon(
    fit: LaneFit,
    transform: PerspectiveTransform,
    config: RenderConfig,
) -> ColorImage:
    """Project the filled lane area back into camera coordinates."""
    birdseye_polygon = create_lane_polygon(
        fit,
        transform.image_size,
        config,
    )

    return unwarp_color_image(
        birdseye_polygon,
        transform,
    )


def blend_lane_overlay(
    image: ColorImage,
    projected_polygon: ColorImage,
    config: RenderConfig,
) -> ColorImage:
    """Blend the lane color only where the projected polygon exists."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise RenderingError("Expected a three-channel BGR image")

    if image.dtype != np.uint8:
        raise RenderingError("Expected a uint8 camera image")

    if projected_polygon.shape != image.shape:
        raise RenderingError("Projected polygon dimensions must match the image")

    if projected_polygon.dtype != np.uint8:
        raise RenderingError("Expected a uint8 projected polygon")

    blended = cv2.addWeighted(
        image,
        1.0 - config.overlay_alpha,
        projected_polygon,
        config.overlay_alpha,
        0.0,
    )

    result = image.copy()
    lane_mask = np.any(projected_polygon != 0, axis=2)
    result[lane_mask] = blended[lane_mask]

    return result


def format_measurement_lines(
    measurements: LaneMeasurements,
) -> tuple[str, str, str, str]:
    """Format lane measurements for display on an image."""

    def curvature_text(value: float) -> str:
        return "straight" if isinf(value) else f"{value:.0f} m"

    offset = measurements.vehicle_offset_m

    if abs(offset) < 0.005:
        position = "centered"
    elif offset > 0:
        position = f"{offset:.2f} m right of center"
    else:
        position = f"{abs(offset):.2f} m left of center"

    return (
        f"Left curvature: {curvature_text(measurements.left_curvature_m)}",
        f"Right curvature: {curvature_text(measurements.right_curvature_m)}",
        f"Lane width: {measurements.lane_width_m:.2f} m",
        f"Vehicle: {position}",
    )


def draw_measurement_text(
    image: ColorImage,
    measurements: LaneMeasurements,
    config: RenderConfig,
) -> ColorImage:
    """Draw lane measurements without modifying the input image."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise RenderingError("Expected a three-channel BGR image")

    if image.dtype != np.uint8 or image.size == 0:
        raise RenderingError("Expected a nonempty uint8 image")

    result = image.copy()
    x_position = 20
    first_baseline = 40
    line_spacing = 35

    for index, line in enumerate(format_measurement_lines(measurements)):
        y_position = first_baseline + index * line_spacing

        cv2.putText(
            result,
            line,
            (x_position, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.font_scale,
            config.text_color_bgr,
            config.line_thickness,
            lineType=cv2.LINE_AA,
        )

    return result
