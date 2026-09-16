from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from advanced_lane_finding.calibration import CameraCalibration, undistort_image
from advanced_lane_finding.config import AdvancedLaneFindingConfig
from advanced_lane_finding.detector import (
    DetectionError,
    LaneFit,
    LanePixels,
    find_lane_pixels,
    find_lane_pixels_near_fit,
    fit_lane_polynomials,
)
from advanced_lane_finding.measurements import (
    LaneMeasurements,
    calculate_lane_measurements,
)
from advanced_lane_finding.perspective import (
    PerspectiveTransform,
    create_perspective_transform,
    warp_binary_image,
)
from advanced_lane_finding.rendering import (
    blend_lane_overlay,
    draw_measurement_text,
    project_lane_polygon,
)
from advanced_lane_finding.thresholding import combined_threshold
from advanced_lane_finding.validation import (
    LaneValidationError,
    LaneValidationResult,
    validate_lane_fit,
)

ColorImage = NDArray[np.uint8]
BinaryImage = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class ImagePipelineResult:
    rendered_image: ColorImage
    undistorted_image: ColorImage
    binary_mask: BinaryImage
    warped_binary: BinaryImage
    lane_pixels: LanePixels
    lane_fit: LaneFit
    validation: LaneValidationResult
    measurements: LaneMeasurements


def _fit_and_validate(
    pixels: LanePixels,
    image_size: tuple[int, int],
    config: AdvancedLaneFindingConfig,
) -> tuple[LaneFit, LaneValidationResult]:
    fit = fit_lane_polynomials(
        pixels,
        minimum_pixels=config.validation.minimum_lane_pixels,
    )
    validation = validate_lane_fit(
        fit=fit,
        image_size=image_size,
        perspective=config.perspective,
        measurement=config.measurement,
        validation=config.validation,
    )
    return fit, validation


def render_validated_fit(
    undistorted_image: ColorImage,
    fit: LaneFit,
    transform: PerspectiveTransform,
    config: AdvancedLaneFindingConfig,
) -> tuple[ColorImage, LaneMeasurements]:
    """Render a previously validated fit on an undistorted image."""
    measurements = calculate_lane_measurements(
        fit=fit,
        image_size=transform.image_size,
        perspective=config.perspective,
        measurement=config.measurement,
    )

    projected_polygon = project_lane_polygon(
        fit,
        transform,
        config.render,
    )
    blended = blend_lane_overlay(
        undistorted_image,
        projected_polygon,
        config.render,
    )
    rendered = draw_measurement_text(
        blended,
        measurements,
        config.render,
    )

    return rendered, measurements


def process_image(
    image: ColorImage,
    calibration: CameraCalibration,
    config: AdvancedLaneFindingConfig,
    previous_fit: LaneFit | None = None,
) -> ImagePipelineResult:
    """Detect and render the lane in one camera image."""
    undistorted = undistort_image(image, calibration)

    binary_mask = combined_threshold(
        undistorted,
        config.threshold,
    )

    height, width = undistorted.shape[:2]
    image_size = (width, height)

    transform = create_perspective_transform(
        image_size,
        config.perspective,
    )
    warped_binary = warp_binary_image(
        binary_mask,
        transform,
    )

    if previous_fit is not None:
        try:
            lane_pixels = find_lane_pixels_near_fit(
                warped_binary,
                previous_fit,
                config.search,
            )
            lane_fit, validation_result = _fit_and_validate(
                lane_pixels,
                image_size,
                config,
            )
        except (DetectionError, LaneValidationError):
            lane_pixels = find_lane_pixels(
                warped_binary,
                config.search,
            )
            lane_fit, validation_result = _fit_and_validate(
                lane_pixels,
                image_size,
                config,
            )
    else:
        lane_pixels = find_lane_pixels(
            warped_binary,
            config.search,
        )
        lane_fit, validation_result = _fit_and_validate(
            lane_pixels,
            image_size,
            config,
        )

    rendered, measurements = render_validated_fit(
        undistorted,
        lane_fit,
        transform,
        config,
    )

    return ImagePipelineResult(
        rendered_image=rendered,
        undistorted_image=undistorted,
        binary_mask=binary_mask,
        warped_binary=warped_binary,
        lane_pixels=lane_pixels,
        lane_fit=lane_fit,
        validation=validation_result,
        measurements=measurements,
    )
