from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from advanced_lane_finding.calibration import CameraCalibration, undistort_image
from advanced_lane_finding.config import AdvancedLaneFindingConfig
from advanced_lane_finding.detector import (
    LaneFit,
    LanePixels,
    find_lane_pixels,
    fit_lane_polynomials,
)
from advanced_lane_finding.measurements import (
    LaneMeasurements,
    calculate_lane_measurements,
)
from advanced_lane_finding.perspective import (
    create_perspective_transform,
    warp_binary_image,
)
from advanced_lane_finding.rendering import (
    blend_lane_overlay,
    draw_measurement_text,
    project_lane_polygon,
)
from advanced_lane_finding.thresholding import combined_threshold
from advanced_lane_finding.validation import LaneValidationResult, validate_lane_fit

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


def process_image(
    image: ColorImage,
    calibration: CameraCalibration,
    config: AdvancedLaneFindingConfig,
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

    lane_pixels = find_lane_pixels(
        warped_binary,
        config.search,
    )
    lane_fit = fit_lane_polynomials(
        lane_pixels,
        minimum_pixels=config.validation.minimum_lane_pixels,
    )

    validation_result = validate_lane_fit(
        fit=lane_fit,
        image_size=image_size,
        perspective=config.perspective,
        measurement=config.measurement,
        validation=config.validation,
    )
    measurements = calculate_lane_measurements(
        fit=lane_fit,
        image_size=image_size,
        perspective=config.perspective,
        measurement=config.measurement,
    )

    projected_polygon = project_lane_polygon(
        lane_fit,
        transform,
        config.render,
    )
    blended = blend_lane_overlay(
        undistorted,
        projected_polygon,
        config.render,
    )
    rendered = draw_measurement_text(
        blended,
        measurements,
        config.render,
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
