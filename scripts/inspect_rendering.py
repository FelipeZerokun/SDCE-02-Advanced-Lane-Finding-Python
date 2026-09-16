from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from advanced_lane_finding.calibration import (
    calibrate_camera,
    undistort_image,
)
from advanced_lane_finding.config import load_config
from advanced_lane_finding.detector import (
    find_lane_pixels,
    fit_lane_polynomials,
)
from advanced_lane_finding.measurements import (
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
from advanced_lane_finding.validation import validate_lane_fit

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "default.toml"
CALIBRATION_DIRECTORY = PROJECT_ROOT / "data" / "camera_calibration"
INPUT_IMAGE_PATH = PROJECT_ROOT / "data" / "test_images" / "test1.jpg"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "images"

Uint8Image = NDArray[np.uint8]


def _write_image(path: Path, image: Uint8Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write diagnostic image: {path}")


def main() -> None:
    config = load_config(CONFIG_PATH)

    calibration_paths = sorted(
        CALIBRATION_DIRECTORY.glob("calibration*.jpg"),
    )

    if not calibration_paths:
        raise RuntimeError(f"No calibration images found in: {CALIBRATION_DIRECTORY}")

    calibration = calibrate_camera(
        calibration_paths,
        config.calibration,
    )

    loaded_image = cv2.imread(
        str(INPUT_IMAGE_PATH),
        cv2.IMREAD_COLOR,
    )

    if loaded_image is None:
        raise RuntimeError(f"Could not read input image: {INPUT_IMAGE_PATH}")

    image = np.asarray(
        loaded_image,
        dtype=np.uint8,
    )

    undistorted = undistort_image(
        image,
        calibration,
    )

    combined = combined_threshold(
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
        combined,
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

    lane_measurements = calculate_lane_measurements(
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
        lane_measurements,
        config.render,
    )

    _write_image(
        OUTPUT_DIRECTORY / "rendering_result.jpg",
        rendered,
    )
    print(f"Calibration RMS error: {calibration.rms_error:.4f}")
    print(f"Left lane pixels: {lane_pixels.left_count}")
    print(f"Right lane pixels: {lane_pixels.right_count}")

    print(
        f"Lane widths: "
        f"top={validation_result.top_width_m:.3f} m, "
        f"middle={validation_result.middle_width_m:.3f} m, "
        f"bottom={validation_result.bottom_width_m:.3f} m"
    )
    print(f"Maximum width variation: {validation_result.width_variation_m:.3f} m")
    print(f"Left curvature: {lane_measurements.left_curvature_m:.2f} m")
    print(f"Right curvature: {lane_measurements.right_curvature_m:.2f} m")
    print(
        f"Vehicle offset: "
        f"{lane_measurements.vehicle_offset_m:+.3f} m "
        f"(positive means right of center)"
    )
    print(f"Saved rendered image to: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
