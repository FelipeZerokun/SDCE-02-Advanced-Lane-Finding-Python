from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from advanced_lane_finding.calibration import (
    calibrate_camera,
    undistort_image,
)
from advanced_lane_finding.config import load_config
from advanced_lane_finding.thresholding import (
    absolute_sobel_threshold,
    combined_threshold,
    gradient_direction_threshold,
    gradient_magnitude_threshold,
    hls_color_threshold,
)

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

    threshold_config = config.threshold

    sobel_x = absolute_sobel_threshold(
        undistorted,
        threshold=threshold_config.gradient_x,
        kernel_size=threshold_config.sobel_kernel,
        orientation="x",
    )

    magnitude = gradient_magnitude_threshold(
        undistorted,
        threshold=threshold_config.gradient_magnitude,
        kernel_size=threshold_config.sobel_kernel,
    )

    direction = gradient_direction_threshold(
        undistorted,
        threshold=threshold_config.gradient_direction,
        kernel_size=threshold_config.sobel_kernel,
    )

    color = hls_color_threshold(
        undistorted,
        lightness_threshold=threshold_config.hls_lightness,
        saturation_threshold=threshold_config.hls_saturation,
    )

    combined = combined_threshold(
        undistorted,
        threshold_config,
    )

    _write_image(
        OUTPUT_DIRECTORY / "threshold_undistorted.jpg",
        undistorted,
    )

    _write_image(
        OUTPUT_DIRECTORY / "threshold_sobel_x.png",
        sobel_x,
    )
    _write_image(
        OUTPUT_DIRECTORY / "threshold_magnitude.png",
        magnitude,
    )
    _write_image(
        OUTPUT_DIRECTORY / "threshold_direction.png",
        direction,
    )
    _write_image(
        OUTPUT_DIRECTORY / "threshold_color.png",
        color,
    )
    _write_image(
        OUTPUT_DIRECTORY / "threshold_combined.png",
        combined,
    )

    print(f"Calibration RMS error: {calibration.rms_error:.4f}")
    print(f"Saved undistorted images to: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
