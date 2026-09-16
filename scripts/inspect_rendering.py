from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from advanced_lane_finding.calibration import (
    calibrate_camera,
)
from advanced_lane_finding.config import load_config
from advanced_lane_finding.pipeline import process_image

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

    result = process_image(
        image,
        calibration,
        config,
    )

    _write_image(
        OUTPUT_DIRECTORY / "rendering_result.jpg",
        result.rendered_image,
    )

    print(f"Calibration RMS error: {calibration.rms_error:.4f}")
    print(f"Left lane pixels: {result.lane_pixels.left_count}")
    print(f"Right lane pixels: {result.lane_pixels.right_count}")
    print(
        f"Lane widths: "
        f"top={result.validation.top_width_m:.3f} m, "
        f"middle={result.validation.middle_width_m:.3f} m, "
        f"bottom={result.validation.bottom_width_m:.3f} m"
    )
    print(f"Maximum width variation: {result.validation.width_variation_m:.3f} m")
    print(f"Left curvature: {result.measurements.left_curvature_m:.2f} m")
    print(f"Right curvature: {result.measurements.right_curvature_m:.2f} m")
    print(
        f"Vehicle offset: "
        f"{result.measurements.vehicle_offset_m:+.3f} m "
        f"(positive means right of center)"
    )
    print(f"Saved rendered image to: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
