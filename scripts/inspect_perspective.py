from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from advanced_lane_finding.calibration import (
    calibrate_camera,
    undistort_image,
)
from advanced_lane_finding.config import load_config
from advanced_lane_finding.perspective import (
    create_perspective_transform,
    warp_binary_image,
)
from advanced_lane_finding.thresholding import combined_threshold

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

    source_points = np.asarray(
        config.perspective.source,
        dtype=np.float32,
    )
    source_scale = np.asarray(
        [width - 1, height - 1],
        dtype=np.float32,
    )
    source_polygon = np.asarray(
        np.rint(source_points * source_scale),
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    source_visualization = undistorted.copy()

    cv2.polylines(
        source_visualization,
        [source_polygon],
        isClosed=True,
        color=(0, 0, 255),
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    _write_image(
        OUTPUT_DIRECTORY / "perspective_source.jpg",
        source_visualization,
    )
    _write_image(
        OUTPUT_DIRECTORY / "perspective_binary.png",
        warped_binary,
    )

    print(f"Calibration RMS error: {calibration.rms_error:.4f}")
    print(f"Saved perspective diagnostics to: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
