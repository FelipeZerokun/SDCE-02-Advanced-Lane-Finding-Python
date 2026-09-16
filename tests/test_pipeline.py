from pathlib import Path

import cv2
import numpy as np
import pytest

from advanced_lane_finding.calibration import (
    CalibrationError,
    CameraCalibration,
    calibrate_camera,
)
from advanced_lane_finding.config import load_config
from advanced_lane_finding.detector import DetectionError
from advanced_lane_finding.pipeline import process_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _identity_calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.asarray(
            [
                [1000.0, 0.0, 640.0],
                [0.0, 1000.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        distortion_coefficients=np.zeros(5, dtype=np.float64),
        image_size=(1280, 720),
        rms_error=0.0,
        attempted_images=1,
        successful_images=1,
    )


def test_processes_real_road_image() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.toml")

    calibration_paths = sorted(
        (PROJECT_ROOT / "data" / "camera_calibration").glob("calibration*.jpg")
    )
    calibration = calibrate_camera(
        calibration_paths,
        config.calibration,
    )

    loaded = cv2.imread(
        str(PROJECT_ROOT / "data" / "test_images" / "test1.jpg"),
        cv2.IMREAD_COLOR,
    )
    assert loaded is not None

    image = np.asarray(loaded, dtype=np.uint8)
    original = image.copy()

    result = process_image(image, calibration, config)

    assert result.rendered_image.shape == image.shape
    assert result.rendered_image.dtype == np.uint8
    assert result.binary_mask.shape == image.shape[:2]
    assert result.warped_binary.shape == image.shape[:2]
    assert result.lane_pixels.left_count >= config.validation.minimum_lane_pixels
    assert result.lane_pixels.right_count >= config.validation.minimum_lane_pixels
    assert (
        config.validation.minimum_lane_width_m
        <= result.measurements.lane_width_m
        <= config.validation.maximum_lane_width_m
    )
    assert np.any(result.rendered_image != result.undistorted_image)
    np.testing.assert_array_equal(image, original)


def test_rejects_image_with_wrong_calibration_size() -> None:
    config = load_config()
    image = np.zeros((720, 1279, 3), dtype=np.uint8)

    with pytest.raises(
        CalibrationError,
        match="dimensions do not match",
    ):
        process_image(image, _identity_calibration(), config)


def test_reports_when_no_lane_can_be_detected() -> None:
    config = load_config()
    image = np.zeros((720, 1280, 3), dtype=np.uint8)

    with pytest.raises(
        DetectionError,
        match="left lane base",
    ):
        process_image(image, _identity_calibration(), config)
