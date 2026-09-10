from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest

from advanced_lane_finding.calibration import (
    CalibrationError,
    CameraCalibration,
    ColorImage,
    calibrate_camera,
    load_calibration,
    save_calibration,
    undistort_image,
)
from advanced_lane_finding.config import load_config

PROJECT_ROOT = Path(__file__).parents[1]
CALIBRATION_DIRECTORY = PROJECT_ROOT / "data" / "camera_calibration"
TEST_IMAGE = PROJECT_ROOT / "data" / "test_images" / "test1.jpg"


@pytest.fixture(scope="module")
def camera_calibration() -> CameraCalibration:
    config = load_config().calibration
    paths = CALIBRATION_DIRECTORY.glob("calibration*.jpg")

    return calibrate_camera(paths, config)


def test_calibrates_supplied_dataset(
    camera_calibration: CameraCalibration,
) -> None:
    assert camera_calibration.attempted_images == 20
    assert camera_calibration.successful_images >= 10
    assert camera_calibration.image_size == (1280, 720)

    assert camera_calibration.camera_matrix.shape == (3, 3)
    assert camera_calibration.distortion_coefficients.size >= 4

    assert np.all(np.isfinite(camera_calibration.camera_matrix))
    assert np.all(np.isfinite(camera_calibration.distortion_coefficients))
    assert np.isfinite(camera_calibration.rms_error)

    assert camera_calibration.rms_error >= 0
    assert camera_calibration.camera_matrix[0, 0] > 0
    assert camera_calibration.camera_matrix[1, 1] > 0
    assert np.isclose(camera_calibration.camera_matrix[2, 2], 1.0)


def test_undistorts_supplied_image(
    camera_calibration: CameraCalibration,
) -> None:
    image = cv2.imread(str(TEST_IMAGE), cv2.IMREAD_COLOR)

    assert image is not None
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.uint8

    color_image = cast(ColorImage, image)
    corrected = undistort_image(color_image, camera_calibration)

    assert corrected.shape == image.shape
    assert corrected.dtype == np.uint8
    assert not np.array_equal(corrected, image)


def test_rejects_empty_image_collection() -> None:
    config = load_config().calibration

    with pytest.raises(
        CalibrationError,
        match="No calibration images were provided",
    ):
        calibrate_camera([], config)


def test_rejects_unreadable_image(tmp_path: Path) -> None:
    config = load_config().calibration
    invalid = tmp_path / "invalid.jpg"
    invalid.write_text("not an image", encoding="utf-8")

    with pytest.raises(
        CalibrationError,
        match="Could not read calibration image",
    ):
        calibrate_camera([invalid], config)


def test_rejects_images_with_different_dimensions(
    tmp_path: Path,
) -> None:
    config = load_config().calibration
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"

    assert cv2.imwrite(
        str(first),
        np.zeros((100, 100, 3), dtype=np.uint8),
    )
    assert cv2.imwrite(
        str(second),
        np.zeros((120, 100, 3), dtype=np.uint8),
    )

    with pytest.raises(
        CalibrationError,
        match="same dimensions",
    ):
        calibrate_camera([first, second], config)


def test_rejects_image_without_chessboard(tmp_path: Path) -> None:
    config = load_config().calibration
    image_path = tmp_path / "blank.jpg"

    assert cv2.imwrite(
        str(image_path),
        np.zeros((100, 100, 3), dtype=np.uint8),
    )

    with pytest.raises(
        CalibrationError,
        match="Chessboard corners were not detected",
    ):
        calibrate_camera([image_path], config)


def test_rejects_incorrect_camera_matrix_shape() -> None:
    with pytest.raises(
        CalibrationError,
        match="Camera matrix must have shape",
    ):
        CameraCalibration(
            camera_matrix=np.eye(2, dtype=np.float64),
            distortion_coefficients=np.zeros(5, dtype=np.float64),
            image_size=(1280, 720),
            rms_error=0.1,
            attempted_images=20,
            successful_images=17,
        )


def test_rejects_undistortion_at_different_resolution(
    camera_calibration: CameraCalibration,
) -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(
        CalibrationError,
        match="do not match the calibration dimensions",
    ):
        undistort_image(image, camera_calibration)


def test_saves_and_loads_calibration(
    tmp_path: Path,
    camera_calibration: CameraCalibration,
) -> None:
    path = tmp_path / "camera-calibration.yml"

    result = save_calibration(camera_calibration, path)
    loaded = load_calibration(path)

    assert result == path
    assert path.is_file()

    np.testing.assert_allclose(
        loaded.camera_matrix,
        camera_calibration.camera_matrix,
    )
    np.testing.assert_allclose(
        loaded.distortion_coefficients,
        camera_calibration.distortion_coefficients,
    )

    assert loaded.image_size == camera_calibration.image_size
    assert loaded.rms_error == pytest.approx(camera_calibration.rms_error)
    assert loaded.attempted_images == camera_calibration.attempted_images
    assert loaded.successful_images == camera_calibration.successful_images


def test_protects_existing_calibration_file(
    tmp_path: Path,
    camera_calibration: CameraCalibration,
) -> None:
    path = tmp_path / "camera-calibration.yml"
    save_calibration(camera_calibration, path)

    with pytest.raises(
        CalibrationError,
        match="already exists",
    ):
        save_calibration(camera_calibration, path)

    result = save_calibration(
        camera_calibration,
        path,
        overwrite=True,
    )

    assert result == path


def test_reports_missing_calibration_file(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.yml"

    with pytest.raises(
        CalibrationError,
        match="Calibration file not found",
    ):
        load_calibration(missing)


def test_rejects_unsupported_calibration_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unsupported.yml"

    storage = cv2.FileStorage(
        str(path),
        cv2.FILE_STORAGE_WRITE,
    )

    assert storage.isOpened()

    try:
        storage.write("format_version", 999)
    finally:
        storage.release()

    with pytest.raises(
        CalibrationError,
        match="Unsupported calibration format version",
    ):
        load_calibration(path)


def test_rejects_corrupt_calibration_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corrupt.yml"
    path.write_text(
        "This is not a valid OpenCV calibration file",
        encoding="utf-8",
    )

    with pytest.raises(
        CalibrationError,
        match="Could not open calibration file",
    ):
        load_calibration(path)
