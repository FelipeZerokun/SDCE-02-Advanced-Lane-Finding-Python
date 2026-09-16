from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from advanced_lane_finding.config import CalibrationConfig

FloatArray = NDArray[np.float64]
ColorImage = NDArray[np.uint8]

CALIBRATION_FORMAT_VERSION: Final = 1


class CalibrationError(RuntimeError):
    """Raised when camera calibration cannot be completed."""


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    camera_matrix: FloatArray
    distortion_coefficients: FloatArray
    image_size: tuple[int, int]
    rms_error: float
    attempted_images: int
    successful_images: int

    def __post_init__(self) -> None:
        if self.camera_matrix.shape != (3, 3):
            raise CalibrationError("Camera matrix must have shape (3, 3)")

        if self.distortion_coefficients.size < 4:
            raise CalibrationError("At least four distortion coefficients are required")

        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise CalibrationError("Calibration image dimensions must be positive")

        if not np.isfinite(self.rms_error) or self.rms_error < 0:
            raise CalibrationError(
                "Calibration RMS error must be finite and nonnegative"
            )

        if self.attempted_images <= 0:
            raise CalibrationError("At least one calibration image must be attempted")

        if not 0 < self.successful_images <= self.attempted_images:
            raise CalibrationError(
                "Successful image count must be between 1 and attempted images"
            )


def _object_points(config: CalibrationConfig) -> NDArray[np.float32]:
    points = np.zeros(
        (
            config.chessboard_columns * config.chessboard_rows,
            3,
        ),
        dtype=np.float32,
    )

    grid = np.mgrid[
        0 : config.chessboard_columns,
        0 : config.chessboard_rows,
    ].T.reshape(-1, 2)

    points[:, :2] = grid.astype(np.float32) * np.float32(config.square_size)
    return points


def calibrate_camera(
    image_paths: Iterable[str | Path],
    config: CalibrationConfig,
) -> CameraCalibration:
    paths = sorted(
        (Path(path) for path in image_paths),
        key=lambda path: str(path).lower(),
    )

    if not paths:
        raise CalibrationError("No calibration images were provided")

    template = _object_points(config)
    object_points: list[NDArray[np.float32]] = []
    image_points: list[NDArray[np.float32]] = []
    image_size: tuple[int, int] | None = None

    chessboard_size = (
        config.chessboard_columns,
        config.chessboard_rows,
    )

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    termination_criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            raise CalibrationError(f"Could not read calibration image: {path}")

        height, width = image.shape[:2]
        current_size = (width, height)

        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise CalibrationError(
                "All calibration images must have the same dimensions"
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            chessboard_size,
            flags=flags,
        )

        if not found or corners is None:
            continue

        refined_corners = np.asarray(
            cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                termination_criteria,
            ),
            dtype=np.float32,
        )

        object_points.append(template.copy())
        image_points.append(refined_corners)

    if image_size is None:
        raise CalibrationError("No readable calibration images were found")

    if not object_points:
        raise CalibrationError(
            "Chessboard corners were not detected in any calibration images"
        )

    rms_error, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    return CameraCalibration(
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        distortion_coefficients=np.asarray(
            distortion_coefficients,
            dtype=np.float64,
        ),
        image_size=image_size,
        rms_error=float(rms_error),
        attempted_images=len(paths),
        successful_images=len(object_points),
    )


def undistort_image(
    image: ColorImage,
    calibration: CameraCalibration,
) -> ColorImage:
    if image.ndim != 3 or image.shape[2] != 3:
        raise CalibrationError("Expected a three-channel BGR image")

    if image.dtype != np.uint8:
        raise CalibrationError("Expected an image with uint8 data")

    height, width = image.shape[:2]

    if (width, height) != calibration.image_size:
        raise CalibrationError(
            "Input image dimensions do not match the calibration dimensions"
        )

    corrected = cv2.undistort(
        image,
        calibration.camera_matrix,
        calibration.distortion_coefficients,
        None,
        calibration.camera_matrix,
    )

    return np.asarray(corrected, dtype=np.uint8)


def _required_node(
    storage: cv2.FileStorage,
    name: str,
) -> cv2.FileNode:
    node = storage.getNode(name)

    if node.empty():
        raise CalibrationError(f"Calibration file is missing required value: {name}")

    return node


def save_calibration(
    calibration: CameraCalibration,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    destination = Path(path)

    if destination.exists() and not overwrite:
        raise CalibrationError(f"Calibration file already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    storage = cv2.FileStorage(
        str(destination),
        cv2.FILE_STORAGE_WRITE,
    )

    if not storage.isOpened():
        raise CalibrationError(
            f"Could not open calibration file for writing: {destination}"
        )

    try:
        width, height = calibration.image_size

        storage.write(
            "format_version",
            CALIBRATION_FORMAT_VERSION,
        )
        storage.write(
            "camera_matrix",
            calibration.camera_matrix,
        )
        storage.write(
            "distortion_coefficients",
            calibration.distortion_coefficients,
        )
        storage.write("image_width", width)
        storage.write("image_height", height)
        storage.write("rms_error", calibration.rms_error)
        storage.write(
            "attempted_images",
            calibration.attempted_images,
        )
        storage.write(
            "successful_images",
            calibration.successful_images,
        )
    finally:
        storage.release()

    return destination


def load_calibration(
    path: str | Path,
) -> CameraCalibration:
    source = Path(path)

    if not source.is_file():
        raise CalibrationError(f"Calibration file not found: {source}")

    try:
        storage = cv2.FileStorage(
            str(source),
            cv2.FILE_STORAGE_READ,
        )
    except (cv2.error, SystemError) as error:
        raise CalibrationError(f"Could not open calibration file: {source}") from error

    if not storage.isOpened():
        raise CalibrationError(f"Could not open calibration file: {source}")

    try:
        format_version = int(_required_node(storage, "format_version").real())

        if format_version != CALIBRATION_FORMAT_VERSION:
            raise CalibrationError(
                f"Unsupported calibration format version: {format_version}"
            )

        camera_matrix = _required_node(
            storage,
            "camera_matrix",
        ).mat()
        distortion_coefficients = _required_node(
            storage,
            "distortion_coefficients",
        ).mat()

        if camera_matrix is None:
            raise CalibrationError("Calibration camera matrix is invalid")

        if distortion_coefficients is None:
            raise CalibrationError("Calibration distortion coefficients are invalid")

        return CameraCalibration(
            camera_matrix=np.asarray(
                camera_matrix,
                dtype=np.float64,
            ),
            distortion_coefficients=np.asarray(
                distortion_coefficients,
                dtype=np.float64,
            ),
            image_size=(
                int(_required_node(storage, "image_width").real()),
                int(_required_node(storage, "image_height").real()),
            ),
            rms_error=float(_required_node(storage, "rms_error").real()),
            attempted_images=int(_required_node(storage, "attempted_images").real()),
            successful_images=int(_required_node(storage, "successful_images").real()),
        )
    finally:
        storage.release()
