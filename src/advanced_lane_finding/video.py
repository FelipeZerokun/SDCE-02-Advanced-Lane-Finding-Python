from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from advanced_lane_finding.calibration import CameraCalibration, undistort_image
from advanced_lane_finding.config import AdvancedLaneFindingConfig
from advanced_lane_finding.tracking import LaneTracker


class VideoProcessingError(RuntimeError):
    """Raised when a video cannot be read, processed, or written."""


@dataclass(frozen=True, slots=True)
class VideoProcessingResult:
    input_path: Path
    output_path: Path
    frame_count: int
    detected_frames: int
    missed_frames: int
    fps: float
    frame_size: tuple[int, int]


def _open_video(
    input_path: Path,
) -> tuple[cv2.VideoCapture, float, tuple[int, int]]:
    """Open a video and return its capture, FPS, and (width, height)."""
    if not input_path.is_file():
        raise VideoProcessingError(f"Input video does not exist: {input_path}")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise VideoProcessingError(f"Could not open input video: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width_value = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_value = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if (
        not math.isfinite(fps)
        or fps <= 0
        or not math.isfinite(width_value)
        or width_value < 1
        or not math.isfinite(height_value)
        or height_value < 1
    ):
        capture.release()
        raise VideoProcessingError(
            f"Input video has invalid FPS or dimensions: {input_path}"
        )

    width = int(width_value)
    height = int(height_value)

    return capture, fps, (width, height)


def _open_writer(
    output_path: Path,
    codec: str,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    """Create a video writer without replacing an existing file."""
    if output_path.exists():
        raise VideoProcessingError(f"Output path already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter.fourcc(codec[0], codec[1], codec[2], codec[3])
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        frame_size,
    )
    if not writer.isOpened():
        writer.release()
        raise VideoProcessingError(f"Could not create output video: {output_path}")

    return writer


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    calibration: CameraCalibration,
    config: AdvancedLaneFindingConfig,
) -> VideoProcessingResult:
    """Track and render lanes throughout an input video."""
    source = Path(input_path)
    destination = Path(output_path)

    capture, fps, frame_size = _open_video(source)
    try:
        if frame_size != calibration.image_size:
            raise VideoProcessingError(
                "Video dimensions do not match the calibration dimensions"
            )

        readable, frame = capture.read()
        if not readable or frame is None:
            raise VideoProcessingError("Input video contains no readable frames")

        writer = _open_writer(destination, config.video.codec, fps, frame_size)
        try:
            tracker = LaneTracker(config.tracking)
            frame_count = 0
            detected_frames = 0
            missed_frames = 0

            while readable and frame is not None:
                image = np.asarray(frame, dtype=np.uint8)
                result = tracker.process_frame(image, calibration, config)

                if result is None:
                    output_image = undistort_image(image, calibration)
                    missed_frames += 1
                else:
                    output_image = result.rendered_image
                    detected_frames += 1

                writer.write(output_image)
                frame_count += 1
                readable, frame = capture.read()

        finally:
            writer.release()
    finally:
        capture.release()

    return VideoProcessingResult(
        input_path=source,
        output_path=destination,
        frame_count=frame_count,
        detected_frames=detected_frames,
        missed_frames=missed_frames,
        fps=fps,
        frame_size=frame_size,
    )
