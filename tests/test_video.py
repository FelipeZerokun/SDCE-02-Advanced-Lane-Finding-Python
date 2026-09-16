from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from advanced_lane_finding import video
from advanced_lane_finding.calibration import CameraCalibration
from advanced_lane_finding.config import load_config
from advanced_lane_finding.video import (
    VideoProcessingError,
    _open_video,
    _open_writer,
)


def test_rejects_missing_input_video(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"

    with pytest.raises(VideoProcessingError, match="Input video does not exist"):
        _open_video(missing)


def test_does_not_overwrite_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"keep this file")

    with pytest.raises(VideoProcessingError, match="Output path already exists"):
        _open_writer(output, "mp4v", 25.0, (1280, 720))

    assert output.read_bytes() == b"keep this file"


def test_writes_detected_and_missed_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    overlay = np.full_like(frame, 255)

    calibration = CameraCalibration(
        camera_matrix=np.eye(3, dtype=np.float64),
        distortion_coefficients=np.zeros(5, dtype=np.float64),
        image_size=(4, 3),
        rms_error=0.0,
        attempted_images=1,
        successful_images=1,
    )
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "default.toml"
    )

    capture = Mock()
    capture.read.side_effect = [
        (True, frame),
        (True, frame),
        (False, None),
    ]
    writer = Mock()
    tracker = Mock()
    tracker.process_frame.side_effect = [
        Mock(rendered_image=overlay),
        None,
    ]

    monkeypatch.setattr(video, "_open_video", lambda _: (capture, 25.0, (4, 3)))
    monkeypatch.setattr(video, "_open_writer", lambda *args: writer)
    monkeypatch.setattr(video, "LaneTracker", lambda _: tracker)
    monkeypatch.setattr(video, "undistort_image", lambda image, _: image)

    result = video.process_video(
        tmp_path / "input.mp4",
        tmp_path / "output.mp4",
        calibration,
        config,
    )

    assert result.frame_count == 2
    assert result.detected_frames == 1
    assert result.missed_frames == 1
    assert result.fps == 25.0
    assert result.frame_size == (4, 3)
    assert writer.write.call_count == 2
    np.testing.assert_array_equal(writer.write.call_args_list[0].args[0], overlay)
    np.testing.assert_array_equal(writer.write.call_args_list[1].args[0], frame)
    writer.release.assert_called_once()
    capture.release.assert_called_once()


def test_rejects_video_with_wrong_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = CameraCalibration(
        camera_matrix=np.eye(3, dtype=np.float64),
        distortion_coefficients=np.zeros(5, dtype=np.float64),
        image_size=(4, 3),
        rms_error=0.0,
        attempted_images=1,
        successful_images=1,
    )
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "default.toml"
    )

    capture = Mock()
    open_writer = Mock()

    # The video says 8×6, but the calibration was made for 4×3.
    monkeypatch.setattr(video, "_open_video", lambda _: (capture, 25.0, (8, 6)))
    monkeypatch.setattr(video, "_open_writer", open_writer)

    with pytest.raises(
        VideoProcessingError,
        match="Video dimensions do not match",
    ):
        video.process_video(
            tmp_path / "input.mp4",
            tmp_path / "output.mp4",
            calibration,
            config,
        )

    capture.release.assert_called_once()
    open_writer.assert_not_called()


def test_rejects_nonfinite_video_dimensions(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.mp4"
    input_path.write_bytes(b"fake video")

    with patch("advanced_lane_finding.video.cv2.VideoCapture") as capture_class:
        capture = capture_class.return_value
        capture.isOpened.return_value = True
        capture.get.side_effect = [25.0, float("nan"), 720.0]

        with pytest.raises(
            VideoProcessingError,
            match="invalid FPS or dimensions",
        ):
            _open_video(input_path)

        capture.release.assert_called_once()


def test_processes_real_short_video(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    frame_size = (16, 16)

    source_writer = cv2.VideoWriter(
        str(input_path),
        cv2.VideoWriter.fourcc("m", "p", "4", "v"),
        10.0,
        frame_size,
    )
    if not source_writer.isOpened():
        pytest.skip("mp4v video encoding is unavailable")

    try:
        source_writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
        source_writer.write(np.full((16, 16, 3), 120, dtype=np.uint8))
    finally:
        source_writer.release()

    calibration = CameraCalibration(
        camera_matrix=np.eye(3, dtype=np.float64),
        distortion_coefficients=np.zeros(5, dtype=np.float64),
        image_size=frame_size,
        rms_error=0.0,
        attempted_images=1,
        successful_images=1,
    )
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "default.toml"
    )

    # Keep this test focused on real video I/O; tracking has its own tests.
    with patch.object(video.LaneTracker, "process_frame", return_value=None):
        result = video.process_video(
            input_path,
            output_path,
            calibration,
            config,
        )

    assert result.frame_count == 2
    assert result.missed_frames == 2
    assert output_path.is_file()

    output_capture = cv2.VideoCapture(str(output_path))
    try:
        assert output_capture.isOpened()
        readable_frames = 0
        while True:
            readable, _ = output_capture.read()
            if not readable:
                break
            readable_frames += 1
        assert readable_frames == 2
    finally:
        output_capture.release()


def test_rejects_video_without_readable_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = CameraCalibration(
        camera_matrix=np.eye(3, dtype=np.float64),
        distortion_coefficients=np.zeros(5, dtype=np.float64),
        image_size=(4, 3),
        rms_error=0.0,
        attempted_images=1,
        successful_images=1,
    )
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "default.toml"
    )

    capture = Mock()
    capture.read.return_value = (False, None)
    open_writer = Mock()

    monkeypatch.setattr(video, "_open_video", lambda _: (capture, 25.0, (4, 3)))
    monkeypatch.setattr(video, "_open_writer", open_writer)

    input_path = tmp_path / "empty.mp4"
    output_path = tmp_path / "output.mp4"

    with pytest.raises(
        VideoProcessingError,
        match="no readable frames",
    ):
        video.process_video(input_path, output_path, calibration, config)

    capture.release.assert_called_once()
    open_writer.assert_not_called()
    assert not output_path.exists()
