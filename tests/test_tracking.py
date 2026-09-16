from typing import cast
from unittest.mock import Mock, patch

import numpy as np
import pytest

from advanced_lane_finding.calibration import CalibrationError, CameraCalibration
from advanced_lane_finding.config import TrackingConfig, load_config
from advanced_lane_finding.detector import DetectionError, LaneFit, LanePixels
from advanced_lane_finding.measurements import LaneMeasurements
from advanced_lane_finding.pipeline import ImagePipelineResult
from advanced_lane_finding.tracking import LaneTracker
from advanced_lane_finding.validation import LaneValidationResult


def _fit(shift: float) -> LaneFit:
    return LaneFit(
        left_coefficients=np.asarray(
            [0.0, 0.0, 20.0 + shift],
            dtype=np.float64,
        ),
        right_coefficients=np.asarray(
            [0.0, 0.0, 70.0 + shift],
            dtype=np.float64,
        ),
    )


def _tracker() -> LaneTracker:
    return LaneTracker(
        TrackingConfig(
            history_size=3,
            maximum_failures=2,
        )
    )


def _pipeline_result(fit: LaneFit) -> ImagePipelineResult:
    image = np.zeros((51, 101, 3), dtype=np.uint8)
    binary = np.zeros((51, 101), dtype=np.uint8)
    empty = np.asarray([], dtype=np.int64)

    return ImagePipelineResult(
        rendered_image=image.copy(),
        undistorted_image=image,
        binary_mask=binary,
        warped_binary=binary.copy(),
        lane_pixels=LanePixels(
            left_x=empty,
            left_y=empty,
            right_x=empty,
            right_y=empty,
        ),
        lane_fit=fit,
        validation=LaneValidationResult(
            top_width_m=3.0,
            middle_width_m=3.0,
            bottom_width_m=3.0,
        ),
        measurements=LaneMeasurements(
            left_curvature_m=float("inf"),
            right_curvature_m=float("inf"),
            lane_width_m=3.0,
            vehicle_offset_m=0.0,
        ),
    )


def test_tracker_starts_without_a_fit() -> None:
    tracker = _tracker()

    assert not tracker.has_fit
    assert tracker.current_fit is None
    assert tracker.consecutive_failures == 0


def test_tracker_averages_recent_successful_fits() -> None:
    tracker = _tracker()

    for shift in (0.0, 3.0, 6.0, 9.0):
        tracker.record_success(_fit(shift))

    current = tracker.current_fit
    assert current is not None

    # The history holds 3, 6, and 9; their average shift is 6.
    np.testing.assert_allclose(
        current.left_coefficients,
        [0.0, 0.0, 26.0],
    )
    np.testing.assert_allclose(
        current.right_coefficients,
        [0.0, 0.0, 76.0],
    )


def test_tracker_discards_fit_after_maximum_failures() -> None:
    tracker = _tracker()
    tracker.record_success(_fit(0.0))

    tracker.record_failure()
    assert tracker.has_fit
    assert tracker.consecutive_failures == 1

    tracker.record_failure()
    assert not tracker.has_fit
    assert tracker.current_fit is None
    assert tracker.consecutive_failures == 2


def test_success_and_reset_clear_failure_state() -> None:
    tracker = _tracker()
    tracker.record_success(_fit(0.0))
    tracker.record_failure()

    tracker.record_success(_fit(3.0))
    assert tracker.has_fit
    assert tracker.consecutive_failures == 0

    tracker.reset()
    assert not tracker.has_fit
    assert tracker.current_fit is None
    assert tracker.consecutive_failures == 0


def test_process_frame_uses_previous_fit_after_success() -> None:
    tracker = _tracker()
    config = load_config()
    image = np.zeros((51, 101, 3), dtype=np.uint8)
    calibration = cast(CameraCalibration, Mock(spec=CameraCalibration))
    result = _pipeline_result(_fit(0.0))

    with patch(
        "advanced_lane_finding.tracking.process_image",
        return_value=result,
    ) as process:
        first = tracker.process_frame(image, calibration, config)
        second = tracker.process_frame(image, calibration, config)

    assert first is result
    assert second is not None
    assert second is not result

    assert process.call_args_list[0].kwargs["previous_fit"] is None

    previous_fit = process.call_args_list[1].kwargs["previous_fit"]
    assert isinstance(previous_fit, LaneFit)
    np.testing.assert_allclose(
        previous_fit.left_coefficients,
        result.lane_fit.left_coefficients,
    )
    assert tracker.has_fit
    assert tracker.consecutive_failures == 0


def test_process_frame_records_detection_failures() -> None:
    tracker = _tracker()
    tracker.record_success(_fit(0.0))

    config = load_config()
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    calibration = cast(CameraCalibration, Mock(spec=CameraCalibration))

    with patch(
        "advanced_lane_finding.tracking.process_image",
        side_effect=DetectionError("No lane found"),
    ):
        assert tracker.process_frame(image, calibration, config) is None
        assert tracker.has_fit
        assert tracker.consecutive_failures == 1
        assert tracker.last_failure_reason == "DetectionError: No lane found"

        assert tracker.process_frame(image, calibration, config) is None

    assert not tracker.has_fit
    assert tracker.consecutive_failures == 2


def test_process_frame_does_not_hide_calibration_error() -> None:
    tracker = _tracker()
    config = load_config()
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    calibration = cast(CameraCalibration, Mock(spec=CameraCalibration))

    with (
        patch(
            "advanced_lane_finding.tracking.process_image",
            side_effect=CalibrationError("Wrong camera calibration"),
        ),
        pytest.raises(
            CalibrationError,
            match="Wrong camera calibration",
        ),
    ):
        tracker.process_frame(image, calibration, config)

    assert tracker.consecutive_failures == 0
    assert not tracker.has_fit


def test_process_frame_renders_average_of_two_fits() -> None:
    tracker = _tracker()
    config = load_config()
    image = np.zeros((51, 101, 3), dtype=np.uint8)
    calibration = cast(CameraCalibration, Mock(spec=CameraCalibration))

    first_result = _pipeline_result(_fit(0.0))
    second_result = _pipeline_result(_fit(6.0))

    with patch(
        "advanced_lane_finding.tracking.process_image",
        side_effect=[first_result, second_result],
    ):
        first = tracker.process_frame(image, calibration, config)
        second = tracker.process_frame(image, calibration, config)

    assert first is first_result
    assert second is not None
    np.testing.assert_allclose(
        second.lane_fit.left_coefficients,
        [0.0, 0.0, 23.0],
    )
    np.testing.assert_allclose(
        second.lane_fit.right_coefficients,
        [0.0, 0.0, 73.0],
    )
    assert np.any(second.rendered_image != 0)


def test_tracker_recovers_after_one_failed_frame() -> None:
    tracker = _tracker()
    config = load_config()
    image = np.zeros((51, 101, 3), dtype=np.uint8)
    calibration = cast(CameraCalibration, Mock(spec=CameraCalibration))

    first_result = _pipeline_result(_fit(0.0))
    recovered_result = _pipeline_result(_fit(3.0))

    with patch(
        "advanced_lane_finding.tracking.process_image",
        side_effect=[
            first_result,
            DetectionError("Lane temporarily missing"),
            recovered_result,
        ],
    ) as process:
        first = tracker.process_frame(image, calibration, config)
        missed = tracker.process_frame(image, calibration, config)

        assert missed is None
        assert tracker.has_fit
        assert tracker.consecutive_failures == 1

        recovered = tracker.process_frame(image, calibration, config)

    assert first is first_result
    assert recovered is not None
    assert tracker.consecutive_failures == 0

    previous_fit = process.call_args_list[2].kwargs["previous_fit"]
    assert isinstance(previous_fit, LaneFit)
    np.testing.assert_allclose(
        previous_fit.left_coefficients,
        _fit(0.0).left_coefficients,
    )

    # The recovered display averages shifts 0 and 3.
    np.testing.assert_allclose(
        recovered.lane_fit.left_coefficients,
        [0.0, 0.0, 21.5],
    )
