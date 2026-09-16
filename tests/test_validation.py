import numpy as np
import pytest

from advanced_lane_finding.config import (
    MeasurementConfig,
    PerspectiveConfig,
    ValidationConfig,
)
from advanced_lane_finding.detector import LaneFit
from advanced_lane_finding.validation import (
    LaneValidationError,
    validate_lane_fit,
)


def _perspective_config() -> PerspectiveConfig:
    destination = (
        (0.2, 1.0),
        (0.2, 0.0),
        (0.8, 0.0),
        (0.8, 1.0),
    )

    return PerspectiveConfig(
        source=destination,
        destination=destination,
    )


def _measurement_config() -> MeasurementConfig:
    return MeasurementConfig(
        lane_width_m=3.6,
        visible_road_length_m=25.0,
    )


def _validation_config() -> ValidationConfig:
    return ValidationConfig(
        minimum_lane_pixels=300,
        minimum_lane_width_m=2.8,
        maximum_lane_width_m=4.5,
        maximum_width_variation_m=1.0,
    )


def _straight_fit(
    left_x: float,
    right_x: float,
) -> LaneFit:
    return LaneFit(
        left_coefficients=np.asarray(
            [0.0, 0.0, left_x],
            dtype=np.float64,
        ),
        right_coefficients=np.asarray(
            [0.0, 0.0, right_x],
            dtype=np.float64,
        ),
    )


def test_accepts_valid_lane_geometry() -> None:
    result = validate_lane_fit(
        fit=_straight_fit(20.0, 80.0),
        image_size=(101, 51),
        perspective=_perspective_config(),
        measurement=_measurement_config(),
        validation=_validation_config(),
    )

    assert result.top_width_m == pytest.approx(3.6)
    assert result.middle_width_m == pytest.approx(3.6)
    assert result.bottom_width_m == pytest.approx(3.6)
    assert result.width_variation_m == pytest.approx(0.0)


def test_rejects_lane_that_is_too_narrow() -> None:
    with pytest.raises(
        LaneValidationError,
        match="below the configured minimum",
    ):
        validate_lane_fit(
            fit=_straight_fit(30.0, 70.0),
            image_size=(101, 51),
            perspective=_perspective_config(),
            measurement=_measurement_config(),
            validation=_validation_config(),
        )


def test_rejects_lane_that_is_too_wide() -> None:
    with pytest.raises(
        LaneValidationError,
        match="above the configured maximum",
    ):
        validate_lane_fit(
            fit=_straight_fit(10.0, 90.0),
            image_size=(101, 51),
            perspective=_perspective_config(),
            measurement=_measurement_config(),
            validation=_validation_config(),
        )


def test_rejects_crossed_lane_boundaries() -> None:
    with pytest.raises(
        LaneValidationError,
        match="must remain to the right",
    ):
        validate_lane_fit(
            fit=_straight_fit(60.0, 40.0),
            image_size=(101, 51),
            perspective=_perspective_config(),
            measurement=_measurement_config(),
            validation=_validation_config(),
        )


def test_rejects_excessive_lane_width_variation() -> None:
    fit = LaneFit(
        left_coefficients=np.asarray(
            [0.0, 0.0, 20.0],
            dtype=np.float64,
        ),
        right_coefficients=np.asarray(
            [0.008, 0.0, 70.0],
            dtype=np.float64,
        ),
    )

    with pytest.raises(
        LaneValidationError,
        match="variation exceeds",
    ):
        validate_lane_fit(
            fit=fit,
            image_size=(101, 51),
            perspective=_perspective_config(),
            measurement=_measurement_config(),
            validation=_validation_config(),
        )


def test_accepts_narrow_far_field_when_near_widths_are_valid() -> None:
    fit = LaneFit(
        left_coefficients=np.asarray(
            [0.0, 0.0, 20.0],
            dtype=np.float64,
        ),
        right_coefficients=np.asarray(
            [0.0, 0.3, 62.5],
            dtype=np.float64,
        ),
    )

    result = validate_lane_fit(
        fit=fit,
        image_size=(101, 51),
        perspective=_perspective_config(),
        measurement=_measurement_config(),
        validation=_validation_config(),
    )

    assert result.top_width_m == pytest.approx(2.55)
    assert result.middle_width_m == pytest.approx(3.0)
    assert result.bottom_width_m == pytest.approx(3.45)
    assert result.width_variation_m == pytest.approx(0.9)
