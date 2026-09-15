import numpy as np
import pytest

from advanced_lane_finding.config import (
    MeasurementConfig,
    PerspectiveConfig,
    Quadrilateral,
)
from advanced_lane_finding.measurements import (
    LaneFit,
    MeasurementError,
    PixelScale,
    calculate_lane_measurements,
    calculate_pixel_scale,
    curvature_radius_m,
)

DESTINATION: Quadrilateral = (
    (0.2, 1.0),
    (0.2, 0.0),
    (0.8, 0.0),
    (0.8, 1.0),
)


def _perspective_config(
    destination: Quadrilateral = DESTINATION,
) -> PerspectiveConfig:
    return PerspectiveConfig(
        source=DESTINATION,
        destination=destination,
    )


def _measurement_config() -> MeasurementConfig:
    return MeasurementConfig(
        lane_width_m=3.6,
        visible_road_length_m=25.0,
    )


def _straight_lane_fit(
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


def test_calculates_pixel_scale() -> None:
    scale = calculate_pixel_scale(
        image_size=(101, 51),
        perspective=_perspective_config(),
        measurement=_measurement_config(),
    )

    assert scale.x_m_per_pixel == pytest.approx(0.06)
    assert scale.y_m_per_pixel == pytest.approx(0.5)


def test_rejects_measurement_image_that_is_too_small() -> None:
    with pytest.raises(
        MeasurementError,
        match="at least two pixels",
    ):
        calculate_pixel_scale(
            image_size=(1, 51),
            perspective=_perspective_config(),
            measurement=_measurement_config(),
        )


def test_rejects_zero_destination_lane_width() -> None:
    destination: Quadrilateral = (
        (0.5, 1.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 1.0),
    )

    with pytest.raises(
        MeasurementError,
        match="lane width must be positive",
    ):
        calculate_pixel_scale(
            image_size=(101, 51),
            perspective=_perspective_config(destination),
            measurement=_measurement_config(),
        )


def test_rejects_zero_destination_road_length() -> None:
    destination: Quadrilateral = (
        (0.2, 0.5),
        (0.2, 0.5),
        (0.8, 0.5),
        (0.8, 0.5),
    )

    with pytest.raises(
        MeasurementError,
        match="road length must be positive",
    ):
        calculate_pixel_scale(
            image_size=(101, 51),
            perspective=_perspective_config(destination),
            measurement=_measurement_config(),
        )


def test_calculates_known_curvature_radius() -> None:
    coefficients = np.asarray(
        [0.5, 0.0, 0.0],
        dtype=np.float64,
    )
    scale = PixelScale(
        x_m_per_pixel=1.0,
        y_m_per_pixel=1.0,
    )

    result = curvature_radius_m(
        coefficients,
        evaluation_y_px=0.0,
        scale=scale,
    )

    assert result == pytest.approx(1.0)


def test_curvature_changes_with_evaluation_position() -> None:
    coefficients = np.asarray(
        [0.5, 0.0, 0.0],
        dtype=np.float64,
    )
    scale = PixelScale(
        x_m_per_pixel=1.0,
        y_m_per_pixel=1.0,
    )

    result = curvature_radius_m(
        coefficients,
        evaluation_y_px=1.0,
        scale=scale,
    )

    assert result == pytest.approx(2.0**1.5)


def test_straight_lane_has_infinite_curvature_radius() -> None:
    coefficients = np.asarray(
        [0.0, 2.0, 10.0],
        dtype=np.float64,
    )
    scale = PixelScale(
        x_m_per_pixel=1.0,
        y_m_per_pixel=1.0,
    )

    result = curvature_radius_m(
        coefficients,
        evaluation_y_px=50.0,
        scale=scale,
    )

    assert result == float("inf")


def test_rejects_negative_curvature_evaluation_position() -> None:
    coefficients = np.zeros(3, dtype=np.float64)
    scale = PixelScale(
        x_m_per_pixel=1.0,
        y_m_per_pixel=1.0,
    )

    with pytest.raises(
        MeasurementError,
        match="nonnegative and finite",
    ):
        curvature_radius_m(
            coefficients,
            evaluation_y_px=-1.0,
            scale=scale,
        )


def test_calculates_centered_lane_measurements() -> None:
    result = calculate_lane_measurements(
        fit=_straight_lane_fit(20.0, 80.0),
        image_size=(101, 51),
        perspective=_perspective_config(),
        measurement=_measurement_config(),
    )

    assert result.lane_width_m == pytest.approx(3.6)
    assert result.vehicle_offset_m == pytest.approx(0.0)
    assert result.left_curvature_m == float("inf")
    assert result.right_curvature_m == float("inf")


@pytest.mark.parametrize(
    ("left_x", "right_x", "expected_offset_m"),
    [
        (15.0, 75.0, 0.3),
        (25.0, 85.0, -0.3),
    ],
)
def test_calculates_vehicle_offset(
    left_x: float,
    right_x: float,
    expected_offset_m: float,
) -> None:
    result = calculate_lane_measurements(
        fit=_straight_lane_fit(left_x, right_x),
        image_size=(101, 51),
        perspective=_perspective_config(),
        measurement=_measurement_config(),
    )

    assert result.vehicle_offset_m == pytest.approx(expected_offset_m)


def test_rejects_crossed_lane_boundaries() -> None:
    with pytest.raises(
        MeasurementError,
        match="Right lane boundary must be to the right",
    ):
        calculate_lane_measurements(
            fit=_straight_lane_fit(60.0, 40.0),
            image_size=(101, 51),
            perspective=_perspective_config(),
            measurement=_measurement_config(),
        )
