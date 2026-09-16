import numpy as np
import pytest

from advanced_lane_finding.config import PerspectiveConfig, RenderConfig
from advanced_lane_finding.detector import LaneFit
from advanced_lane_finding.perspective import (
    create_perspective_transform,
)
from advanced_lane_finding.rendering import (
    LaneMeasurements,
    RenderingError,
    blend_lane_overlay,
    create_lane_polygon,
    draw_measurement_text,
    format_measurement_lines,
    project_lane_polygon,
)


def _render_config() -> RenderConfig:
    return RenderConfig(
        lane_color_bgr=(0, 255, 0),
        text_color_bgr=(0, 255, 255),
        overlay_alpha=0.3,
        font_scale=1.0,
        line_thickness=2,
    )


def _straight_fit(left_x: float, right_x: float) -> LaneFit:
    return LaneFit(
        left_coefficients=np.asarray([0.0, 0.0, left_x]),
        right_coefficients=np.asarray([0.0, 0.0, right_x]),
    )


def test_creates_filled_lane_polygon() -> None:
    result = create_lane_polygon(
        fit=_straight_fit(2.0, 7.0),
        image_size=(10, 8),
        config=_render_config(),
    )

    assert result.shape == (8, 10, 3)
    assert result.dtype == np.uint8

    np.testing.assert_array_equal(result[4, 4], [0, 255, 0])
    np.testing.assert_array_equal(result[4, 0], [0, 0, 0])
    np.testing.assert_array_equal(result[4, 9], [0, 0, 0])


def test_rejects_crossed_lane_boundaries() -> None:
    with pytest.raises(
        RenderingError,
        match="Right lane boundary must remain right",
    ):
        create_lane_polygon(
            fit=_straight_fit(7.0, 2.0),
            image_size=(10, 8),
            config=_render_config(),
        )


def test_rejects_image_that_is_too_small() -> None:
    with pytest.raises(
        RenderingError,
        match="at least two pixels",
    ):
        create_lane_polygon(
            fit=_straight_fit(0.0, 1.0),
            image_size=(1, 8),
            config=_render_config(),
        )


def test_identity_projection_preserves_lane_polygon() -> None:
    corners = (
        (0.0, 1.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
    )
    transform = create_perspective_transform(
        (10, 8),
        PerspectiveConfig(
            source=corners,
            destination=corners,
        ),
    )
    fit = _straight_fit(2.0, 7.0)
    config = _render_config()

    expected = create_lane_polygon(fit, (10, 8), config)
    result = project_lane_polygon(fit, transform, config)

    np.testing.assert_array_equal(result, expected)


def test_blends_only_inside_lane_polygon() -> None:
    image = np.full((8, 10, 3), 100, dtype=np.uint8)
    polygon = create_lane_polygon(
        _straight_fit(2.0, 7.0),
        (10, 8),
        _render_config(),
    )

    result = blend_lane_overlay(
        image,
        polygon,
        _render_config(),
    )

    np.testing.assert_array_equal(result[4, 0], [100, 100, 100])
    np.testing.assert_array_equal(result[4, 4], [70, 146, 70])


def test_formats_measurements_and_offset_direction() -> None:
    measurements = LaneMeasurements(
        left_curvature_m=879.1,
        right_curvature_m=float("inf"),
        lane_width_m=3.57,
        vehicle_offset_m=-0.199,
    )

    assert format_measurement_lines(measurements) == (
        "Left curvature: 879 m",
        "Right curvature: straight",
        "Lane width: 3.57 m",
        "Vehicle: 0.20 m left of center",
    )


def test_draws_measurements_without_modifying_input() -> None:
    image = np.zeros((200, 640, 3), dtype=np.uint8)
    original = image.copy()

    measurements = LaneMeasurements(
        left_curvature_m=879.1,
        right_curvature_m=361.75,
        lane_width_m=3.57,
        vehicle_offset_m=-0.199,
    )

    result = draw_measurement_text(
        image,
        measurements,
        _render_config(),
    )

    np.testing.assert_array_equal(image, original)
    assert result.shape == image.shape
    assert np.any(result != 0)
