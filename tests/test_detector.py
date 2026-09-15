from typing import cast

import numpy as np
import pytest

from advanced_lane_finding.config import SearchConfig
from advanced_lane_finding.detector import (
    BinaryImage,
    DetectionError,
    LaneFit,
    LanePixels,
    evaluate_lane_fit,
    find_lane_pixels,
    find_lane_pixels_near_fit,
    fit_lane_polynomials,
    lower_half_histogram,
)


def test_lane_pixels_reports_coordinate_counts() -> None:
    pixels = LanePixels(
        left_x=np.asarray([1, 2, 3], dtype=np.int64),
        left_y=np.asarray([4, 5, 6], dtype=np.int64),
        right_x=np.asarray([7, 8], dtype=np.int64),
        right_y=np.asarray([4, 5], dtype=np.int64),
    )

    assert pixels.left_count == 3
    assert pixels.right_count == 2


def test_lane_pixels_rejects_mismatched_coordinates() -> None:
    with pytest.raises(
        DetectionError,
        match="Left lane x and y coordinates",
    ):
        LanePixels(
            left_x=np.asarray([1, 2], dtype=np.int64),
            left_y=np.asarray([3], dtype=np.int64),
            right_x=np.asarray([], dtype=np.int64),
            right_y=np.asarray([], dtype=np.int64),
        )


def test_lower_half_histogram_counts_selected_pixels() -> None:
    image = np.zeros((8, 10), dtype=np.uint8)

    image[4:8, 2] = 255
    image[6:8, 7] = 255
    image[0:4, 5] = 255

    result = lower_half_histogram(image)

    expected = np.zeros(10, dtype=np.int64)
    expected[2] = 4
    expected[7] = 2

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.int64


def test_histogram_rejects_three_channel_image() -> None:
    image = np.zeros((8, 10, 3), dtype=np.uint8)

    with pytest.raises(
        DetectionError,
        match="single-channel",
    ):
        lower_half_histogram(
            cast(BinaryImage, image),
        )


def test_histogram_rejects_non_uint8_image() -> None:
    image = np.zeros((8, 10), dtype=np.float32)

    with pytest.raises(
        DetectionError,
        match="uint8",
    ):
        lower_half_histogram(
            cast(BinaryImage, image),
        )


def test_histogram_rejects_empty_image() -> None:
    image = np.zeros((0, 0), dtype=np.uint8)

    with pytest.raises(
        DetectionError,
        match="cannot be empty",
    ):
        lower_half_histogram(image)


def _search_config(
    *,
    windows: int = 8,
) -> SearchConfig:
    return SearchConfig(
        windows=windows,
        margin=6,
        minimum_pixels=5,
        previous_fit_margin=5,
    )


def test_finds_synthetic_curved_lane_pixels() -> None:
    height = 80
    width = 100
    image = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        curve_offset = round(0.003 * (y - 40) ** 2)
        left_x = 20 + curve_offset
        right_x = 70 + curve_offset

        image[y, left_x - 1 : left_x + 2] = 255
        image[y, right_x - 1 : right_x + 2] = 255

    result = find_lane_pixels(
        image,
        _search_config(),
    )

    expected_left_x = 20 + np.rint(0.003 * (result.left_y - 40) ** 2).astype(np.int64)
    expected_right_x = 70 + np.rint(0.003 * (result.right_y - 40) ** 2).astype(np.int64)

    assert result.left_count == height * 3
    assert result.right_count == height * 3

    assert np.max(np.abs(result.left_x - expected_left_x)) <= 1
    assert np.max(np.abs(result.right_x - expected_right_x)) <= 1


def test_rejects_image_without_lane_bases() -> None:
    image = np.zeros((80, 100), dtype=np.uint8)

    with pytest.raises(
        DetectionError,
        match="left lane base",
    ):
        find_lane_pixels(
            image,
            _search_config(),
        )


def test_rejects_more_windows_than_image_rows() -> None:
    image = np.zeros((8, 100), dtype=np.uint8)

    with pytest.raises(
        DetectionError,
        match="cannot exceed image height",
    ):
        find_lane_pixels(
            image,
            _search_config(windows=9),
        )


def test_fits_known_lane_polynomials() -> None:
    y_values = np.arange(10, dtype=np.int64)

    left_x = y_values**2 + 2 * y_values + 3
    right_x = y_values**2 + 2 * y_values + 53

    pixels = LanePixels(
        left_x=left_x,
        left_y=y_values,
        right_x=right_x,
        right_y=y_values,
    )

    result = fit_lane_polynomials(
        pixels,
        minimum_pixels=10,
    )

    np.testing.assert_allclose(
        result.left_coefficients,
        np.asarray([1.0, 2.0, 3.0]),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.right_coefficients,
        np.asarray([1.0, 2.0, 53.0]),
        atol=1e-10,
    )


def test_rejects_insufficient_left_lane_pixels() -> None:
    pixels = LanePixels(
        left_x=np.asarray([1, 2], dtype=np.int64),
        left_y=np.asarray([1, 2], dtype=np.int64),
        right_x=np.asarray([7, 8, 9], dtype=np.int64),
        right_y=np.asarray([1, 2, 3], dtype=np.int64),
    )

    with pytest.raises(
        DetectionError,
        match="Not enough left lane pixels",
    ):
        fit_lane_polynomials(
            pixels,
            minimum_pixels=3,
        )


def test_rejects_invalid_minimum_pixel_count() -> None:
    empty = np.asarray([], dtype=np.int64)

    pixels = LanePixels(
        left_x=empty,
        left_y=empty,
        right_x=empty,
        right_y=empty,
    )

    with pytest.raises(
        DetectionError,
        match="At least three pixels",
    ):
        fit_lane_polynomials(
            pixels,
            minimum_pixels=2,
        )


def test_evaluates_lane_fit() -> None:
    fit = LaneFit(
        left_coefficients=np.asarray(
            [1.0, 2.0, 3.0],
            dtype=np.float64,
        ),
        right_coefficients=np.asarray(
            [1.0, 2.0, 53.0],
            dtype=np.float64,
        ),
    )
    y_values = np.asarray(
        [0.0, 1.0, 2.0],
        dtype=np.float64,
    )

    left_x, right_x = evaluate_lane_fit(
        fit,
        y_values,
    )

    np.testing.assert_allclose(
        left_x,
        np.asarray([3.0, 6.0, 11.0]),
    )
    np.testing.assert_allclose(
        right_x,
        np.asarray([53.0, 56.0, 61.0]),
    )


def test_rejects_empty_polynomial_coordinates() -> None:
    fit = LaneFit(
        left_coefficients=np.zeros(3, dtype=np.float64),
        right_coefficients=np.zeros(3, dtype=np.float64),
    )
    empty = np.asarray([], dtype=np.float64)

    with pytest.raises(
        DetectionError,
        match="cannot be empty",
    ):
        evaluate_lane_fit(
            fit,
            empty,
        )


def test_finds_lane_pixels_near_previous_fit() -> None:
    height = 80
    width = 100
    image = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        curve_offset = round(0.003 * (y - 40) ** 2)
        left_x = 20 + curve_offset
        right_x = 70 + curve_offset

        image[y, left_x - 1 : left_x + 2] = 255
        image[y, right_x - 1 : right_x + 2] = 255

    # Noise far from both expected lane curves.
    image[:, 45] = 255

    previous_fit = LaneFit(
        left_coefficients=np.asarray(
            [0.003, -0.24, 24.8],
            dtype=np.float64,
        ),
        right_coefficients=np.asarray(
            [0.003, -0.24, 74.8],
            dtype=np.float64,
        ),
    )

    result = find_lane_pixels_near_fit(
        image,
        previous_fit,
        _search_config(),
    )

    assert result.left_count == height * 3
    assert result.right_count == height * 3
    assert not np.any(result.left_x == 45)
    assert not np.any(result.right_x == 45)


def test_near_fit_search_returns_empty_lanes_for_empty_mask() -> None:
    image = np.zeros((80, 100), dtype=np.uint8)

    previous_fit = LaneFit(
        left_coefficients=np.asarray(
            [0.0, 0.0, 20.0],
            dtype=np.float64,
        ),
        right_coefficients=np.asarray(
            [0.0, 0.0, 70.0],
            dtype=np.float64,
        ),
    )

    result = find_lane_pixels_near_fit(
        image,
        previous_fit,
        _search_config(),
    )

    assert result.left_count == 0
    assert result.right_count == 0
