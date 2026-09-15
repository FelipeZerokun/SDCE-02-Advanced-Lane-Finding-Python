from typing import cast

import numpy as np
import pytest

from advanced_lane_finding.config import ThresholdConfig
from advanced_lane_finding.thresholding import (
    ColorImage,
    ThresholdingError,
    absolute_sobel_threshold,
    combined_threshold,
    gradient_direction_threshold,
    gradient_magnitude_threshold,
    hls_color_threshold,
)


def test_detects_vertical_edge() -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    image[:, 40:] = 255

    result = absolute_sobel_threshold(
        image,
        threshold=(20, 255),
        kernel_size=3,
        orientation="x",
    )

    assert result.shape == image.shape[:2]
    assert result.dtype == np.uint8
    assert np.count_nonzero(result) > 0
    assert set(np.unique(result)) <= {0, 255}


def test_constant_image_produces_empty_mask() -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)

    result = absolute_sobel_threshold(
        image,
        threshold=(20, 255),
        kernel_size=3,
    )

    assert not np.any(result)


def test_rejects_grayscale_image() -> None:
    grayscale = np.zeros((60, 80), dtype=np.uint8)

    with pytest.raises(
        ThresholdingError,
        match="three-channel BGR",
    ):
        absolute_sobel_threshold(
            cast(ColorImage, grayscale),
            threshold=(20, 255),
            kernel_size=3,
        )


def test_rejects_non_uint8_image() -> None:
    image = np.zeros((60, 80, 3), dtype=np.float32)

    with pytest.raises(
        ThresholdingError,
        match="uint8",
    ):
        absolute_sobel_threshold(
            cast(ColorImage, image),
            threshold=(20, 255),
            kernel_size=3,
        )


def test_gradient_magnitude_detects_edge() -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    image[:, 40:] = 255

    result = gradient_magnitude_threshold(
        image,
        threshold=(20, 255),
        kernel_size=3,
    )

    assert result.shape == image.shape[:2]
    assert result.dtype == np.uint8
    assert np.count_nonzero(result) > 0
    assert set(np.unique(result)) <= {0, 255}


def test_gradient_direction_detects_diagonal_edge() -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)

    for row in range(image.shape[0]):
        image[row, row:] = 255

    result = gradient_direction_threshold(
        image,
        threshold=(0.6, 0.9),
        kernel_size=3,
    )

    assert result.shape == image.shape[:2]
    assert result.dtype == np.uint8
    assert np.count_nonzero(result) > 0
    assert set(np.unique(result)) <= {0, 255}


def test_hls_detects_white_and_yellow_regions() -> None:
    image = np.zeros((60, 100, 3), dtype=np.uint8)

    # White in BGR: high lightness, low saturation.
    image[10:30, 10:30] = (255, 255, 255)

    # Yellow in BGR: moderate lightness, high saturation.
    image[10:30, 60:80] = (0, 255, 255)

    result = hls_color_threshold(
        image,
        lightness_threshold=(200, 255),
        saturation_threshold=(140, 255),
    )

    assert result.shape == image.shape[:2]
    assert result.dtype == np.uint8
    assert np.all(result[10:30, 10:30] == 255)
    assert np.all(result[10:30, 60:80] == 255)
    assert np.all(result[40:, :] == 0)
    assert set(np.unique(result)) <= {0, 255}


def test_combined_threshold_detects_gradient_and_color() -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)

    # A neutral vertical region produces x-gradient edges.
    image[:, 30:50] = (100, 100, 100)

    # A white region should be selected through HLS lightness.
    image[20:40, 80:100] = (255, 255, 255)

    config = ThresholdConfig(
        sobel_kernel=3,
        gradient_x=(50, 255),
        gradient_magnitude=(50, 255),
        gradient_direction=(0.6, 0.9),
        hls_saturation=(200, 255),
        hls_lightness=(240, 255),
    )

    result = combined_threshold(image, config)

    assert result.shape == image.shape[:2]
    assert result.dtype == np.uint8
    assert set(np.unique(result)) <= {0, 255}

    # The neutral region's vertical boundaries are detected.
    assert np.any(result[:, 28:32])
    assert np.any(result[:, 48:52])

    # The white region is selected by its HLS lightness.
    assert np.all(result[20:40, 80:100] == 255)


def test_combined_threshold_rejects_black_image() -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)

    config = ThresholdConfig(
        sobel_kernel=3,
        gradient_x=(30, 200),
        gradient_magnitude=(70, 170),
        gradient_direction=(0.3, 1.1),
        hls_saturation=(140, 255),
        hls_lightness=(200, 255),
    )

    result = combined_threshold(image, config)

    assert not np.any(result)
