from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from advanced_lane_finding.config import ThresholdConfig

ColorImage = NDArray[np.uint8]
BinaryImage = NDArray[np.uint8]
FloatImage = NDArray[np.float64]


class ThresholdingError(ValueError):
    """Raised when an image cannot be thresholded."""


def _validate_color_image(image: ColorImage) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ThresholdingError("Expected a three-channel BGR image")

    if image.dtype != np.uint8:
        raise ThresholdingError("Expected an image with uint8 data")

    if image.size == 0:
        raise ThresholdingError("Image cannot be empty")


def _scale_to_uint8(values: FloatImage) -> BinaryImage:
    maximum = float(np.max(values))

    if maximum <= 0:
        return np.zeros(values.shape, dtype=np.uint8)

    scaled = np.rint(values * (255.0 / maximum))
    return np.asarray(scaled, dtype=np.uint8)


def absolute_sobel_threshold(
    image: ColorImage,
    threshold: tuple[int, int],
    kernel_size: int,
    *,
    orientation: Literal["x", "y"] = "x",
) -> BinaryImage:
    """Apply an absolute Sobel threshold in one direction."""
    _validate_color_image(image)

    gray = np.asarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        dtype=np.uint8,
    )

    if orientation == "x":
        dx, dy = 1, 0
    else:
        dx, dy = 0, 1

    gradient = np.asarray(
        cv2.Sobel(
            gray,
            cv2.CV_64F,
            dx,
            dy,
            ksize=kernel_size,
        ),
        dtype=np.float64,
    )

    absolute_gradient = np.abs(gradient)
    scaled_gradient = _scale_to_uint8(absolute_gradient)

    low, high = threshold
    binary = np.zeros(gray.shape, dtype=np.uint8)
    binary[(scaled_gradient >= low) & (scaled_gradient <= high)] = 255

    return binary


def gradient_magnitude_threshold(
    image: ColorImage,
    threshold: tuple[int, int],
    kernel_size: int,
) -> BinaryImage:
    """Select pixels using combined Sobel gradient magnitude."""
    _validate_color_image(image)

    gray = np.asarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        dtype=np.uint8,
    )

    gradient_x = np.asarray(
        cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0,
            ksize=kernel_size,
        ),
        dtype=np.float64,
    )
    gradient_y = np.asarray(
        cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=kernel_size,
        ),
        dtype=np.float64,
    )

    magnitude = np.hypot(gradient_x, gradient_y)
    scaled_magnitude = _scale_to_uint8(magnitude)

    low, high = threshold
    binary = np.zeros(gray.shape, dtype=np.uint8)
    binary[(scaled_magnitude >= low) & (scaled_magnitude <= high)] = 255

    return binary


def gradient_direction_threshold(
    image: ColorImage,
    threshold: tuple[float, float],
    kernel_size: int,
) -> BinaryImage:
    """Select pixels using Sobel gradient direction."""
    _validate_color_image(image)

    gray = np.asarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        dtype=np.uint8,
    )

    gradient_x = np.asarray(
        cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0,
            ksize=kernel_size,
        ),
        dtype=np.float64,
    )
    gradient_y = np.asarray(
        cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=kernel_size,
        ),
        dtype=np.float64,
    )

    direction = np.arctan2(
        np.abs(gradient_y),
        np.abs(gradient_x),
    )

    low, high = threshold
    binary = np.zeros(gray.shape, dtype=np.uint8)
    binary[(direction >= low) & (direction <= high)] = 255

    return binary


def hls_color_threshold(
    image: ColorImage,
    lightness_threshold: tuple[int, int],
    saturation_threshold: tuple[int, int],
) -> BinaryImage:
    """Select bright or strongly saturated HLS pixels."""
    _validate_color_image(image)

    hls = np.asarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2HLS),
        dtype=np.uint8,
    )

    lightness = hls[:, :, 1]
    saturation = hls[:, :, 2]

    lightness_low, lightness_high = lightness_threshold
    saturation_low, saturation_high = saturation_threshold

    lightness_selected = (lightness >= lightness_low) & (lightness <= lightness_high)
    saturation_selected = (saturation >= saturation_low) & (
        saturation <= saturation_high
    )

    binary = np.zeros(image.shape[:2], dtype=np.uint8)
    binary[lightness_selected | saturation_selected] = 255

    return binary


def combined_threshold(
    image: ColorImage,
    config: ThresholdConfig,
) -> BinaryImage:
    """Create the combined binary lane-pixel mask."""
    _validate_color_image(image)

    gradient_x = absolute_sobel_threshold(
        image,
        threshold=config.gradient_x,
        kernel_size=config.sobel_kernel,
        orientation="x",
    )

    magnitude = gradient_magnitude_threshold(
        image,
        threshold=config.gradient_magnitude,
        kernel_size=config.sobel_kernel,
    )

    direction = gradient_direction_threshold(
        image,
        threshold=config.gradient_direction,
        kernel_size=config.sobel_kernel,
    )

    color = hls_color_threshold(
        image,
        lightness_threshold=config.hls_lightness,
        saturation_threshold=config.hls_saturation,
    )

    gradient_selected = (gradient_x != 0) | ((magnitude != 0) & (direction != 0))
    color_selected = color != 0

    combined = np.zeros(image.shape[:2], dtype=np.uint8)
    combined[gradient_selected | color_selected] = 255

    return combined
