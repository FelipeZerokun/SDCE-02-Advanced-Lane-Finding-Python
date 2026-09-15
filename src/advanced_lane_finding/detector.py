from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from advanced_lane_finding.config import SearchConfig

BinaryImage = NDArray[np.uint8]
IndexArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


class DetectionError(RuntimeError):
    """Raised when lane pixels or lane curves cannot be detected."""


@dataclass(frozen=True, slots=True)
class LanePixels:
    left_x: IndexArray
    left_y: IndexArray
    right_x: IndexArray
    right_y: IndexArray

    def __post_init__(self) -> None:
        arrays = (
            self.left_x,
            self.left_y,
            self.right_x,
            self.right_y,
        )

        if any(array.ndim != 1 for array in arrays):
            raise DetectionError("Lane pixel coordinates must be one-dimensional")

        if self.left_x.size != self.left_y.size:
            raise DetectionError(
                "Left lane x and y coordinates must have equal lengths"
            )

        if self.right_x.size != self.right_y.size:
            raise DetectionError(
                "Right lane x and y coordinates must have equal lengths"
            )

    @property
    def left_count(self) -> int:
        return int(self.left_x.size)

    @property
    def right_count(self) -> int:
        return int(self.right_x.size)


@dataclass(frozen=True, slots=True)
class LaneFit:
    left_coefficients: FloatArray
    right_coefficients: FloatArray

    def __post_init__(self) -> None:
        if self.left_coefficients.shape != (3,):
            raise DetectionError("Left lane polynomial must contain three coefficients")

        if self.right_coefficients.shape != (3,):
            raise DetectionError(
                "Right lane polynomial must contain three coefficients"
            )

        if not np.all(np.isfinite(self.left_coefficients)):
            raise DetectionError("Left lane polynomial must contain only finite values")

        if not np.all(np.isfinite(self.right_coefficients)):
            raise DetectionError(
                "Right lane polynomial must contain only finite values"
            )


def _validate_binary_image(image: BinaryImage) -> None:
    if image.ndim != 2:
        raise DetectionError("Expected a single-channel binary image")

    if image.dtype != np.uint8:
        raise DetectionError("Expected a binary image with uint8 data")

    if image.size == 0:
        raise DetectionError("Binary image cannot be empty")


def lower_half_histogram(
    image: BinaryImage,
) -> IndexArray:
    """Count selected pixels in each column of the image's lower half."""

    _validate_binary_image(image)

    height = image.shape[0]
    lower_half = image[height // 2 :, :]

    histogram = np.count_nonzero(
        lower_half,
        axis=0,
    )

    return np.asarray(
        histogram,
        dtype=np.int64,
    )


def find_lane_pixels(
    image: BinaryImage,
    config: SearchConfig,
) -> LanePixels:
    """Find left and right lane pixels using sliding windows."""

    _validate_binary_image(image)

    height, width = image.shape

    if width < 2:
        raise DetectionError("Binary image must be at least two pixels wide")

    if config.windows > height:
        raise DetectionError("Number of sliding windows cannot exceed image height")

    histogram = lower_half_histogram(image)
    midpoint = width // 2

    left_histogram = histogram[:midpoint]
    right_histogram = histogram[midpoint:]

    if not np.any(left_histogram):
        raise DetectionError("Could not locate the left lane base")

    if not np.any(right_histogram):
        raise DetectionError("Could not locate the right lane base")

    left_current = int(np.argmax(left_histogram))
    right_current = int(np.argmax(right_histogram)) + midpoint

    nonzero_y_raw, nonzero_x_raw = np.nonzero(image)

    nonzero_x = np.asarray(
        nonzero_x_raw,
        dtype=np.int64,
    )
    nonzero_y = np.asarray(
        nonzero_y_raw,
        dtype=np.int64,
    )

    left_indices: list[IndexArray] = []
    right_indices: list[IndexArray] = []

    window_boundaries = np.linspace(
        height,
        0,
        config.windows + 1,
        dtype=np.int64,
    )

    for window in range(config.windows):
        y_high = int(window_boundaries[window])
        y_low = int(window_boundaries[window + 1])

        left_x_low = max(0, left_current - config.margin)
        left_x_high = min(
            width,
            left_current + config.margin + 1,
        )
        right_x_low = max(0, right_current - config.margin)
        right_x_high = min(
            width,
            right_current + config.margin + 1,
        )

        left_window = (
            (nonzero_y >= y_low)
            & (nonzero_y < y_high)
            & (nonzero_x >= left_x_low)
            & (nonzero_x < left_x_high)
        )
        right_window = (
            (nonzero_y >= y_low)
            & (nonzero_y < y_high)
            & (nonzero_x >= right_x_low)
            & (nonzero_x < right_x_high)
        )

        left_found = np.asarray(
            np.flatnonzero(left_window),
            dtype=np.int64,
        )
        right_found = np.asarray(
            np.flatnonzero(right_window),
            dtype=np.int64,
        )

        left_indices.append(left_found)
        right_indices.append(right_found)

        if left_found.size >= config.minimum_pixels:
            left_current = int(np.rint(np.mean(nonzero_x[left_found])))

        if right_found.size >= config.minimum_pixels:
            right_current = int(np.rint(np.mean(nonzero_x[right_found])))

    all_left_indices = np.concatenate(left_indices)
    all_right_indices = np.concatenate(right_indices)

    return LanePixels(
        left_x=nonzero_x[all_left_indices],
        left_y=nonzero_y[all_left_indices],
        right_x=nonzero_x[all_right_indices],
        right_y=nonzero_y[all_right_indices],
    )


def fit_lane_polynomials(
    pixels: LanePixels,
    minimum_pixels: int,
) -> LaneFit:
    """Fit second-degree polynomials to left and right lane pixels."""

    if minimum_pixels < 3:
        raise DetectionError(
            "At least three pixels must be required for polynomial fitting"
        )

    if pixels.left_count < minimum_pixels:
        raise DetectionError("Not enough left lane pixels for polynomial fitting")

    if pixels.right_count < minimum_pixels:
        raise DetectionError("Not enough right lane pixels for polynomial fitting")

    left_coefficients = np.asarray(
        np.polyfit(
            pixels.left_y,
            pixels.left_x,
            2,
        ),
        dtype=np.float64,
    )

    right_coefficients = np.asarray(
        np.polyfit(
            pixels.right_y,
            pixels.right_x,
            2,
        ),
        dtype=np.float64,
    )

    return LaneFit(
        left_coefficients=left_coefficients,
        right_coefficients=right_coefficients,
    )


def evaluate_lane_fit(
    fit: LaneFit,
    y_values: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate both lane polynomials at the supplied y-coordinates."""

    if y_values.ndim != 1:
        raise DetectionError("Polynomial y-coordinates must be one-dimensional")

    if y_values.size == 0:
        raise DetectionError("Polynomial y-coordinates cannot be empty")

    if not np.all(np.isfinite(y_values)):
        raise DetectionError("Polynomial y-coordinates must contain only finite values")

    left_x = np.asarray(
        np.polyval(fit.left_coefficients, y_values),
        dtype=np.float64,
    )
    right_x = np.asarray(
        np.polyval(fit.right_coefficients, y_values),
        dtype=np.float64,
    )

    return left_x, right_x


def find_lane_pixels_near_fit(
    image: BinaryImage,
    previous_fit: LaneFit,
    config: SearchConfig,
) -> LanePixels:
    """Find lane pixels near previously fitted lane curves."""

    _validate_binary_image(image)

    nonzero_y_raw, nonzero_x_raw = np.nonzero(image)

    nonzero_x = np.asarray(
        nonzero_x_raw,
        dtype=np.int64,
    )
    nonzero_y = np.asarray(
        nonzero_y_raw,
        dtype=np.int64,
    )

    left_centers = np.asarray(
        np.polyval(
            previous_fit.left_coefficients,
            nonzero_y,
        ),
        dtype=np.float64,
    )
    right_centers = np.asarray(
        np.polyval(
            previous_fit.right_coefficients,
            nonzero_y,
        ),
        dtype=np.float64,
    )

    left_distances = np.abs(nonzero_x - left_centers)
    right_distances = np.abs(nonzero_x - right_centers)

    margin = config.previous_fit_margin

    left_selected = (left_distances <= margin) & (left_distances < right_distances)
    right_selected = (right_distances <= margin) & (right_distances < left_distances)

    return LanePixels(
        left_x=nonzero_x[left_selected],
        left_y=nonzero_y[left_selected],
        right_x=nonzero_x[right_selected],
        right_y=nonzero_y[right_selected],
    )
