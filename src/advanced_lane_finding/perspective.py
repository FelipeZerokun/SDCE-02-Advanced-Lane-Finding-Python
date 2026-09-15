from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from advanced_lane_finding.config import (
    PerspectiveConfig,
    Quadrilateral,
)

FloatArray = NDArray[np.float64]
BinaryImage = NDArray[np.uint8]
ColorImage = NDArray[np.uint8]


class PerspectiveError(ValueError):
    """Raised when a perspective transform cannot be created or applied."""


@dataclass(frozen=True, slots=True)
class PerspectiveTransform:
    matrix: FloatArray
    inverse_matrix: FloatArray
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        if self.matrix.shape != (3, 3):
            raise PerspectiveError("Perspective matrix must have shape (3, 3)")

        if self.inverse_matrix.shape != (3, 3):
            raise PerspectiveError("Inverse perspective matrix must have shape (3, 3)")

        if not np.all(np.isfinite(self.matrix)):
            raise PerspectiveError("Perspective matrix must contain only finite values")

        if not np.all(np.isfinite(self.inverse_matrix)):
            raise PerspectiveError(
                "Inverse perspective matrix must contain only finite values"
            )

        width, height = self.image_size

        if width <= 0 or height <= 0:
            raise PerspectiveError("Perspective image dimensions must be positive")


def _normalized_points_to_pixels(
    points: Quadrilateral,
    image_size: tuple[int, int],
) -> NDArray[np.float32]:
    width, height = image_size

    if width <= 0 or height <= 0:
        raise PerspectiveError("Perspective image dimensions must be positive")

    normalized = np.asarray(
        points,
        dtype=np.float32,
    )
    scale = np.asarray(
        [width - 1, height - 1],
        dtype=np.float32,
    )

    return np.asarray(
        normalized * scale,
        dtype=np.float32,
    )


def create_perspective_transform(
    image_size: tuple[int, int],
    config: PerspectiveConfig,
) -> PerspectiveTransform:
    """Create forward and inverse perspective matrices."""

    source_points = _normalized_points_to_pixels(
        config.source,
        image_size,
    )
    destination_points = _normalized_points_to_pixels(
        config.destination,
        image_size,
    )

    matrix = np.asarray(
        cv2.getPerspectiveTransform(
            source_points,
            destination_points,
        ),
        dtype=np.float64,
    )

    inverse_matrix = np.asarray(
        cv2.getPerspectiveTransform(
            destination_points,
            source_points,
        ),
        dtype=np.float64,
    )

    return PerspectiveTransform(
        matrix=matrix,
        inverse_matrix=inverse_matrix,
        image_size=image_size,
    )


def warp_binary_image(
    image: BinaryImage,
    transform: PerspectiveTransform,
) -> BinaryImage:
    """Warp a binary mask into a bird's-eye view."""

    if image.ndim != 2:
        raise PerspectiveError("Expected a single-channel binary image")

    if image.dtype != np.uint8:
        raise PerspectiveError("Expected a binary image with uint8 data")

    if image.size == 0:
        raise PerspectiveError("Binary image cannot be empty")

    height, width = image.shape

    if (width, height) != transform.image_size:
        raise PerspectiveError(
            "Input image dimensions do not match the perspective transform"
        )

    warped = cv2.warpPerspective(
        image,
        transform.matrix,
        transform.image_size,
        flags=cv2.INTER_NEAREST,
    )

    return np.asarray(
        warped,
        dtype=np.uint8,
    )


def unwarp_color_image(
    image: ColorImage,
    transform: PerspectiveTransform,
) -> ColorImage:
    """Transform a color image from bird's-eye view back to camera view."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise PerspectiveError("Expected a three-channel BGR image")

    if image.dtype != np.uint8:
        raise PerspectiveError("Expected a color image with uint8 data")

    if image.size == 0:
        raise PerspectiveError("Color image cannot be empty")

    height, width = image.shape[:2]

    if (width, height) != transform.image_size:
        raise PerspectiveError(
            "Input image dimensions do not match the perspective transform"
        )

    unwarped = cv2.warpPerspective(
        image,
        transform.inverse_matrix,
        transform.image_size,
        flags=cv2.INTER_LINEAR,
    )

    return np.asarray(
        unwarped,
        dtype=np.uint8,
    )
