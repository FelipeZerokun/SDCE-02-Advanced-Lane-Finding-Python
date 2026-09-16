from typing import cast

import numpy as np
import pytest

from advanced_lane_finding.config import PerspectiveConfig
from advanced_lane_finding.perspective import (
    BinaryImage,
    ColorImage,
    PerspectiveError,
    create_perspective_transform,
    unwarp_color_image,
    warp_binary_image,
)


def _identity_config() -> PerspectiveConfig:
    corners = (
        (0.0, 1.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
    )

    return PerspectiveConfig(
        source=corners,
        destination=corners,
    )


def test_creates_identity_perspective_transform() -> None:
    transform = create_perspective_transform(
        (8, 6),
        _identity_config(),
    )

    np.testing.assert_allclose(
        transform.matrix,
        np.eye(3),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        transform.inverse_matrix,
        np.eye(3),
        atol=1e-12,
    )
    assert transform.image_size == (8, 6)


def test_identity_transform_preserves_binary_image() -> None:
    image = np.zeros((6, 8), dtype=np.uint8)
    image[1:5, 3:6] = 255

    transform = create_perspective_transform(
        (8, 6),
        _identity_config(),
    )

    result = warp_binary_image(
        image,
        transform,
    )

    np.testing.assert_array_equal(result, image)
    assert result.dtype == np.uint8


def test_rejects_image_with_wrong_dimensions() -> None:
    transform = create_perspective_transform(
        (8, 6),
        _identity_config(),
    )
    image = np.zeros((5, 8), dtype=np.uint8)

    with pytest.raises(
        PerspectiveError,
        match="dimensions do not match",
    ):
        warp_binary_image(image, transform)


def test_rejects_three_channel_image() -> None:
    transform = create_perspective_transform(
        (8, 6),
        _identity_config(),
    )
    image = np.zeros((6, 8, 3), dtype=np.uint8)

    with pytest.raises(
        PerspectiveError,
        match="single-channel",
    ):
        warp_binary_image(
            cast(BinaryImage, image),
            transform,
        )


def test_rejects_non_uint8_binary_image() -> None:
    transform = create_perspective_transform(
        (8, 6),
        _identity_config(),
    )
    image = np.zeros((6, 8), dtype=np.float32)

    with pytest.raises(
        PerspectiveError,
        match="uint8",
    ):
        warp_binary_image(
            cast(BinaryImage, image),
            transform,
        )


def test_rejects_empty_binary_image() -> None:
    transform = create_perspective_transform(
        (8, 6),
        _identity_config(),
    )
    image = np.zeros((0, 0), dtype=np.uint8)

    with pytest.raises(
        PerspectiveError,
        match="cannot be empty",
    ):
        warp_binary_image(
            image,
            transform,
        )


def test_maps_trapezoid_to_destination_rectangle() -> None:
    config = PerspectiveConfig(
        source=(
            (0.2, 1.0),
            (0.4, 0.3),
            (0.6, 0.3),
            (0.8, 1.0),
        ),
        destination=(
            (0.2, 1.0),
            (0.2, 0.0),
            (0.8, 0.0),
            (0.8, 1.0),
        ),
    )

    transform = create_perspective_transform(
        (101, 101),
        config,
    )

    source_points = np.asarray(
        [
            [20.0, 100.0, 1.0],
            [40.0, 30.0, 1.0],
            [60.0, 30.0, 1.0],
            [80.0, 100.0, 1.0],
        ],
        dtype=np.float64,
    ).T

    expected_destination = np.asarray(
        [
            [20.0, 100.0],
            [20.0, 0.0],
            [80.0, 0.0],
            [80.0, 100.0],
        ],
        dtype=np.float64,
    )

    transformed = transform.matrix @ source_points
    transformed /= transformed[2]

    np.testing.assert_allclose(
        transformed[:2].T,
        expected_destination,
        atol=1e-5,
    )

    identity = transform.inverse_matrix @ transform.matrix
    identity /= identity[2, 2]

    np.testing.assert_allclose(
        identity,
        np.eye(3),
        atol=1e-5,
    )


def test_identity_inverse_transform_preserves_color_image() -> None:
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    image[1:5, 2:6] = (20, 140, 255)

    transform = create_perspective_transform(
        (8, 6),
        _identity_config(),
    )

    result = unwarp_color_image(
        image,
        transform,
    )

    np.testing.assert_array_equal(result, image)
    assert result.dtype == np.uint8


def test_inverse_transform_rejects_single_channel_image() -> None:
    transform = create_perspective_transform(
        (8, 6),
        _identity_config(),
    )
    image = np.zeros((6, 8), dtype=np.uint8)

    with pytest.raises(
        PerspectiveError,
        match="three-channel",
    ):
        unwarp_color_image(
            cast(ColorImage, image),
            transform,
        )


def test_inverse_transform_rejects_non_uint8_image() -> None:
    transform = create_perspective_transform(
        (8, 6),
        _identity_config(),
    )
    image = np.zeros((6, 8, 3), dtype=np.float32)

    with pytest.raises(
        PerspectiveError,
        match="uint8",
    ):
        unwarp_color_image(
            cast(ColorImage, image),
            transform,
        )
