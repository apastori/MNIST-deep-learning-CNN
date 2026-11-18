"""Tests for MNIST image augmentation helpers."""

from __future__ import annotations

import numpy as np

from mnistnn.utils.images import (
    _elastic_distort_image,
    _scale_image,
    random_shift_and_rotate,
)


def test_random_shift_and_rotate_preserves_shape_and_range() -> None:
    """Augmentation keeps (N, 1, 28, 28) and roughly [-0.5, 0.5] values."""
    rng: np.random.Generator = np.random.default_rng(seed=42)
    images: np.ndarray = rng.uniform(-0.5, 0.5, size=(8, 1, 28, 28))
    augmented: np.ndarray = random_shift_and_rotate(images, rng)
    assert augmented.shape == (8, 1, 28, 28)
    assert augmented.dtype == np.float64
    assert float(augmented.min()) >= -0.6
    assert float(augmented.max()) <= 0.6


def test_random_shift_and_rotate_is_deterministic_with_seed() -> None:
    """The same RNG seed yields identical augmented batches."""
    images: np.ndarray = np.random.default_rng(0).uniform(
        -0.5, 0.5, size=(4, 1, 28, 28)
    )
    first: np.ndarray = random_shift_and_rotate(
        images, np.random.default_rng(seed=123)
    )
    second: np.ndarray = random_shift_and_rotate(
        images, np.random.default_rng(seed=123)
    )
    np.testing.assert_array_equal(first, second)


def test_elastic_distort_preserves_shape() -> None:
    """Elastic distortion returns a same-shaped 28x28 plane."""
    rng: np.random.Generator = np.random.default_rng(seed=7)
    image: np.ndarray = rng.uniform(-0.5, 0.5, size=(28, 28))
    distorted: np.ndarray = _elastic_distort_image(
        image, rng, alpha=18.0, sigma=3.0, fill_value=-0.5
    )
    assert distorted.shape == (28, 28)
    assert distorted.dtype == np.float64


def test_scale_image_identity_and_zoom() -> None:
    """Scale 1.0 is identity; other scales stay 28x28."""
    image: np.ndarray = np.linspace(-0.5, 0.5, 28 * 28).reshape(28, 28)
    identity: np.ndarray = _scale_image(image, 1.0, fill_value=-0.5)
    np.testing.assert_allclose(identity, image)
    zoomed: np.ndarray = _scale_image(image, 1.1, fill_value=-0.5)
    assert zoomed.shape == (28, 28)
