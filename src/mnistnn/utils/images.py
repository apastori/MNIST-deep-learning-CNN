"""Provide digit-safe image augmentation helpers for 28x28 MNIST batches."""

from __future__ import annotations

import numpy as np
from PIL import Image

MAX_SHIFT_PIXELS: int = 2
MAX_ROTATION_DEGREES: float = 8.0
AUGMENT_FILL_VALUE: float = -0.5
AUGMENT_PROBABILITY: float = 0.5


def _shift_image(
    image: np.ndarray, shift_y: int, shift_x: int, fill_value: float
) -> np.ndarray:
    """Translate one 28x28 image, filling empty borders with ``fill_value``."""
    height: int
    width: int
    height, width = image.shape
    shifted: np.ndarray = np.full(
        (height, width), fill_value, dtype=image.dtype
    )
    src_y_start: int = max(0, -shift_y)
    src_y_end: int = min(height, height - shift_y)
    src_x_start: int = max(0, -shift_x)
    src_x_end: int = min(width, width - shift_x)
    dst_y_start: int = max(0, shift_y)
    dst_y_end: int = min(height, height + shift_y)
    dst_x_start: int = max(0, shift_x)
    dst_x_end: int = min(width, width + shift_x)
    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image[
        src_y_start:src_y_end, src_x_start:src_x_end
    ]
    return shifted


def _rotate_image(
    image: np.ndarray, angle_degrees: float, fill_value: float
) -> np.ndarray:
    """Rotate one 28x28 image with bilinear resampling around its center."""
    # Map mean-centered pixels [-0.5, 0.5] to [0, 255] for Pillow, then back.
    scaled: np.ndarray = np.clip((image + 0.5) * 255.0, 0.0, 255.0)
    pil_image: Image.Image = Image.fromarray(scaled.astype(np.uint8), mode='L')
    fill_uint8: int = int(np.clip((fill_value + 0.5) * 255.0, 0.0, 255.0))
    rotated: Image.Image = pil_image.rotate(
        angle_degrees,
        resample=Image.Resampling.BILINEAR,
        fillcolor=fill_uint8,
    )
    rotated_array: np.ndarray = np.asarray(rotated, dtype=np.float64)
    return rotated_array / 255.0 - 0.5


def random_shift_and_rotate(
    images: np.ndarray,
    rng: np.random.Generator,
    max_shift: int = MAX_SHIFT_PIXELS,
    max_rotation_degrees: float = MAX_ROTATION_DEGREES,
    fill_value: float = AUGMENT_FILL_VALUE,
) -> np.ndarray:
    """Apply random ±shift and ±rotation to each flattened MNIST image.

    Each transform is applied independently with probability
    ``AUGMENT_PROBABILITY``. Intended for training batches only. Images must
    already be normalized to approximately ``[-0.5, 0.5]``.
    """
    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError(
            f'Images to augment must have shape (N, 784), got {images.shape}'
        )
    if max_shift < 0:
        raise ValueError('max_shift must be non-negative')
    if max_rotation_degrees < 0.0:
        raise ValueError('max_rotation_degrees must be non-negative')

    batch: np.ndarray = images.reshape(-1, 28, 28).copy()
    for index in range(len(batch)):
        augmented: np.ndarray = batch[index]
        if rng.random() < AUGMENT_PROBABILITY:
            shift_y: int = int(rng.integers(-max_shift, max_shift + 1))
            shift_x: int = int(rng.integers(-max_shift, max_shift + 1))
            if shift_y != 0 or shift_x != 0:
                augmented = _shift_image(
                    augmented, shift_y, shift_x, fill_value
                )
        if rng.random() < AUGMENT_PROBABILITY:
            angle: float = float(
                rng.uniform(-max_rotation_degrees, max_rotation_degrees)
            )
            if abs(angle) > 1e-8:
                augmented = _rotate_image(augmented, angle, fill_value)
        batch[index] = augmented
    return batch.reshape(-1, 784)
