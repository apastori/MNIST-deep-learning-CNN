"""Provide digit-safe image augmentation helpers for (N, 1, 28, 28) MNIST batches."""

from __future__ import annotations

import numpy as np
from PIL import Image

MAX_SHIFT_PIXELS: int = 2
MAX_ROTATION_DEGREES: float = 8.0
AUGMENT_FILL_VALUE: float = -0.5
AUGMENT_PROBABILITY: float = 0.5
ELASTIC_ALPHA: float = 18.0
ELASTIC_SIGMA: float = 3.0
SCALE_RANGE_MIN: float = 0.9
SCALE_RANGE_MAX: float = 1.1


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


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Build a normalized 1D Gaussian kernel covering about ±3 sigma."""
    if sigma <= 0.0:
        raise ValueError('sigma must be positive')
    radius: int = max(1, int(np.ceil(3.0 * sigma)))
    offsets: np.ndarray = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel: np.ndarray = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel


def _gaussian_filter2d(field: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a separable Gaussian blur to a 2D array."""
    kernel: np.ndarray = _gaussian_kernel_1d(sigma)
    radius: int = (kernel.size - 1) // 2
    padded_y: np.ndarray = np.pad(field, ((radius, radius), (0, 0)), mode='reflect')
    blurred_y: np.ndarray = np.empty_like(field, dtype=np.float64)
    for index, weight in enumerate(kernel):
        if index == 0:
            blurred_y = weight * padded_y[index : index + field.shape[0]]
        else:
            blurred_y += weight * padded_y[index : index + field.shape[0]]
    padded_x: np.ndarray = np.pad(
        blurred_y, ((0, 0), (radius, radius)), mode='reflect'
    )
    blurred: np.ndarray = np.empty_like(field, dtype=np.float64)
    for index, weight in enumerate(kernel):
        if index == 0:
            blurred = weight * padded_x[:, index : index + field.shape[1]]
        else:
            blurred += weight * padded_x[:, index : index + field.shape[1]]
    return blurred


def _bilinear_sample(
    image: np.ndarray, sample_y: np.ndarray, sample_x: np.ndarray, fill_value: float
) -> np.ndarray:
    """Sample ``image`` at floating coordinates with bilinear interpolation."""
    height: int
    width: int
    height, width = image.shape
    y0: np.ndarray = np.floor(sample_y).astype(np.int64)
    x0: np.ndarray = np.floor(sample_x).astype(np.int64)
    y1: np.ndarray = y0 + 1
    x1: np.ndarray = x0 + 1
    wy: np.ndarray = sample_y - y0.astype(np.float64)
    wx: np.ndarray = sample_x - x0.astype(np.float64)

    sampled: np.ndarray = np.zeros((height, width), dtype=np.float64)
    for src_y, src_x, weight_y, weight_x in (
        (y0, x0, 1.0 - wy, 1.0 - wx),
        (y0, x1, 1.0 - wy, wx),
        (y1, x0, wy, 1.0 - wx),
        (y1, x1, wy, wx),
    ):
        valid: np.ndarray = (
            (src_y >= 0)
            & (src_y < height)
            & (src_x >= 0)
            & (src_x < width)
        )
        weights: np.ndarray = weight_y * weight_x
        if np.any(valid):
            sampled[valid] += (
                image[src_y[valid], src_x[valid]] * weights[valid]
            )
        if np.any(~valid):
            # Outside the image, contribute fill_value for the missing corner.
            sampled[~valid] += fill_value * weights[~valid]
    return sampled


def _elastic_distort_image(
    image: np.ndarray,
    rng: np.random.Generator,
    alpha: float,
    sigma: float,
    fill_value: float,
) -> np.ndarray:
    """Apply Simard-style elastic distortion to one 28x28 image."""
    height: int
    width: int
    height, width = image.shape
    # Classic Simard elastic: smooth N(0,1) displacement fields, then * alpha.
    dx: np.ndarray = (
        _gaussian_filter2d(
            rng.normal(loc=0.0, scale=1.0, size=(height, width)), sigma
        )
        * alpha
    )
    dy: np.ndarray = (
        _gaussian_filter2d(
            rng.normal(loc=0.0, scale=1.0, size=(height, width)), sigma
        )
        * alpha
    )
    grid_y: np.ndarray
    grid_x: np.ndarray
    grid_y, grid_x = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing='ij',
    )
    return _bilinear_sample(image, grid_y + dy, grid_x + dx, fill_value)


def _scale_image(
    image: np.ndarray, scale: float, fill_value: float
) -> np.ndarray:
    """Zoom an image about its center and crop/pad back to 28x28."""
    if abs(scale - 1.0) < 1e-8:
        return image
    height: int
    width: int
    height, width = image.shape
    scaled_pixels: np.ndarray = np.clip((image + 0.5) * 255.0, 0.0, 255.0)
    pil_image: Image.Image = Image.fromarray(
        scaled_pixels.astype(np.uint8), mode='L'
    )
    new_height: int = max(1, int(round(height * scale)))
    new_width: int = max(1, int(round(width * scale)))
    resized: Image.Image = pil_image.resize(
        (new_width, new_height), resample=Image.Resampling.BILINEAR
    )
    resized_array: np.ndarray = (
        np.asarray(resized, dtype=np.float64) / 255.0 - 0.5
    )
    output: np.ndarray = np.full(
        (height, width), fill_value, dtype=np.float64
    )
    src_y_start: int = max(0, (new_height - height) // 2)
    src_x_start: int = max(0, (new_width - width) // 2)
    dst_y_start: int = max(0, (height - new_height) // 2)
    dst_x_start: int = max(0, (width - new_width) // 2)
    copy_height: int = min(height, new_height)
    copy_width: int = min(width, new_width)
    output[
        dst_y_start : dst_y_start + copy_height,
        dst_x_start : dst_x_start + copy_width,
    ] = resized_array[
        src_y_start : src_y_start + copy_height,
        src_x_start : src_x_start + copy_width,
    ]
    return output


def random_shift_and_rotate(
    images: np.ndarray,
    rng: np.random.Generator,
    max_shift: int = MAX_SHIFT_PIXELS,
    max_rotation_degrees: float = MAX_ROTATION_DEGREES,
    fill_value: float = AUGMENT_FILL_VALUE,
    elastic_alpha: float = ELASTIC_ALPHA,
    elastic_sigma: float = ELASTIC_SIGMA,
    scale_min: float = SCALE_RANGE_MIN,
    scale_max: float = SCALE_RANGE_MAX,
) -> np.ndarray:
    """Apply random digit-safe augmentations to each MNIST image batch.

    Independently (each with probability ``AUGMENT_PROBABILITY``): ±shift,
    ±rotation, elastic distortion, and mild scale/zoom. Intended for training
    batches only. Images must already be normalized to approximately
    ``[-0.5, 0.5]`` and shaped ``(N, 1, 28, 28)``.
    """
    if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError(
            f'Images to augment must have shape (N, 1, 28, 28), '
            f'got {images.shape}'
        )
    if max_shift < 0:
        raise ValueError('max_shift must be non-negative')
    if max_rotation_degrees < 0.0:
        raise ValueError('max_rotation_degrees must be non-negative')
    if elastic_alpha < 0.0:
        raise ValueError('elastic_alpha must be non-negative')
    if elastic_sigma <= 0.0:
        raise ValueError('elastic_sigma must be positive')
    if scale_min <= 0.0 or scale_max <= 0.0:
        raise ValueError('scale_min and scale_max must be positive')
    if scale_min > scale_max:
        raise ValueError('scale_min must be <= scale_max')

    batch: np.ndarray = np.squeeze(images, axis=1).copy()
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
        if elastic_alpha > 0.0 and rng.random() < AUGMENT_PROBABILITY:
            augmented = _elastic_distort_image(
                augmented, rng, elastic_alpha, elastic_sigma, fill_value
            )
        if rng.random() < AUGMENT_PROBABILITY:
            scale: float = float(rng.uniform(scale_min, scale_max))
            augmented = _scale_image(augmented, scale, fill_value)
        batch[index] = augmented
    return np.expand_dims(batch, axis=1)
