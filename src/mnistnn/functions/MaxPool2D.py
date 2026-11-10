"""Provide 2D max-pooling for channels-first feature maps."""

import numpy as np

from mnistnn.operation import Operation
from mnistnn.utils.asarray import as_array


class MaxPool2D(Operation):
    """Max-pool over spatial windows on ``(N, C, H, W)`` tensors.

    For each non-overlapping (by default) window, keeps the maximum value and
    remembers which spatial index produced it so gradients can be routed back.
    """

    def __init__(
        self, kernel_size: int = 2, stride: int | None = None
    ) -> None:
        """Initialize pooling window size and stride."""
        if kernel_size <= 0:
            raise ValueError('kernel_size must be a positive integer')
        if stride is None:
            stride = kernel_size
        if stride <= 0:
            raise ValueError('stride must be a positive integer')
        super().__init__()
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        # Flat argmax index inside each window, shape (N, C, out_h, out_w)
        self.max_indices: np.ndarray | None = None

    def _output(self) -> np.ndarray:
        """Take the max over each pooling window."""
        input_: np.ndarray = self._require_input()
        if input_.ndim != 4:
            raise ValueError(
                f'MaxPool2D expects shape (N, C, H, W), got {input_.shape}'
            )

        batch_size: int
        channels: int
        height: int
        width: int
        batch_size, channels, height, width = input_.shape
        kernel: int = self.kernel_size
        stride: int = self.stride
        out_h: int = (height - kernel) // stride + 1
        out_w: int = (width - kernel) // stride + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f'Pooling window {kernel} with stride {stride} does not fit '
                f'input spatial size {(height, width)}'
            )

        output: np.ndarray = np.empty(
            (batch_size, channels, out_h, out_w), dtype=input_.dtype
        )
        self.max_indices = np.empty(
            (batch_size, channels, out_h, out_w), dtype=np.int64
        )

        for out_y in range(out_h):
            y_start: int = out_y * stride
            y_end: int = y_start + kernel
            for out_x in range(out_w):
                x_start: int = out_x * stride
                x_end: int = x_start + kernel
                window: np.ndarray = input_[
                    :, :, y_start:y_end, x_start:x_end
                ]
                flat_window: np.ndarray = window.reshape(
                    batch_size, channels, -1
                )
                max_idx: np.ndarray = np.argmax(flat_window, axis=2)
                self.max_indices[:, :, out_y, out_x] = max_idx
                output[:, :, out_y, out_x] = np.take_along_axis(
                    flat_window, max_idx[:, :, None], axis=2
                )[:, :, 0]

        return as_array(output)

    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
        """Route each output gradient back to the max location in its window."""
        if self.max_indices is None:
            raise RuntimeError(
                'MaxPool2D backward requires a previous forward pass'
            )
        input_: np.ndarray = self._require_input()
        batch_size: int
        channels: int
        height: int
        width: int
        batch_size, channels, height, width = input_.shape
        kernel: int = self.kernel_size
        stride: int = self.stride
        out_h: int = grad.shape[2]
        out_w: int = grad.shape[3]

        input_grad: np.ndarray = np.zeros_like(input_)
        batch_idx: np.ndarray = np.arange(batch_size)[:, None]
        channel_idx: np.ndarray = np.arange(channels)[None, :]

        for out_y in range(out_h):
            y_start: int = out_y * stride
            for out_x in range(out_w):
                x_start: int = out_x * stride
                max_idx: np.ndarray = self.max_indices[:, :, out_y, out_x]
                local_y: np.ndarray = max_idx // kernel
                local_x: np.ndarray = max_idx % kernel
                input_grad[
                    batch_idx,
                    channel_idx,
                    y_start + local_y,
                    x_start + local_x,
                ] += grad[:, :, out_y, out_x]

        return as_array(input_grad)
