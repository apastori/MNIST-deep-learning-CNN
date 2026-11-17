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
        out_h: int = int((height - kernel) / stride) + 1
        out_w: int = int((width - kernel) / stride) + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f'Pooling window {kernel} with stride {stride} does not fit '
                f'input spatial size {(height, width)}'
            )

        # Create a new output tensor with the shape (N, C, out_h, out_w)
        output: np.ndarray = np.empty(
            (batch_size, channels, out_h, out_w), dtype=input_.dtype
        )
        # Create a new max_indices tensor with the shape (N, C, out_h, out_w)
        self.max_indices = np.empty(
            (batch_size, channels, out_h, out_w), dtype=np.int64
        )

        for out_y in range(out_h):
            y_start: int = out_y * stride
            y_end: int = y_start + kernel
            for out_x in range(out_w):
                x_start: int = out_x * stride
                x_end: int = x_start + kernel
                # Get the window from the input tensor
                # All Batch and Channels are kept,
                # only the spatial dimensions are sliced (Height and Width)
                # The output window shape is 4D (B, C, kh, kw)
                window: np.ndarray = input_[:, :, y_start:y_end, x_start:x_end]
                # Reshape the window to 3D (B, C, kw * kh)
                flat_window: np.ndarray = window.reshape(
                    batch_size, channels, -1
                )
                # Get the max index for each window
                # The output shape is 3D (B, C)
                max_idx: np.ndarray = np.argmax(flat_window, axis=2)
                # Expand the max indices to 3D (B, C, 1)
                max_idx_expanded: np.ndarray = np.expand_dims(max_idx, axis=2)
                # Get the max values for each window
                # The output shape is 3D (B, C, 1)
                max_values: np.ndarray = np.take_along_axis(
                    flat_window, max_idx_expanded, axis=2
                )
                # Store the max values for each window
                # The output shape is 4D (B, C, out_h, out_w)
                self.max_indices[:, :, out_y, out_x] = max_idx
                output[:, :, out_y, out_x] = np.squeeze(max_values, axis=2)

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

        # Create a new input gradient tensor with the shape (B, C, H, W)
        input_grad: np.ndarray = np.zeros_like(input_)

        # Create a new 1D tensor with the shape (B,)
        batch_indices: np.ndarray = np.arange(batch_size)
        # Create a new batch index tensor with the shape (B, 1)
        batch_idx: np.ndarray = np.expand_dims(batch_indices, axis=1)
        # Create a new 1D tensor with the shape (C,)
        channel_indices: np.ndarray = np.arange(channels)
        # Create a new channel index tensor with the shape (1, C)
        channel_idx: np.ndarray = np.expand_dims(channel_indices, axis=0)

        for out_y in range(out_h):
            y_start: int = out_y * stride
            for out_x in range(out_w):
                x_start: int = out_x * stride
                max_idx: np.ndarray = self.max_indices[:, :, out_y, out_x]
                window_row: np.ndarray = max_idx // kernel
                window_col: np.ndarray = max_idx % kernel
                input_row: np.ndarray = y_start + window_row
                input_col: np.ndarray = x_start + window_col
                input_grad[
                    batch_idx,
                    channel_idx,
                    input_row,
                    input_col,
                ] += grad[:, :, out_y, out_x]

        return as_array(input_grad)
