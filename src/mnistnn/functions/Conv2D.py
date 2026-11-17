"""Provide a 2D convolution operation for channels-first feature maps."""

import numpy as np

from mnistnn.operation import Operation
from mnistnn.utils.asarray import as_array


class Conv2D(Operation):
    """Cross-correlation of ``(N, C, H, W)`` inputs with learnable kernels.

    Kernel layout is ``(out_channels, in_channels, kH, kW)``. Bias is one
    value per output channel, broadcast over spatial locations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        """Initialize convolution kernels with He scaling and zero bias."""
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError('in_channels and out_channels must be positive')
        if kernel_size <= 0:
            raise ValueError('kernel_size must be a positive integer')
        if stride <= 0:
            raise ValueError('stride must be a positive integer')
        if padding < 0:
            raise ValueError('padding must be non-negative')

        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        self.padding: int = padding
        # Calculate the fan-in for the weights, shape (out_channels, in_channels, kernel_size, kernel_size)
        fan_in: int = in_channels * kernel_size * kernel_size
        # Define the weights matrix with shape 4D (out_channels, in_channels, kernel_size, kernel_size)
        # Initialize the weights matrix with random values from a normal distribution with mean 0 and standard deviation sqrt(2.0 / fan_in)
        # He initialization
        self.weights: np.ndarray = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * np.sqrt(2.0 / fan_in)
        # Create a bias vector with shape (out_channels) 1-D array of zeros
        self.bias: np.ndarray = np.zeros(out_channels, dtype=np.float64)
        self.grad_weights: np.ndarray | None = None
        self.grad_bias: np.ndarray | None = None
        # Cached im2col matrix from the forward pass, shape (N, C*kH*kW, out_h*out_w)
        self._cols: np.ndarray | None = None
        self._out_h: int | None = None
        self._out_w: int | None = None

    def _pad(self, input_: np.ndarray) -> np.ndarray:
        """Pad spatial axes with zeros when ``padding > 0``."""
        if self.padding == 0:
            return input_
        return np.pad(
            input_,
            (
                # 1st axis: Batch size
                (0, 0),
                # 2nd axis: Channels
                (0, 0),
                # 3rd axis: Height
                (self.padding, self.padding),
                # 4th axis: Width
                (self.padding, self.padding),
            ),
            mode='constant',
            constant_values=0.0,
        )

    def _im2col(self, input_: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Extract sliding patches into columns for a batched matmul."""
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
                f'Convolution kernel {kernel} with stride {stride} and '
                f'padding {self.padding} does not fit input spatial size '
                f'{(height - 2 * self.padding, width - 2 * self.padding)}'
            )

        cols: np.ndarray = np.empty(
            (batch_size, channels, kernel, kernel, out_h, out_w),
            dtype=input_.dtype,
        )
        # Loop through the kernel height and width
        for ky in range(kernel):
            y_end: int = ky + stride * out_h
            for kx in range(kernel):
                x_end: int = kx + stride * out_w
                cols[:, :, ky, kx, :, :] = input_[
                    :, :, ky:y_end:stride, kx:x_end:stride
                ]
        cols = cols.reshape(
            batch_size, channels * kernel * kernel, out_h * out_w
        )
        return cols, out_h, out_w

    def _col2im(
        self,
        cols: np.ndarray,
        padded_shape: tuple[int, int, int, int],
        out_h: int,
        out_w: int,
    ) -> np.ndarray:
        """Scatter column gradients back into a padded feature map."""
        batch_size: int
        channels: int
        height: int
        width: int
        batch_size, channels, height, width = padded_shape
        kernel: int = self.kernel_size
        stride: int = self.stride
        cols_reshaped: np.ndarray = cols.reshape(
            batch_size, channels, kernel, kernel, out_h, out_w
        )
        input_grad: np.ndarray = np.zeros(padded_shape, dtype=cols.dtype)
        for ky in range(kernel):
            y_end: int = ky + stride * out_h
            for kx in range(kernel):
                x_end: int = kx + stride * out_w
                input_grad[
                    :, :, ky:y_end:stride, kx:x_end:stride
                ] += cols_reshaped[:, :, ky, kx, :, :]
        return input_grad

    def _output(self) -> np.ndarray:
        """Convolve the input with kernels and add per-channel bias."""
        input_: np.ndarray = self._require_input()
        if input_.ndim != 4:
            raise ValueError(
                f'Conv2D expects shape (N, C, H, W), got {input_.shape}'
            )
        if input_.shape[1] != self.in_channels:
            raise ValueError(
                f'Conv2D expected {self.in_channels} input channels, '
                f'got {input_.shape[1]}'
            )

        padded: np.ndarray = self._pad(input_)
        cols: np.ndarray
        out_h: int
        out_w: int
        cols, out_h, out_w = self._im2col(padded)
        self._cols = cols
        self._out_h = out_h
        self._out_w = out_w

        # Reshape the weights matrix to 2D (out_channels, in_channels * kernel_size * kernel_size)
        weights_mat: np.ndarray = self.weights.reshape(self.out_channels, -1)
        # (out_c, K) @ (N, K, HW) -> (N, out_c, HW) using einsum
        # The output shape is (N, out_channels, out_h * out_w)
        output: np.ndarray = np.einsum('ok,nkp->nop', weights_mat, cols)
        # Lets add to Axis 0 and 2 the bias so that it is (1, bias, 1)
        bias: np.ndarray = self.bias.reshape(1, self.out_channels, 1)
        output += bias
        # Reshape the output to (N, out_channels, out_h, out_w)
        return as_array(
            output.reshape(input_.shape[0], self.out_channels, out_h, out_w)
        )

    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
        """Compute input, weight, and bias gradients for the convolution."""
        if self._cols is None or self._out_h is None or self._out_w is None:
            raise RuntimeError(
                'Conv2D backward requires a previous forward pass'
            )

        input_: np.ndarray = self._require_input()
        batch_size: int = input_.shape[0]
        out_h: int = self._out_h
        out_w: int = self._out_w
        # Reshape the gradient to (batch_size, out_channels, out_h * out_w)
        grad_flat: np.ndarray = grad.reshape(
            batch_size, self.out_channels, out_h * out_w
        )

        # Lets sum the gradient over the Batch size and the spatial dimensions (H and W) (Axis 0, 2, and 3)
        # to get the bias gradient
        # The output shape is 1D (out_channels)
        self.grad_bias = as_array(np.sum(grad, axis=(0, 2, 3)))

        # dW: (out_c, N, HW) x (N, K, HW) -> (out_c, K)
        weights_grad_mat: np.ndarray = np.einsum(
            'nop,nkp->ok', grad_flat, self._cols
        )

        # Reshape the weights gradient to the weights shape
        # from 2D (out_channels, in_channels * kernel_size * kernel_size)
        # to 4D (out_channels, in_channels, kernel_size, kernel_size)
        self.grad_weights = as_array(
            weights_grad_mat.reshape(self.weights.shape)
        )

        # Reshape the Weights 4D Tensor (Batch, out_channels, kw, kh)
        # to 2D (out_channels, in_channels * kw * kh)
        weights_mat: np.ndarray = self.weights.reshape(self.out_channels, -1)
        # dCols: (out_c, K)^T @ (N, out_c, HW) -> (N, K, HW)
        # Results Cols Gradient (Batch, in_channels * kw * kh, out_h * out_w)
        cols_grad: np.ndarray = np.einsum(
            'ok,nop->nkp', weights_mat, grad_flat
        )

        padded_height: int = input_.shape[2] + 2 * self.padding
        padded_width: int = input_.shape[3] + 2 * self.padding
        # Scatter column gradients back into a padded feature map
        padded_grad: np.ndarray = self._col2im(
            cols_grad,
            (batch_size, self.in_channels, padded_height, padded_width),
            out_h,
            out_w,
        )
        if self.padding == 0:
            return as_array(padded_grad)
        # Remove the padding from the padded gradient
        return as_array(
            padded_grad[
                :,
                :,
                self.padding : padded_height - self.padding,
                self.padding : padded_width - self.padding,
            ]
        )
