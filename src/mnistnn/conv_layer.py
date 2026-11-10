"""Provide a convolutional layer with optional pooling."""

import numpy as np

from mnistnn.functions.Conv2D import Conv2D
from mnistnn.functions.MaxPool2D import MaxPool2D
from mnistnn.layer_parent import LayerParent
from mnistnn.operation import Operation


class ConvLayer(LayerParent):
    """Conv layer: Conv2D, activation, and optional max pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: Operation | None = None,
        pool_size: int | None = 2,
    ) -> None:
        """Initialize convolution, activation, and optional pooling."""
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                'in_channels and out_channels must be positive integers.'
            )
        if activation is None:
            raise ValueError('Activation function must be provided.')
        if pool_size is not None and pool_size <= 0:
            raise ValueError('pool_size must be a positive integer or None.')

        self.conv: Conv2D = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.activation: Operation = activation
        self.pool: MaxPool2D | None = (
            MaxPool2D(kernel_size=pool_size) if pool_size is not None else None
        )
        self.conv_output: np.ndarray | None = None

    def forward(self, input_: np.ndarray) -> np.ndarray:
        """Compute the layer output for the given input."""
        self.conv_output = self.conv.forward(input_)
        output: np.ndarray = self.activation.forward(self.conv_output)
        if self.pool is not None:
            output = self.pool.forward(output)
        return output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """Propagate gradients backward through the layer."""
        output_grad_final: np.ndarray = output_grad
        if self.pool is not None:
            output_grad_final = self.pool.backward(output_grad_final)
        output_grad_final = self.activation.backward(output_grad_final)
        return self.conv.backward(output_grad_final)

    def get_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the convolution kernels and bias."""
        return self.conv.weights, self.conv.bias

    def get_grads(self) -> tuple[np.ndarray, np.ndarray]:
        """Return gradients for the convolution kernels and bias."""
        if self.conv.grad_weights is None or self.conv.grad_bias is None:
            raise RuntimeError(
                'ConvLayer gradients are not available before backward.'
            )
        return self.conv.grad_weights, self.conv.grad_bias
