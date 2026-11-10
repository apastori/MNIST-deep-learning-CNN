"""Provide a standalone max-pooling layer."""

import numpy as np

from mnistnn.functions.MaxPool2D import MaxPool2D
from mnistnn.layer_parent import LayerParent


class MaxPoolLayer(LayerParent):
    """Max-pooling layer wrapping the ``MaxPool2D`` operation.

    Has no learnable parameters. Callers must not use ``get_params`` /
    ``get_grads``; those methods raise to make that explicit.
    """

    def __init__(
        self, kernel_size: int = 2, stride: int | None = None
    ) -> None:
        """Initialize the underlying max-pool operation."""
        self.pool: MaxPool2D = MaxPool2D(
            kernel_size=kernel_size, stride=stride
        )

    def forward(self, input_: np.ndarray) -> np.ndarray:
        """Pool the input feature maps."""
        return self.pool.forward(input_)

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """Route gradients back through the pooling windows."""
        return self.pool.backward(output_grad)

    def get_params(self) -> tuple[np.ndarray, ...]:
        """Reject parameter access; max pooling is not trainable."""
        raise RuntimeError(
            'MaxPoolLayer has no trainable parameters; skip it during updates'
        )

    def get_grads(self) -> tuple[np.ndarray, ...]:
        """Reject gradient access; max pooling is not trainable."""
        raise RuntimeError(
            'MaxPoolLayer has no trainable parameters; skip it during updates'
        )
