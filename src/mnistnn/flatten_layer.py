"""Provide a standalone flatten layer for CNN-to-dense transitions."""

import numpy as np

from mnistnn.functions.Flatten import Flatten
from mnistnn.layer_parent import LayerParent


class FlattenLayer(LayerParent):
    """Flatten layer wrapping the ``Flatten`` operation.

    Has no learnable parameters. Callers must not use ``get_params`` /
    ``get_grads``; those methods raise to make that explicit.
    """

    def __init__(self) -> None:
        """Initialize the underlying flatten operation."""
        self.flatten: Flatten = Flatten()

    def forward(self, input_: np.ndarray) -> np.ndarray:
        """Flatten all non-batch dimensions into a feature vector."""
        return self.flatten.forward(input_)

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """Restore the pre-flatten spatial shape of the gradient."""
        return self.flatten.backward(output_grad)

    def get_params(self) -> tuple[np.ndarray, ...]:
        """Reject parameter access; flatten is not trainable."""
        raise RuntimeError(
            'FlattenLayer has no trainable parameters; skip it during updates'
        )

    def get_grads(self) -> tuple[np.ndarray, ...]:
        """Reject gradient access; flatten is not trainable."""
        raise RuntimeError(
            'FlattenLayer has no trainable parameters; skip it during updates'
        )
