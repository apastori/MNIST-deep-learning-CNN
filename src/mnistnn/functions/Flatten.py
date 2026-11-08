"""Provide a flatten operation from spatial tensors to feature vectors."""

import numpy as np

from mnistnn.operation import Operation
from mnistnn.utils.asarray import as_array


class Flatten(Operation):
    """Flatten all non-batch dimensions into a single feature axis.

    Forward:  ``(N, ...)`` -> ``(N, F)`` where ``F`` is the product of the
    remaining axes (for CNN maps, typically ``(N, C, H, W)`` -> ``(N, C*H*W)``).
    Backward: reshape the gradient back to the original input shape.
    """

    def _output(self) -> np.ndarray:
        """Collapse every axis after the batch into one feature dimension."""
        input_: np.ndarray = self._require_input()
        if input_.ndim < 2:
            raise ValueError(
                f'Flatten expects at least 2 dimensions (batch, ...), '
                f'got shape {input_.shape}'
            )
        return as_array(input_.reshape(input_.shape[0], -1))

    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
        """Restore the spatial (or original) shape of the input gradient."""
        input_: np.ndarray = self._require_input()
        return as_array(grad.reshape(input_.shape))
