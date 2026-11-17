"""Provide the MNIST neural-network implementation."""

from pathlib import Path

import numpy as np

from mnistnn.conv_layer import ConvLayer
from mnistnn.dense_layer import DenseLayer
from mnistnn.flatten_layer import FlattenLayer
from mnistnn.functions.CrossEntropyLoss import CrossEntropyLoss
from mnistnn.functions.ReLU import ReLU
from mnistnn.functions.SoftmaxStable import SoftmaxStable
from mnistnn.layer_parent import LayerParent
from mnistnn.loss import Loss
from mnistnn.max_pool_layer import MaxPoolLayer

# After two 2x2 pools: 28 -> 14 -> 7; with 64 channels: 64 * 7 * 7
_FLAT_FEATURES: int = 64 * 7 * 7


class MNISTNeuralNetwork:
    """Convolutional neural network for MNIST digit classification.

    Architecture (channels-first input ``(N, 1, 28, 28)``):
    - ConvLayer: 1 -> 32 channels, ReLU (no built-in pool)
    - MaxPoolLayer: 2x2
    - ConvLayer: 32 -> 64 channels, ReLU (no built-in pool)
    - MaxPoolLayer: 2x2
    - FlattenLayer
    - DenseLayer: 3136 -> 128 with ReLU and dropout 0.2
    - DenseLayer: 128 -> 10 with Softmax
    """

    def __init__(self) -> None:
        """Initialize the network layers and loss function."""
        self.layers: list[LayerParent] = [
            ConvLayer(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
                activation=ReLU(),
                pool_size=None,
            ),
            MaxPoolLayer(kernel_size=2),
            ConvLayer(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                activation=ReLU(),
                pool_size=None,
            ),
            MaxPoolLayer(kernel_size=2),
            FlattenLayer(),
            # with Flatten the number of features is the number of channels * the number of pixels
            # in this case 64 * 7 * 7 = 3136
            DenseLayer(
                input_size=_FLAT_FEATURES,
                output_size=128,
                activation=ReLU(),
                dropout_rate=0.2,
            ),
            DenseLayer(
                input_size=128, output_size=10, activation=SoftmaxStable()
            ),
        ]
        self.loss_fn: Loss = CrossEntropyLoss()
        # SGD momentum buffers for trainable layers only
        self._weight_velocities: list[np.ndarray] | None = None
        self._bias_velocities: list[np.ndarray] | None = None

    def _trainable_layers(self) -> list[LayerParent]:
        """Return layers that expose weights and biases for SGD."""
        return [
            layer
            for layer in self.layers
            if isinstance(layer, (ConvLayer, DenseLayer))
        ]

    def _set_training(self, training: bool) -> None:
        """Enable or disable dropout on every dense layer."""
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.training = training

    def forward(self, input_: np.ndarray, training: bool = True) -> np.ndarray:
        """Perform a forward pass through the network."""
        self._set_training(training)
        if input_.ndim != 4 or input_.shape[1:] != (1, 28, 28):
            raise ValueError(
                f'Expected images with shape (batch_size, 1, 28, 28), '
                f'got {input_.shape}'
            )

        x: np.ndarray = input_
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def _backward(self, output_grad: np.ndarray) -> np.ndarray:
        """Perform a backward pass through the network."""
        input_grad: np.ndarray = output_grad

        for layer in reversed(self.layers):
            input_grad = layer.backward(input_grad)

        return input_grad

    def _compute_loss(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> float:
        """Compute the cross-entropy loss."""
        return self.loss_fn.forward(predictions, targets)

    def _update_parameters(
        self,
        learning_rate: float,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ) -> None:
        """Update weights and biases using SGD with momentum and L2 decay."""
        trainable_layers: list[LayerParent] = self._trainable_layers()
        if self._weight_velocities is None or self._bias_velocities is None:
            self._weight_velocities = []
            self._bias_velocities = []
            for layer in trainable_layers:
                init_weights: np.ndarray
                init_bias: np.ndarray
                init_weights, init_bias = layer.get_params()
                self._weight_velocities.append(np.zeros_like(init_weights))
                self._bias_velocities.append(np.zeros_like(init_bias))

        for index, layer in enumerate(trainable_layers):
            weights: np.ndarray
            bias: np.ndarray
            weights, bias = layer.get_params()
            grad_weights: np.ndarray
            grad_bias: np.ndarray
            grad_weights, grad_bias = layer.get_grads()
            weight_velocity: np.ndarray
            bias_velocity: np.ndarray
            weight_velocity = (
                momentum * self._weight_velocities[index] + grad_weights
            )
            bias_velocity = momentum * self._bias_velocities[index] + grad_bias
            self._weight_velocities[index] = weight_velocity
            self._bias_velocities[index] = bias_velocity
            weights -= learning_rate * (
                weight_velocity + weight_decay * weights
            )
            bias -= learning_rate * bias_velocity

    def train_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ) -> float:
        """Perform a single training step."""
        predictions: np.ndarray = self.forward(x, training=True)
        loss: float = self._compute_loss(predictions, y)
        grad: np.ndarray = self.loss_fn.backward()
        self._backward(grad)
        self._update_parameters(learning_rate, momentum, weight_decay)
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        predictions: np.ndarray = self.forward(x, training=False)
        return np.asarray(np.argmax(predictions, axis=1))

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate accuracy on a dataset."""
        predictions: np.ndarray = self.predict(x)

        labels: np.ndarray = y
        if len(y.shape) == 2:
            labels = np.asarray(np.argmax(y, axis=1))

        accuracy: float = float(np.mean(predictions == labels))
        return accuracy

    def save_weights(self, path: str | Path) -> None:
        """Save trainable layer weights and biases to an ``.npz`` file."""
        arrays_model_weights: dict[str, np.ndarray] = {}
        for index, layer in enumerate(self._trainable_layers()):
            weights: np.ndarray
            bias: np.ndarray
            weights, bias = layer.get_params()
            arrays_model_weights[f'weights_{index}'] = weights
            arrays_model_weights[f'bias_{index}'] = bias
        np.savez(path, **arrays_model_weights)

    def load_weights(self, path: str | Path) -> None:
        """Load trainable layer weights and biases from an ``.npz`` file."""
        with np.load(path) as arrays_model_weights:
            for index, layer in enumerate(self._trainable_layers()):
                weights: np.ndarray
                bias: np.ndarray
                weights, bias = layer.get_params()
                loaded_weights: np.ndarray = arrays_model_weights[
                    f'weights_{index}'
                ]
                loaded_bias: np.ndarray = arrays_model_weights[f'bias_{index}']
                if loaded_weights.shape != weights.shape:
                    raise ValueError(
                        f'Weight shape mismatch for layer {index}: '
                        f'expected {weights.shape}, got {loaded_weights.shape}'
                    )
                if loaded_bias.shape != bias.shape:
                    raise ValueError(
                        f'Bias shape mismatch for layer {index}: '
                        f'expected {bias.shape}, got {loaded_bias.shape}'
                    )
                weights[:] = loaded_weights
                bias[:] = loaded_bias
        self._weight_velocities = None
        self._bias_velocities = None
