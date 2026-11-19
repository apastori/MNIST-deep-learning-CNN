"""Provide a neural network for MNIST digit classification.

Architecture (input ``(N, 1, 28, 28)``):
- Conv 1 -> 32 + MaxPool
- Conv 32 -> 64 + MaxPool
- Flatten
- Dense 3136 -> 256 with ReLU and dropout 0.2
- Dense 256 -> 10 with Softmax
"""
