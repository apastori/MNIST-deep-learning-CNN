# Basic Deep Learning MNIST

From-scratch NumPy **CNN** for MNIST digit classification, plus Black, Ruff, and MyPy for code quality.

## Architecture

Channels-first input shape: `(N, 1, 28, 28)`.

| Stage | Layer | Details |
|---|---|---|
| 1 | Conv2D + ReLU | 1 → 32 filters, kernel 3×3, padding 1 (keeps 28×28) |
| 2 | MaxPool2D | 2×2, stride 2 → 14×14 |
| 3 | Conv2D + ReLU | 32 → 64 filters, kernel 3×3, padding 1 (keeps 14×14) |
| 4 | MaxPool2D | 2×2, stride 2 → 7×7 |
| 5 | Flatten | 64 × 7 × 7 = 3136 features |
| 6 | Dense + ReLU + Dropout | 3136 → 256, dropout 0.2 |
| 7 | Dense + Softmax | 256 → 10 class probabilities |

Loss is cross-entropy. Optimizer is SGD with momentum and L2 weight decay on weights only. During training the best test accuracy is checkpointed to a temporary `models/mnist_cnn_best.npz`; at the end that checkpoint is restored into `models/mnist_cnn.npz` and the temporary file is deleted, so the only saved artifact is the best epoch (not necessarily the last).

## Results

Training loss is mean batch cross-entropy over the epoch. Numbers below are from the last completed run logged in `log/mnist-training.log` (32/64 convs, dense 128, rich aug: **99.34%** best). Next experiment: same 32/64 convs, denser head **256**, mild shift/rotate, **16** epochs — update this table after that run finishes.

| Metric | Value |
|---|---|
| Test accuracy (saved model, prior richer-aug run) | **99.34%** |
| Test accuracy (best epoch, original 12-epoch mild-aug run) | **99.37%** (epoch 11) |
| Epochs (default) | 16 |
| Wall time (CPU, prior 24-epoch richer-aug run) | ~5.3 hours (18971.34s) |

**How to read these numbers:** report the **saved** test accuracy in `models/mnist_cnn.npz` (best epoch after the upgraded trainer).

## Tooling

This boilerplate uses the following tools to maintain code quality:

### Black

[Black](https://github.com/psf/black) is the uncompromising Python code formatter. It enforces a strict and consistent code style, which helps to eliminate debates over formatting and makes code review faster. By using Black, you ensure that the code looks the same regardless of who wrote it.

### Ruff

[Ruff](https://github.com/astral-sh/ruff) is an extremely fast Python linter, written in Rust. It can check for a wide range of errors and style issues, from simple syntax mistakes to complex bugs. It also includes a code formatter. In this boilerplate, it is used for linting and fixing common issues automatically.

### MyPy

[MyPy](http://mypy-lang.org/) is a static type checker for Python. It helps you write cleaner, more robust code by adding type hints to your functions and variables. MyPy can catch type-related errors before you even run your code, which can save you a lot of time debugging.

## Automation

This project uses `make` and `invoke` to automate common development tasks.

### Makefile

A `Makefile` is included to provide a simple and familiar interface for common commands. You can see the available commands by running `make help`.

### Invoke & tasks.py

[Invoke](https://www.pyinvoke.org/) is a Python task execution tool. The `tasks.py` file defines a set of tasks that can be run from the command line. This allows for more complex logic than a `Makefile` and is written in Python. You can see the available tasks by running `invoke --list`.

## Train the neural network

Install dependencies first (`invoke install` or `make install`). Run every command from the project root. The package lives under `src/`, so Python needs `PYTHONPATH=src` (the Invoke and Make targets set this for you).

All of the commands below run `python -m mnistnn`, which calls `train_mnist()` with these defaults:

- 16 epochs
- batch size 64
- learning rate 0.01 (decayed by 0.95 after each epoch, floor `1e-4`)
- SGD momentum 0.9
- L2 weight decay `1e-4` (weights only)
- dropout `0.2` on the dense hidden layer (`3136 → 256`)
- digit-safe train augmentation (each ~50% chance): ±2 px shift, ±8° rotation (elastic/scale helpers exist but are disabled by default)
- best-epoch checkpointing via temporary `models/mnist_cnn_best.npz`, then copy into `models/mnist_cnn.npz` and delete the temp file
- pixel values scaled to `[-0.5, 0.5]`, input shape `(N, 1, 28, 28)`
- sample plots enabled (`visualize=True`)

Training progress is written to `log/mnist-training.log`.

After a successful training run, the best-epoch weights are saved to `models/mnist_cnn.npz` (Conv and Dense layers only) and can be committed with the repo. On the next `make train` / `invoke train`, if that file exists the network loads it, reports test accuracy, and skips training. Delete `models/mnist_cnn.npz`, or call `train_mnist(force_retrain=True)`, to force a full retrain.

### Invoke

Works the same in bash, PowerShell, and cmd:

```text
invoke train
```

### Make

Use this in bash, Git Bash, or WSL. The Makefile uses Unix `PYTHONPATH=src python -m mnistnn`, so it is not for stock Windows cmd or PowerShell unless `make` is installed.

```bash
make train
```

### Python interpreter

Set `PYTHONPATH` in the current shell, then run the module.

bash, Git Bash, or WSL:

```bash
export PYTHONPATH=src
python -m mnistnn
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m mnistnn
```

cmd:

```bat
set PYTHONPATH=src
python -m mnistnn
```

To train without plot windows, set `PYTHONPATH` as above, then:

```text
python -c "from mnistnn.train_mnist import train_mnist; train_mnist(visualize=False)"
```

To force a full retrain even when `models/mnist_cnn.npz` already exists:

```text
python -c "from mnistnn.train_mnist import train_mnist; train_mnist(visualize=False, force_retrain=True)"
```

## Available Commands

You can use either `make` or `invoke` to run the following commands:

| Command | Makefile | Invoke | Description |
|---|---|---|---|
| **Train Network** | `make train` | `invoke train` | Trains the MNIST network (`python -m mnistnn`). |
| **Format Code** | `make format` | `invoke format` | Formats the code using Black. |
| **Check Formatting** | `make format-check` | `invoke format-check` | Checks if the code is formatted correctly without making changes. |
| **Lint Code** | `make lint` | `invoke lint` | Lints the code with Ruff to find potential issues. |
| **Lint and Fix** | `make lint-fix` | `invoke lint-fix` | Lints the code with Ruff and automatically fixes any issues it can. |
| **Type Check** | `make typecheck` | `invoke typecheck` | Runs MyPy to check for type errors. |
| **Quality Assurance** | `make qa` | `invoke qa` | Runs all the quality checks: format, lint-fix, and typecheck. |
| **Install Dependencies**| `make install` | `invoke install` | Installs the required Python packages from `requirements.txt`. |
| **Clean Project** | `make clean` | `invoke clean` | Removes cache directories and temporary files. |
| **Help** | `make help` | `invoke help` | Shows the list of available commands. |
