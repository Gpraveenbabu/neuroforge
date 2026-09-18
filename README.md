# NeuroForge

A neural network framework built from scratch in Python and NumPy.

## Features

- Automatic differentiation (Autograd)
- Scalar and NumPy tensor operations
- Broadcasting support
- Matrix multiplication
- Neural network layers and MLP
- ReLU and Leaky ReLU activation functions
- Gradient-based optimization
- Training utilities
- Comprehensive test suite

## Installation

Clone the repository:

```bash
git clone https://github.com/Gpraveenbabu/neuroforge.git
cd neuroforge


## Performance Benchmark

NeuroForge was benchmarked against PyTorch using a small MLP
(1 → 8 → 1) trained for 50 epochs on 100 samples.

| Framework | Average Training Time | Final Loss |
|-----------|----------------------:|-----------:|
| NeuroForge | 0.665 seconds | 0.008763 |
| PyTorch | 0.005 seconds | 0.003052 |

NeuroForge was approximately 128× slower than PyTorch.

This difference is expected because NeuroForge uses Python-level
scalar operations, while PyTorch uses optimized tensor operations.

These results are intended as an educational performance comparison,
not a production-level benchmark.
