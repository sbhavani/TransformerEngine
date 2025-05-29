# Basic MNIST Example with Transformer Engine and FP8

This example demonstrates how to use Transformer Engine's FP8 capabilities with a simple MNIST classification task. It serves as an introduction to FP8 training and shows the performance benefits of using Transformer Engine's optimized Linear layers.

## What this example demonstrates

The script trains a convolutional neural network on the MNIST dataset with three different configurations:

1. **Baseline PyTorch**: Standard PyTorch `nn.Linear` layers with default precision
2. **Transformer Engine**: TE `Linear` layers with BF16 precision for improved performance
3. **Transformer Engine + FP8**: TE `Linear` layers with FP8 precision for maximum performance

This comparison allows you to see the progressive performance improvements from using Transformer Engine optimizations.

## Model Architecture

The example uses a hybrid CNN-FC neural network.

### Configurable Layers:
Only the fully connected layers (`fc1` and `fc2`) in this example can use Transformer Engine:
- **Standard mode**: Uses `torch.nn.Linear`
- **TE mode** (`--use-te`): Uses `transformer_engine.pytorch.Linear`
- **FP8 mode** (`--use-fp8`): Uses TE Linear with FP8 autocast

## Usage

### Basic Commands

```bash
# Run with standard PyTorch Linear layers (baseline)
python main.py

# Run with Transformer Engine Linear layers (BF16 precision)
python main.py --use-te

# Run with Transformer Engine Linear layers and FP8 precision
python main.py --use-fp8
```

### Advanced Usage Examples

```bash
# Custom training configuration
python main.py --use-fp8 --epochs 20 --batch-size 128 --lr 0.5

# FP8 inference only (with calibration)
python main.py --use-fp8-infer --save-model
```

## Command Line Arguments

### Core Training Arguments
- `--batch-size`: Training batch size (default: 64)
- `--test-batch-size`: Testing batch size (default: 1000)  
- `--epochs`: Number of training epochs (default: 14)
- `--lr`: Learning rate (default: 1.0)
- `--gamma`: Learning rate decay factor (default: 0.7)

### Transformer Engine Arguments
- `--use-te`: Use Transformer Engine Linear layers instead of PyTorch
- `--use-fp8`: Enable FP8 precision for training and inference (implies `--use-te`)
- `--use-fp8-infer`: Enable FP8 for inference only with calibration

### Utility Arguments
- `--seed`: Random seed for reproducibility (default: 1)
- `--log-interval`: Batches between training status logs (default: 10)
- `--dry-run`: Run single batch for quick testing
- `--save-model`: Save trained model to `mnist_cnn.pt`

## Key Features Demonstrated

### 1. FP8 Autocast
```python
with te.fp8_autocast(enabled=use_fp8):
    output = model(data)
```
Shows how to wrap model forward passes with FP8 precision.

### 2. Layer Substitution
```python
if use_te:
    self.fc1 = te.Linear(9216, 128)
    self.fc2 = te.Linear(128, 16)
else:
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 16)
```
Demonstrates drop-in replacement of PyTorch layers with TE layers.

### 3. FP8 Calibration
```python
with te.fp8_autocast(enabled=fp8, calibrating=True):
    output = model(data)
```
Shows calibration process for FP8 inference.
