# torch.compile benchmarks

This directory contains benchmarks and tests for evaluating the performance impact of selectively applying `torch.compile` to specific operations while keeping TE's FP8 linear layers in eager mode (surgical compilation approach).

## Overview

The surgical torch.compile approach targets performance-critical tensor operations for compilation while preserving TE's optimized FP8 linear layers in their original eager implementation. This strategy aims to achieve speedups without disrupting the specialized FP8 optimizations.

## Test Files

### `test_rope_speedup.py`
**Purpose**: Isolated benchmark for Rotary Position Embedding (RoPE) operations
**Focus**: Compares eager vs compiled RoPE implementations
**Key Findings**: 2.79x speedup with torch.compile on RoPE operations
**Usage**:
```bash
python3 test_rope_speedup.py
```

### `test_surgical_compile.py`
**Purpose**: Tests surgical compilation on individual TE operations
**Focus**: Validates that TE layers work correctly with selective compilation
**Key Findings**: Confirms compatibility and identifies compilation boundaries
**Usage**:
```bash
python3 test_surgical_compile.py
```

### `test_te_surgical.py`
**Purpose**: Integration test for TE layers with surgical torch.compile
**Focus**: End-to-end validation of mixed eager/compiled execution
**Key Findings**: Demonstrates stable integration patterns
**Usage**:
```bash
python3 test_te_surgical.py
```

### `test_end_to_end_surgical.py`
**Purpose**: Comprehensive end-to-end benchmark with realistic transformer model
**Focus**: Measures real-world performance impact on multi-layer transformer
**Key Findings**: 1.11x speedup (10.7% improvement) on full model inference
**Configuration**:
- Batch size: 8
- Sequence length: 2048
- Hidden size: 2048
- Number of heads: 16
- Number of layers: 4
- FFN hidden size: 4096

**Usage**:
```bash
python3 test_end_to_end_surgical.py
```

## Key Results Summary

### H100 PCIe Performance Results
**Hardware**: NVIDIA H100 PCIe
**PyTorch**: 2.8.0a0+34c6371d24.nv25.08

| Test | Operation | Eager Time | Compiled Time | Speedup | Notes |
|------|-----------|------------|---------------|---------|-------|
| RoPE Isolated | RoPE only | ~2.79x baseline | 1x baseline | 2.79x | Significant isolated speedup |
| End-to-End | Full Model | 31.284 ms | 28.261 ms | 1.11x | 10.7% improvement |

**End-to-End Model Configuration**:
- Batch size: 8, Sequence length: 2048, Hidden size: 2048
- 4 transformer layers, 16 attention heads
- Average over 50 trials with 10 warmup iterations
- **Baseline (All Eager)**: 31.284 ms avg, 31.038 ms min
- **Surgical (Compiled Tensor Ops)**: 28.261 ms avg, 27.910 ms min

## Hardware Requirements

- NVIDIA GPU with CUDA support
- TransformerEngine compatible GPU (tested on H100)
- PyTorch with CUDA support
- TransformerEngine library properly installed

## Implementation Strategy

### What Gets Compiled (Surgical Targets)
- **RoPE Operations**: Complex trigonometric computations benefit significantly
- **Tensor Reshaping**: QKV projection reshaping and attention output combination
- **Standard PyTorch Operations**: Attention computations, activations

### What Stays Eager (Preserved Operations)
- **TransformerEngine FP8 Linear Layers**: Maintain specialized optimizations
- **FP8 Autocast Contexts**: Preserve precision management
- **Custom CUDA Kernels**: Keep existing kernel optimizations

### Code Pattern
```python
# Compile specific tensor operations
rope_compiled = torch.compile(apply_rope, mode="max-autotune")
reshape_compiled = torch.compile(qkv_reshape_split, mode="default")

class MiniTransformerLayer(nn.Module):
    def __init__(self, use_surgical: bool = False):
        # TE layers stay eager
        self.qkv_proj = te.Linear(hidden_size, 3 * hidden_size)

        # Choose compiled vs eager for tensor ops
        if use_surgical:
            self.rope_fn = rope_compiled
            self.reshape_fn = reshape_compiled
        else:
            self.rope_fn = apply_rope
            self.reshape_fn = qkv_reshape_split
```

## Performance Analysis

### Numerical Accuracy
- **Status**: Currently showing small numerical differences (max diff: 0.578125)
- **Root Cause**: Compilation optimizations may change operation order
- **Recommendation**: Further investigation needed for production use

### Memory Usage
- **FP8 Linear Layers**: Preserved memory efficiency
- **Compiled Operations**: Standard torch.compile memory overhead
- **Overall**: Minimal additional memory impact

### Compilation Overhead
- **First Run**: Initial compilation cost (one-time)
- **Subsequent Runs**: Full performance benefits
- **Training**: Compilation cost amortized over many iterations

## Usage Recommendations

### For Development
1. Start with isolated tests (`test_rope_speedup.py`)
2. Validate integration (`test_te_surgical.py`)
3. Measure end-to-end impact (`test_end_to_end_surgical.py`)

## Future Work

1. **Numerical Stability**: Investigate and resolve output differences
2. **Additional Operations**: Identify other compilation candidates
3. **Memory Optimization**: Profile memory usage patterns
4. **Multi-GPU**: Extend benchmarks to distributed scenarios
5. **Training Benchmarks**: Add backward pass performance tests
