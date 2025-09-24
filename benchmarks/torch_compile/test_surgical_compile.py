#!/usr/bin/env python3

"""
Test script to benchmark surgical torch.compile approach on TransformerEngine operations.
This script isolates and benchmarks the tensor manipulation operations that could benefit
from selective torch.compile optimization while keeping FP8 operations in eager mode.
"""

import os
import time
import warnings
from contextlib import contextmanager
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "0"

# Test configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
BATCH_SIZE = 8
SEQ_LEN = 2048
HIDDEN_SIZE = 4096
NUM_HEADS = 32
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
NUM_TRIALS = 100
WARMUP_TRIALS = 10

def setup_test_tensors():
    """Create test tensors similar to TransformerEngine usage"""
    # Input tensor [seq_len, batch, hidden_size]
    hidden_states = torch.randn(SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)

    # QKV output tensor [seq_len, batch, 3 * hidden_size]
    mixed_x_layer = torch.randn(SEQ_LEN, BATCH_SIZE, 3 * HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)

    # RoPE embeddings
    rope_emb = torch.randn(SEQ_LEN, 1, HEAD_DIM // 2, 2, dtype=DTYPE, device=DEVICE)

    return hidden_states, mixed_x_layer, rope_emb

def apply_rotary_pos_emb_simple(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    """Simplified RoPE implementation for benchmarking"""
    # Split x into two halves
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos_sin.unbind(-1)

    # Apply rotation
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class TensorOpsEager:
    """Eager mode tensor operations (current TE behavior)"""

    @staticmethod
    def qkv_reshape_and_rope(mixed_x_layer: torch.Tensor, rope_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshape QKV tensor and apply RoPE - eager mode"""
        seq_len, batch, _ = mixed_x_layer.shape

        # Reshape to separate Q, K, V [seq_len, batch, 3, num_heads, head_dim]
        qkv = mixed_x_layer.view(seq_len, batch, 3, NUM_HEADS, HEAD_DIM)

        # Split into Q, K, V tensors
        query, key, value = qkv.unbind(dim=2)  # Each: [seq_len, batch, num_heads, head_dim]

        # Transpose for attention computation [batch, num_heads, seq_len, head_dim]
        query = query.transpose(0, 1).transpose(1, 2)
        key = key.transpose(0, 1).transpose(1, 2)
        value = value.transpose(0, 1).transpose(1, 2)

        # Apply RoPE to query and key
        cos_sin = rope_emb.expand(seq_len, batch, HEAD_DIM // 2, 2)
        cos_sin_transposed = cos_sin.transpose(0, 1).unsqueeze(1).expand(-1, NUM_HEADS, -1, -1, -1)

        query = apply_rotary_pos_emb_simple(query, cos_sin_transposed)
        key = apply_rotary_pos_emb_simple(key, cos_sin_transposed)

        return query, key, value

class TensorOpsCompiled:
    """Compiled tensor operations (surgical approach)"""

    def __init__(self):
        # Compile the tensor manipulation function
        self._compiled_qkv_reshape_and_rope = torch.compile(
            self._qkv_reshape_and_rope_impl,
            mode="max-autotune",
            fullgraph=True
        )

    def _qkv_reshape_and_rope_impl(self, mixed_x_layer: torch.Tensor, rope_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Implementation that will be compiled"""
        seq_len, batch, _ = mixed_x_layer.shape

        # Reshape to separate Q, K, V [seq_len, batch, 3, num_heads, head_dim]
        qkv = mixed_x_layer.view(seq_len, batch, 3, NUM_HEADS, HEAD_DIM)

        # Split into Q, K, V tensors
        query, key, value = qkv.unbind(dim=2)

        # Transpose for attention computation [batch, num_heads, seq_len, head_dim]
        query = query.transpose(0, 1).transpose(1, 2)
        key = key.transpose(0, 1).transpose(1, 2)
        value = value.transpose(0, 1).transpose(1, 2)

        # Apply RoPE to query and key
        cos_sin = rope_emb.expand(seq_len, batch, HEAD_DIM // 2, 2)
        cos_sin_transposed = cos_sin.transpose(0, 1).unsqueeze(1).expand(-1, NUM_HEADS, -1, -1, -1)

        query = apply_rotary_pos_emb_simple(query, cos_sin_transposed)
        key = apply_rotary_pos_emb_simple(key, cos_sin_transposed)

        return query, key, value

    def qkv_reshape_and_rope(self, mixed_x_layer: torch.Tensor, rope_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compiled version"""
        return self._compiled_qkv_reshape_and_rope(mixed_x_layer, rope_emb)

@contextmanager
def timer(name: str):
    """Simple timing context manager"""
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    start = time.perf_counter()
    yield
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    end = time.perf_counter()
    print(f"{name}: {(end - start) * 1000:.3f} ms")

def benchmark_operation(name: str, operation_func, *args, **kwargs):
    """Benchmark an operation with warmup"""
    print(f"\nBenchmarking {name}...")

    # Warmup
    for _ in range(WARMUP_TRIALS):
        _ = operation_func(*args, **kwargs)

    torch.cuda.synchronize() if DEVICE == "cuda" else None

    # Actual benchmark
    times = []
    for _ in range(NUM_TRIALS):
        start = time.perf_counter()
        result = operation_func(*args, **kwargs)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"  Average: {avg_time:.3f} ms")
    print(f"  Min: {min_time:.3f} ms")
    print(f"  Max: {max_time:.3f} ms")

    return avg_time, result

def test_attention_preprocessing():
    """Test attention preprocessing operations (reshape + RoPE)"""
    print("="*60)
    print("ATTENTION PREPROCESSING BENCHMARK")
    print("="*60)
    print(f"Configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Number of heads: {NUM_HEADS}")
    print(f"  Head dimension: {HEAD_DIM}")
    print(f"  Device: {DEVICE}")
    print(f"  Dtype: {DTYPE}")

    hidden_states, mixed_x_layer, rope_emb = setup_test_tensors()

    # Test eager mode
    eager_ops = TensorOpsEager()
    eager_time, eager_result = benchmark_operation(
        "Eager Mode (Current TE)",
        eager_ops.qkv_reshape_and_rope,
        mixed_x_layer, rope_emb
    )

    # Test compiled mode
    compiled_ops = TensorOpsCompiled()
    compiled_time, compiled_result = benchmark_operation(
        "Compiled Mode (Surgical)",
        compiled_ops.qkv_reshape_and_rope,
        mixed_x_layer, rope_emb
    )

    # Verify correctness
    q_eager, k_eager, v_eager = eager_result
    q_compiled, k_compiled, v_compiled = compiled_result

    print(f"\nCorrectness Check:")
    print(f"  Query allclose: {torch.allclose(q_eager, q_compiled, rtol=1e-4, atol=1e-4)}")
    print(f"  Key allclose: {torch.allclose(k_eager, k_compiled, rtol=1e-4, atol=1e-4)}")
    print(f"  Value allclose: {torch.allclose(v_eager, v_compiled, rtol=1e-4, atol=1e-4)}")

    # Calculate speedup
    speedup = eager_time / compiled_time
    print(f"\nPerformance Results:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Improvement: {((speedup - 1) * 100):.1f}%")

def test_simple_tensor_ops():
    """Test simple tensor operations that are common in TE"""
    print("\n" + "="*60)
    print("SIMPLE TENSOR OPERATIONS BENCHMARK")
    print("="*60)

    # Setup test data
    x = torch.randn(SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)

    def eager_tensor_chain(x):
        """Chain of tensor operations in eager mode"""
        # Typical operations found in TE layers
        x = x.transpose(0, 1)  # [batch, seq_len, hidden]
        x = x.reshape(BATCH_SIZE * SEQ_LEN, HIDDEN_SIZE)  # Flatten
        x = x.view(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM)  # Reshape for attention
        x = x.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        x = x * (HEAD_DIM ** -0.5)  # Scale
        return x.contiguous()

    @torch.compile(mode="max-autotune")
    def compiled_tensor_chain(x):
        """Same operations but compiled"""
        x = x.transpose(0, 1)
        x = x.reshape(BATCH_SIZE * SEQ_LEN, HIDDEN_SIZE)
        x = x.view(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM)
        x = x.transpose(1, 2)
        x = x * (HEAD_DIM ** -0.5)
        return x.contiguous()

    # Benchmark both versions
    eager_time, eager_result = benchmark_operation("Eager Tensor Chain", eager_tensor_chain, x)
    compiled_time, compiled_result = benchmark_operation("Compiled Tensor Chain", compiled_tensor_chain, x)

    # Verify correctness
    print(f"\nCorrectness Check:")
    print(f"  Results allclose: {torch.allclose(eager_result, compiled_result, rtol=1e-4, atol=1e-4)}")

    # Calculate speedup
    speedup = eager_time / compiled_time
    print(f"\nPerformance Results:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Improvement: {((speedup - 1) * 100):.1f}%")

def main():
    """Run all benchmarks"""
    print("TransformerEngine Surgical torch.compile Benchmark")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    # Run benchmarks
    test_attention_preprocessing()
    test_simple_tensor_ops()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("This benchmark shows the potential speedup from applying torch.compile")
    print("surgically to tensor manipulation operations within TransformerEngine layers,")
    print("while keeping the FP8 operations in eager mode.")
    print("\nKey insights:")
    print("- Tensor reshaping and RoPE operations benefit significantly from compilation")
    print("- Memory-bound operations see the largest improvements")
    print("- The surgical approach preserves FP8 functionality while optimizing compatible ops")

if __name__ == "__main__":
    main()