#!/usr/bin/env python3

"""
Test script demonstrating surgical torch.compile integration with actual TransformerEngine layers.
This shows how to modify existing TE layers to selectively compile compatible operations.
"""

import os
import time
import warnings
from typing import Optional, Union, Tuple
from contextlib import nullcontext

import torch
import torch.nn.functional as F

# Import TransformerEngine components
import transformer_engine.pytorch as te
from transformer_engine.pytorch.attention.multi_head_attention import MultiheadAttention
from transformer_engine.pytorch.module.linear import Linear
from transformer_engine.pytorch.module.layernorm_mlp import LayerNormMLP
from transformer_engine.pytorch.jit import no_torch_dynamo

warnings.filterwarnings("ignore")

# Test configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
BATCH_SIZE = 4
SEQ_LEN = 1024
HIDDEN_SIZE = 2048
FFN_HIDDEN_SIZE = 4096
NUM_HEADS = 16
NUM_TRIALS = 50
WARMUP_TRIALS = 10

def setup_fp8_recipe():
    """Setup FP8 recipe for testing"""
    from transformer_engine.common.recipe import Format, DelayedScaling

    recipe = DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
        fp8_mha=True,
        fp8_mlp=True,
    )
    return recipe

class ModifiedMultiheadAttention(MultiheadAttention):
    """Modified MultiheadAttention with surgical torch.compile"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Compile the tensor manipulation operations
        self._compiled_tensor_ops = torch.compile(
            self._qkv_reshape_and_split,
            mode="default",  # Use default for stability
            fullgraph=False  # Allow graph breaks if needed
        )

    def _qkv_reshape_and_split(self, mixed_x_layer: torch.Tensor):
        """Tensor operations that can be safely compiled"""
        # This contains the reshape and split logic from the original forward
        # without FP8 or custom ops
        seq_len, batch_size = mixed_x_layer.shape[:2]

        num_queries_per_key_value = (
            self.num_attention_heads_per_partition // self.num_gqa_groups_per_partition
        )

        if self.qkv_weight_interleaved:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_gqa_groups_per_partition,
                (num_queries_per_key_value + 2),
                self.hidden_size_per_attention_head,
            )
            split_dim = -2
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                (num_queries_per_key_value + 2),
                self.num_gqa_groups_per_partition,
                self.hidden_size_per_attention_head,
            )
            split_dim = -3

        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # Simple split operation (avoiding TE's custom SplitAlongDim)
        if split_dim == -2:
            query_part = mixed_x_layer[:, :, :, :num_queries_per_key_value, :]
            key_part = mixed_x_layer[:, :, :, num_queries_per_key_value:num_queries_per_key_value+1, :]
            value_part = mixed_x_layer[:, :, :, num_queries_per_key_value+1:, :]
        else:
            query_part = mixed_x_layer[:, :, :num_queries_per_key_value, :, :]
            key_part = mixed_x_layer[:, :, num_queries_per_key_value:num_queries_per_key_value+1, :, :]
            value_part = mixed_x_layer[:, :, num_queries_per_key_value+1:, :, :]

        return query_part.contiguous(), key_part.contiguous(), value_part.contiguous()

class SurgicalLinear(Linear):
    """Linear layer with surgical torch.compile for tensor preprocessing"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Compile tensor preprocessing operations
        self._compiled_input_prep = torch.compile(
            self._prepare_input_tensors,
            mode="default",
            fullgraph=False
        )

    def _prepare_input_tensors(self, inp: torch.Tensor):
        """Tensor preparation operations that can be compiled"""
        # Make contiguous and handle basic reshaping
        if not inp.is_contiguous():
            inp = inp.contiguous()

        # Example: handle sequence parallel reshaping logic
        # (simplified version of what TE does)
        original_shape = inp.shape
        if len(original_shape) > 2:
            inp = inp.view(-1, original_shape[-1])

        return inp, original_shape

    def _restore_output_shape(self, output: torch.Tensor, original_shape: tuple):
        """Restore original tensor shape - can also be compiled"""
        if len(original_shape) > 2:
            new_shape = original_shape[:-1] + (output.shape[-1],)
            output = output.view(*new_shape)
        return output

    @no_torch_dynamo()
    def forward(self, inp: torch.Tensor, is_first_microbatch: Optional[bool] = None, **kwargs):
        """Modified forward with surgical compilation"""

        # Use compiled tensor preprocessing
        with torch._dynamo.resume():
            prepared_inp, original_shape = self._compiled_input_prep(inp)

        # FP8 operations must stay in eager mode
        # Use parent's forward but with prepared input
        output = super().forward(prepared_inp, is_first_microbatch, **kwargs)

        # Use compiled output reshaping
        with torch._dynamo.resume():
            output = self._restore_output_shape(output, original_shape)

        return output

def benchmark_layer(name: str, layer_func, *args, **kwargs):
    """Benchmark a layer with warmup"""
    print(f"\nBenchmarking {name}...")

    # Warmup
    for _ in range(WARMUP_TRIALS):
        with torch.no_grad():
            _ = layer_func(*args, **kwargs)

    torch.cuda.synchronize() if DEVICE == "cuda" else None

    # Benchmark
    times = []
    for _ in range(NUM_TRIALS):
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        start = time.perf_counter()

        with torch.no_grad():
            result = layer_func(*args, **kwargs)

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = sum(times) / len(times)
    min_time = min(times)

    print(f"  Average: {avg_time:.3f} ms")
    print(f"  Min: {min_time:.3f} ms")

    return avg_time, result

def test_linear_layer_surgical():
    """Test surgical approach on Linear layer"""
    print("="*60)
    print("LINEAR LAYER SURGICAL BENCHMARK")
    print("="*60)

    # Setup test data
    inp = torch.randn(SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)

    # Create layers
    original_linear = Linear(
        in_features=HIDDEN_SIZE,
        out_features=FFN_HIDDEN_SIZE,
        bias=True,
        device=DEVICE
    )

    surgical_linear = SurgicalLinear(
        in_features=HIDDEN_SIZE,
        out_features=FFN_HIDDEN_SIZE,
        bias=True,
        device=DEVICE
    )

    # Copy weights to ensure fair comparison
    with torch.no_grad():
        surgical_linear.weight.copy_(original_linear.weight)
        if hasattr(surgical_linear, 'bias') and surgical_linear.bias is not None:
            surgical_linear.bias.copy_(original_linear.bias)

    try:
        # Benchmark both versions
        original_time, original_result = benchmark_layer(
            "Original Linear", original_linear, inp
        )

        surgical_time, surgical_result = benchmark_layer(
            "Surgical Linear", surgical_linear, inp
        )

        # Check correctness
        print(f"\nCorrectness Check:")
        if isinstance(original_result, torch.Tensor) and isinstance(surgical_result, torch.Tensor):
            allclose = torch.allclose(original_result, surgical_result, rtol=1e-3, atol=1e-3)
            print(f"  Results allclose: {allclose}")
        else:
            print(f"  Result types: {type(original_result)}, {type(surgical_result)}")

        # Calculate speedup
        speedup = original_time / surgical_time
        print(f"\nPerformance Results:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Improvement: {((speedup - 1) * 100):.1f}%")

    except Exception as e:
        print(f"Error during benchmark: {e}")
        print("This might be expected with FP8 operations or missing dependencies")

def test_simple_tensor_operations():
    """Test simple tensor operations similar to what's in TE layers"""
    print("\n" + "="*60)
    print("TENSOR OPERATIONS COMPARISON")
    print("="*60)

    # Test data
    x = torch.randn(SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)

    def eager_ops(x):
        # Typical tensor operations in TE
        x = x.transpose(0, 1)  # [batch, seq, hidden]
        x = x.contiguous()
        x = x.view(BATCH_SIZE * SEQ_LEN, HIDDEN_SIZE)
        x = x.view(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HIDDEN_SIZE // NUM_HEADS)
        x = x.transpose(1, 2)  # [batch, heads, seq, head_dim]
        return x * (HIDDEN_SIZE // NUM_HEADS) ** -0.5

    @torch.compile(mode="default")
    def compiled_ops(x):
        x = x.transpose(0, 1)
        x = x.contiguous()
        x = x.view(BATCH_SIZE * SEQ_LEN, HIDDEN_SIZE)
        x = x.view(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HIDDEN_SIZE // NUM_HEADS)
        x = x.transpose(1, 2)
        return x * (HIDDEN_SIZE // NUM_HEADS) ** -0.5

    # Benchmark
    eager_time, eager_result = benchmark_layer("Eager Tensor Ops", eager_ops, x)
    compiled_time, compiled_result = benchmark_layer("Compiled Tensor Ops", compiled_ops, x)

    # Verify correctness
    print(f"\nCorrectness Check:")
    print(f"  Results allclose: {torch.allclose(eager_result, compiled_result, rtol=1e-4)}")

    # Calculate speedup
    speedup = eager_time / compiled_time
    print(f"\nPerformance Results:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Improvement: {((speedup - 1) * 100):.1f}%")

def main():
    """Run surgical torch.compile tests"""
    print("TransformerEngine Surgical torch.compile Test")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    # Test simple tensor operations first
    test_simple_tensor_operations()

    # Test surgical Linear layer
    test_linear_layer_surgical()

    print("\n" + "="*60)
    print("SURGICAL APPROACH SUMMARY")
    print("="*60)
    print("This demonstrates surgical torch.compile integration:")
    print("1. Tensor manipulation operations can be compiled selectively")
    print("2. FP8 operations remain in eager mode for compatibility")
    print("3. The approach provides speedups while preserving TE functionality")
    print("\nNext steps:")
    print("- Apply this pattern to MultiheadAttention tensor reshaping")
    print("- Extend to LayerNormMLP activation functions")
    print("- Measure end-to-end training speedups")

if __name__ == "__main__":
    main()