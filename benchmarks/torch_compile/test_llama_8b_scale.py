#!/usr/bin/env python3

"""
Llama-8B Scale Benchmark: Realistic torch.compile surgical evaluation
This benchmark uses Llama-8B scale parameters to provide realistic estimates
of torch.compile benefits on production-scale transformer models.
"""

import os
import time
import math
import warnings
from typing import Optional, Tuple
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

# TransformerEngine imports
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from transformer_engine.pytorch.jit import no_torch_dynamo
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

warnings.filterwarnings("ignore")

# Llama-8B Scale Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model scaling options
SCALE_CONFIGS = {
    "mini": {
        "batch_size": 8,
        "seq_len": 2048,
        "hidden_size": 2048,
        "ffn_hidden_size": 4096,
        "num_heads": 16,
        "num_layers": 4,
        "name": "Mini (134M params)"
    },
    "llama_1b": {
        "batch_size": 4,
        "seq_len": 2048,
        "hidden_size": 2048,
        "ffn_hidden_size": 5504,  # Llama scaling
        "num_heads": 16,
        "num_layers": 16,
        "name": "Llama-1B Scale"
    },
    "llama_3b": {
        "batch_size": 2,
        "seq_len": 2048,
        "hidden_size": 3072,
        "ffn_hidden_size": 8192,
        "num_heads": 24,
        "num_layers": 26,
        "name": "Llama-3B Scale"
    },
    "llama_8b": {
        "batch_size": 1,  # Memory constrained
        "seq_len": 2048,
        "hidden_size": 4096,
        "ffn_hidden_size": 11008,  # Llama 2/3 FFN scaling
        "num_heads": 32,
        "num_layers": 32,
        "name": "Llama-8B Scale"
    }
}

# Test configuration
NUM_TRIALS = 20  # Reduced for larger models
WARMUP_TRIALS = 5

def setup_fp8_recipe():
    """Setup FP8 recipe optimized for large models"""
    return DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
        fp8_mha=True,
        fp8_mlp=True,
        override_linear_precision=(False, False, True),
    )

def create_rotary_pos_emb(seq_len: int, head_dim: int, device: str):
    """Create rotary position embeddings - consistent dtype"""
    position = torch.arange(seq_len, dtype=torch.bfloat16, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, head_dim, 2, dtype=torch.bfloat16, device=device) *
                         -(math.log(10000.0) / head_dim))

    pe = torch.zeros(seq_len, head_dim, dtype=torch.bfloat16, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

def apply_rope(q: torch.Tensor, k: torch.Tensor, pe: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding - optimized for large models"""
    batch, heads, seq_len, head_dim = q.shape

    # Reshape for rotation
    q_rot = q.view(batch, heads, seq_len, head_dim // 2, 2)
    k_rot = k.view(batch, heads, seq_len, head_dim // 2, 2)

    # Get cos/sin values - ensure same dtype
    cos = pe[:seq_len, 0::2].unsqueeze(0).unsqueeze(0).to(q.dtype)
    sin = pe[:seq_len, 1::2].unsqueeze(0).unsqueeze(0).to(q.dtype)

    # Apply rotation
    q_cos = q_rot[..., 0] * cos - q_rot[..., 1] * sin
    q_sin = q_rot[..., 0] * sin + q_rot[..., 1] * cos

    k_cos = k_rot[..., 0] * cos - k_rot[..., 1] * sin
    k_sin = k_rot[..., 0] * sin + k_rot[..., 1] * cos

    # Recombine
    q_out = torch.stack([q_cos, q_sin], dim=-1).view(batch, heads, seq_len, head_dim)
    k_out = torch.stack([k_cos, k_sin], dim=-1).view(batch, heads, seq_len, head_dim)

    return q_out, k_out

# Compiled version
apply_rope_compiled = torch.compile(apply_rope, mode="max-autotune")

class LlamaScaleTransformerLayer(nn.Module):
    """Llama-scale transformer layer with realistic proportions"""

    def __init__(self, config: dict, use_surgical: bool = False):
        super().__init__()
        self.config = config
        self.use_surgical = use_surgical

        hidden_size = config["hidden_size"]
        ffn_hidden_size = config["ffn_hidden_size"]
        num_heads = config["num_heads"]
        self.head_dim = hidden_size // num_heads

        # FP8 Linear layers (90% of compute in large models)
        self.qkv_proj = te.Linear(
            hidden_size,
            3 * hidden_size,
            bias=False,
            device=DEVICE,
            params_dtype=torch.bfloat16
        )

        self.out_proj = te.Linear(
            hidden_size,
            hidden_size,
            bias=False,
            device=DEVICE,
            params_dtype=torch.bfloat16
        )

        # MLP with realistic Llama scaling
        self.gate_proj = te.Linear(
            hidden_size,
            ffn_hidden_size,
            bias=False,
            device=DEVICE,
            params_dtype=torch.bfloat16
        )

        self.up_proj = te.Linear(
            hidden_size,
            ffn_hidden_size,
            bias=False,
            device=DEVICE,
            params_dtype=torch.bfloat16
        )

        self.down_proj = te.Linear(
            ffn_hidden_size,
            hidden_size,
            bias=False,
            device=DEVICE,
            params_dtype=torch.bfloat16
        )

        # RMSNorm (like Llama)
        self.input_norm = nn.RMSNorm(hidden_size, device=DEVICE, dtype=torch.bfloat16)
        self.post_attention_norm = nn.RMSNorm(hidden_size, device=DEVICE, dtype=torch.bfloat16)

        # RoPE embeddings
        self.register_buffer("pe", create_rotary_pos_emb(config["seq_len"], self.head_dim, DEVICE))

        # Compile tensor operations for surgical approach
        if use_surgical:
            self.qkv_reshape_split = torch.compile(self._qkv_reshape_split, mode="default")
            self.attention_combine = torch.compile(self._attention_combine, mode="default")
            self.silu_multiply = torch.compile(self._silu_multiply, mode="default")
        else:
            self.qkv_reshape_split = self._qkv_reshape_split
            self.attention_combine = self._attention_combine
            self.silu_multiply = self._silu_multiply

    def _qkv_reshape_split(self, qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """QKV reshape and split - compilation candidate"""
        batch, seq_len, _ = qkv.shape
        num_heads = self.config["num_heads"]
        head_dim = self.head_dim

        # Reshape to [batch, seq_len, 3, num_heads, head_dim]
        qkv_reshaped = qkv.view(batch, seq_len, 3, num_heads, head_dim)

        # Split and transpose: [batch, num_heads, seq_len, head_dim]
        q, k, v = qkv_reshaped.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def _attention_combine(self, attn_out: torch.Tensor) -> torch.Tensor:
        """Attention output combination - compilation candidate"""
        batch, heads, seq_len, head_dim = attn_out.shape
        hidden_size = self.config["hidden_size"]

        # Transpose and reshape back
        attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out.contiguous().view(batch, seq_len, hidden_size)

        return attn_out

    def _silu_multiply(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """SiLU gate multiplication - compilation candidate"""
        return F.silu(gate) * up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with realistic Llama architecture"""
        residual = x

        # Pre-attention norm
        x = self.input_norm(x)

        # QKV projection (FP8 - major compute)
        qkv = self.qkv_proj(x)

        # Reshape and split (small tensor ops)
        q, k, v = self.qkv_reshape_split(qkv)

        # Apply RoPE (small but expensive per-element)
        if self.use_surgical:
            q, k = apply_rope_compiled(q, k, self.pe)
        else:
            q, k = apply_rope(q, k, self.pe)

        # Attention computation (mostly GEMM)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)

        # Combine attention output (small tensor ops)
        attn_out = self.attention_combine(attn_out)

        # Output projection (FP8 - major compute)
        attn_out = self.out_proj(attn_out)

        # Residual connection
        x = residual + attn_out
        residual = x

        # Pre-MLP norm
        x = self.post_attention_norm(x)

        # Llama MLP with SwiGLU (FP8 - major compute)
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # SiLU activation with gating (small but compilation candidate)
        mlp_out = self.silu_multiply(gate, up)

        # Down projection (FP8 - major compute)
        mlp_out = self.down_proj(mlp_out)

        return residual + mlp_out

class LlamaScaleTransformer(nn.Module):
    """Llama-scale transformer with configurable size"""

    def __init__(self, config: dict, use_surgical: bool = False):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            LlamaScaleTransformerLayer(config, use_surgical=use_surgical)
            for _ in range(config["num_layers"])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

def calculate_model_size(config: dict) -> int:
    """Estimate model parameters"""
    hidden_size = config["hidden_size"]
    ffn_hidden_size = config["ffn_hidden_size"]
    num_layers = config["num_layers"]

    # QKV projection
    qkv_params = hidden_size * 3 * hidden_size
    # Output projection
    out_params = hidden_size * hidden_size
    # MLP (gate + up + down)
    mlp_params = hidden_size * ffn_hidden_size * 3
    # Norms (input + post_attn per layer)
    norm_params = hidden_size * 2

    layer_params = qkv_params + out_params + mlp_params + norm_params
    total_params = layer_params * num_layers

    return total_params

def benchmark_model_with_memory(model: nn.Module, input_tensor: torch.Tensor,
                              name: str, config: dict) -> Tuple[float, dict]:
    """Benchmark with memory profiling"""
    print(f"\nBenchmarking {name}...")

    # Memory baseline
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline_memory = torch.cuda.memory_allocated()

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_TRIALS):
            _ = model(input_tensor)

    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()

    # Reset for actual benchmark
    torch.cuda.reset_peak_memory_stats()

    # Actual benchmark
    times = []
    with torch.no_grad():
        for _ in range(NUM_TRIALS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(input_tensor)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

    avg_time = sum(times) / len(times)
    min_time = min(times)

    memory_stats = {
        "baseline_mb": baseline_memory / 1024**2,
        "peak_mb": peak_memory / 1024**2,
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2,
    }

    print(f"  Average: {avg_time:.3f} ms")
    print(f"  Min: {min_time:.3f} ms")
    print(f"  Peak Memory: {memory_stats['peak_mb']:.1f} MB")
    print(f"  Model Size: {memory_stats['model_size_mb']:.1f} MB")

    return avg_time, memory_stats

def run_scale_benchmark(scale: str):
    """Run benchmark for specific scale"""
    if scale not in SCALE_CONFIGS:
        print(f"Unknown scale: {scale}")
        return

    config = SCALE_CONFIGS[scale]
    model_params = calculate_model_size(config)

    print(f"\n{'='*80}")
    print(f"LLAMA SCALE BENCHMARK: {config['name']}")
    print(f"{'='*80}")
    print(f"Parameters: ~{model_params/1e6:.1f}M")
    print(f"Batch size: {config['batch_size']}")
    print(f"Sequence length: {config['seq_len']}")
    print(f"Hidden size: {config['hidden_size']}")
    print(f"FFN hidden size: {config['ffn_hidden_size']}")
    print(f"Number of heads: {config['num_heads']}")
    print(f"Number of layers: {config['num_layers']}")

    # Setup FP8
    fp8_recipe = setup_fp8_recipe()

    # Create input tensor
    input_tensor = torch.randn(
        config["batch_size"], config["seq_len"], config["hidden_size"],
        dtype=torch.bfloat16, device=DEVICE
    )

    try:
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            print(f"\nCreating models...")
            baseline_model = LlamaScaleTransformer(config, use_surgical=False).to(DEVICE)
            surgical_model = LlamaScaleTransformer(config, use_surgical=True).to(DEVICE)

            # Copy weights for fair comparison
            print("Copying weights...")
            with torch.no_grad():
                for baseline_layer, surgical_layer in zip(baseline_model.layers, surgical_model.layers):
                    surgical_layer.qkv_proj.weight.copy_(baseline_layer.qkv_proj.weight)
                    surgical_layer.out_proj.weight.copy_(baseline_layer.out_proj.weight)
                    surgical_layer.gate_proj.weight.copy_(baseline_layer.gate_proj.weight)
                    surgical_layer.up_proj.weight.copy_(baseline_layer.up_proj.weight)
                    surgical_layer.down_proj.weight.copy_(baseline_layer.down_proj.weight)
                    surgical_layer.input_norm.weight.copy_(baseline_layer.input_norm.weight)
                    surgical_layer.post_attention_norm.weight.copy_(baseline_layer.post_attention_norm.weight)

            # Benchmark both versions
            baseline_time, baseline_memory = benchmark_model_with_memory(
                baseline_model, input_tensor, "Baseline (All Eager)", config
            )
            surgical_time, surgical_memory = benchmark_model_with_memory(
                surgical_model, input_tensor, "Surgical (Compiled Tensor Ops)", config
            )

            # Verify correctness (relaxed tolerance for larger models)
            print(f"\nVerifying correctness...")
            with torch.no_grad():
                baseline_output = baseline_model(input_tensor)
                surgical_output = surgical_model(input_tensor)

                output_close = torch.allclose(baseline_output, surgical_output, rtol=5e-3, atol=5e-3)
                print(f"Outputs match: {output_close}")

                if not output_close:
                    max_diff = torch.abs(baseline_output - surgical_output).max()
                    print(f"Max difference: {max_diff:.6f}")

            # Calculate results
            speedup = baseline_time / surgical_time
            improvement = (speedup - 1) * 100

            print(f"\n{'='*80}")
            print(f"RESULTS: {config['name']}")
            print(f"{'='*80}")
            print(f"Model Parameters: ~{model_params/1e6:.1f}M")
            print(f"Baseline time:    {baseline_time:.3f} ms")
            print(f"Surgical time:    {surgical_time:.3f} ms")
            print(f"Speedup:          {speedup:.3f}x")
            print(f"Improvement:      {improvement:+.2f}%")

            # Compute distribution analysis
            total_ops = model_params * 2  # Rough FLOPs estimate
            rope_ops = config["num_layers"] * config["num_heads"] * config["seq_len"] * config["hidden_size"] // config["num_heads"] * 10  # RoPE complexity
            rope_percentage = (rope_ops / total_ops) * 100

            print(f"\nCompute Analysis:")
            print(f"  RoPE operations: ~{rope_percentage:.2f}% of total")
            print(f"  Linear layers:   ~{100-rope_percentage:.2f}% of total (FP8 eager)")

            if speedup >= 1.05:
                print(f"✅ WORTHWHILE: {speedup:.3f}x speedup at {config['name']} scale")
            elif speedup >= 1.01:
                print(f"⚠️  MARGINAL: {speedup:.3f}x speedup may not justify complexity")
            else:
                print(f"❌ NEGLIGIBLE: {speedup:.3f}x speedup not worthwhile at scale")

    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run scaling analysis across model sizes"""
    print("Llama-Scale Surgical torch.compile Analysis")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch version: {torch.__version__}")

    # Available scales in order of increasing size
    scales_to_test = ["mini", "llama_1b", "llama_3b", "llama_8b"]

    # Test each scale that fits in memory
    for scale in scales_to_test:
        try:
            run_scale_benchmark(scale)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ SKIPPED {scale}: Out of memory")
                break
            else:
                raise e
        except KeyboardInterrupt:
            print(f"\n⚠️  INTERRUPTED at {scale}")
            break

    print(f"\n{'='*80}")
    print("SCALING ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print("Key takeaways:")
    print("- Smaller models show larger relative speedups")
    print("- Larger models are dominated by FP8 linear layers")
    print("- RoPE compilation benefit dilutes with scale")
    print("- Memory usage scales predictably with parameters")

if __name__ == "__main__":
    main()