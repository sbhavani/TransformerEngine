#!/usr/bin/env python3

"""
Simplified realistic surgical torch.compile benchmark
Uses TE Linear layers with standard PyTorch attention to avoid TE attention backend issues
while still providing realistic FP8 linear layer performance characteristics.
"""

import os
import time
import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

warnings.filterwarnings("ignore")

# Test configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TRIALS = 20
WARMUP_TRIALS = 5

def setup_fp8_recipe():
    """Setup FP8 recipe for realistic testing"""
    return DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
        fp8_mha=True,
        fp8_mlp=True,
        override_linear_precision=(False, False, True),
    )

def create_rotary_pos_emb(seq_len: int, head_dim: int, device: str):
    """Create rotary position embeddings"""
    position = torch.arange(seq_len, dtype=torch.bfloat16, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, head_dim, 2, dtype=torch.bfloat16, device=device) *
                         -(math.log(10000.0) / head_dim))

    pe = torch.zeros(seq_len, head_dim, dtype=torch.bfloat16, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

def apply_rope(q: torch.Tensor, k: torch.Tensor, pe: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding"""
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

def qkv_reshape_split(qkv: torch.Tensor, num_heads: int, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reshape and split QKV tensor"""
    batch, seq_len, _ = qkv.shape

    # Reshape to [batch, seq_len, 3, num_heads, head_dim]
    qkv_reshaped = qkv.view(batch, seq_len, 3, num_heads, head_dim)

    # Split and transpose: [batch, num_heads, seq_len, head_dim]
    q, k, v = qkv_reshaped.unbind(dim=2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    return q, k, v

def attention_combine(attn_out: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """Combine attention output"""
    batch, heads, seq_len, head_dim = attn_out.shape

    # Transpose and reshape back
    attn_out = attn_out.transpose(1, 2)
    attn_out = attn_out.contiguous().view(batch, seq_len, hidden_size)

    return attn_out

def silu_gate(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SiLU gating for MLP"""
    return F.silu(gate) * up

# Compiled versions
apply_rope_compiled = torch.compile(apply_rope, mode="max-autotune")
qkv_reshape_split_compiled = torch.compile(qkv_reshape_split, mode="default")
attention_combine_compiled = torch.compile(attention_combine, mode="default")
silu_gate_compiled = torch.compile(silu_gate, mode="default")

class RealisticLlamaLayer(nn.Module):
    """
    Realistic Llama layer using TE FP8 linear layers for major compute
    with surgical torch.compile on tensor operations
    """

    def __init__(self, config: dict, use_surgical: bool = False):
        super().__init__()
        self.config = config
        self.use_surgical = use_surgical

        hidden_size = config["hidden_size"]
        ffn_hidden_size = config["ffn_hidden_size"]
        num_heads = config["num_heads"]
        self.head_dim = hidden_size // num_heads

        # TE Fused LayerNorm + Linear layers (major compute, highly optimized)
        self.qkv_norm_linear = te.LayerNormLinear(
            hidden_size,
            3 * hidden_size,
            eps=1e-5,
            bias=False,
            normalization="RMSNorm",
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

        # TE Fused LayerNorm + MLP (major compute, highly optimized)
        self.mlp_norm_fused = te.LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            eps=1e-5,
            bias=False,
            normalization="RMSNorm",
            activation="swiglu",
            device=DEVICE,
            params_dtype=torch.bfloat16
        )

        # RoPE embeddings
        self.register_buffer("pe", create_rotary_pos_emb(config["seq_len"], self.head_dim, DEVICE))

        # Choose compiled vs eager tensor operations
        if use_surgical:
            self.rope_fn = apply_rope_compiled
            self.qkv_split_fn = qkv_reshape_split_compiled
            self.attn_combine_fn = attention_combine_compiled
            self.silu_gate_fn = silu_gate_compiled
        else:
            self.rope_fn = apply_rope
            self.qkv_split_fn = qkv_reshape_split
            self.attn_combine_fn = attention_combine
            self.silu_gate_fn = silu_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with surgical compilation and TE fused layers"""
        residual = x

        # Fused LayerNorm + QKV projection (TE optimized - major compute, stays eager)
        qkv = self.qkv_norm_linear(x)

        # Reshape and split (tensor ops - compiled in surgical mode)
        q, k, v = self.qkv_split_fn(qkv, self.config["num_heads"], self.head_dim)

        # Apply RoPE (tensor ops - compiled in surgical mode)
        q, k = self.rope_fn(q, k, self.pe)

        # Attention computation using cuDNN SDPA
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True  # Use causal mask for language modeling
        )

        # Combine attention output (tensor ops - compiled in surgical mode)
        attn_out = self.attn_combine_fn(attn_out, self.config["hidden_size"])

        # Output projection (FP8 - major compute, stays eager)
        attn_out = self.out_proj(attn_out)

        # Residual connection
        x = residual + attn_out

        # Fused LayerNorm + MLP (TE optimized - major compute, stays eager)
        # This handles: norm + gate_proj + up_proj + silu + multiply + down_proj
        mlp_out = self.mlp_norm_fused(x)

        return x + mlp_out

class RealisticLlamaModel(nn.Module):
    """Realistic Llama model with configurable size"""

    def __init__(self, config: dict, use_surgical: bool = False):
        super().__init__()
        self.config = config

        # Embedding layer (not FP8 in this simplified version)
        self.embed_tokens = nn.Embedding(
            config["vocab_size"], config["hidden_size"], device=DEVICE, dtype=torch.bfloat16
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            RealisticLlamaLayer(config, use_surgical=use_surgical)
            for _ in range(config["num_layers"])
        ])

        # Final norm and LM head
        self.norm = nn.RMSNorm(config["hidden_size"], device=DEVICE, dtype=torch.bfloat16)
        self.lm_head = nn.Linear(
            config["hidden_size"], config["vocab_size"], bias=False, device=DEVICE, dtype=torch.bfloat16
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

def benchmark_model_with_memory(model, input_ids, name: str) -> Tuple[float, dict]:
    """Benchmark model with memory profiling"""
    print(f"\nBenchmarking {name}...")

    # Memory baseline
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline_memory = torch.cuda.memory_allocated()

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_TRIALS):
            _ = model(input_ids)

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
            output = model(input_ids)
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

    return avg_time, memory_stats

def create_llama_config(model_size: str) -> dict:
    """Create realistic Llama configurations"""
    configs = {
        "1b": {
            "vocab_size": 32000,
            "hidden_size": 2048,
            "ffn_hidden_size": 5504,
            "num_layers": 16,
            "num_heads": 16,
            "seq_len": 2048,
        },
        "3b": {
            "vocab_size": 32000,
            "hidden_size": 3072,
            "ffn_hidden_size": 8192,
            "num_layers": 26,
            "num_heads": 24,
            "seq_len": 2048,
        },
        "8b": {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "ffn_hidden_size": 11008,
            "num_layers": 32,
            "num_heads": 32,
            "seq_len": 2048,
        },
    }
    return configs[model_size]

def run_realistic_benchmark(model_size: str = "1b", batch_size: int = 1):
    """Run realistic benchmark"""
    print(f"\n{'='*80}")
    print(f"REALISTIC SURGICAL BENCHMARK: {model_size.upper()}")
    print(f"{'='*80}")

    config = create_llama_config(model_size)
    seq_len = config["seq_len"]

    print(f"Configuration:")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  FFN hidden size: {config['ffn_hidden_size']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  Num heads: {config['num_heads']}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")

    # Estimate parameters (more accurate)
    layer_params = (
        config["hidden_size"] * config["ffn_hidden_size"] * 3 +  # gate + up + down
        config["hidden_size"] * config["hidden_size"] * 4 +      # qkv + o
        config["hidden_size"] * 2                                # norms
    )
    total_params = layer_params * config["num_layers"] + config["vocab_size"] * config["hidden_size"] * 2

    print(f"  Estimated params: {total_params / 1e6:.1f}M")

    # Setup FP8
    fp8_recipe = setup_fp8_recipe()

    # Create input tensors
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=DEVICE)

    try:
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            print(f"\nCreating models...")

            # Create baseline and surgical models
            baseline_model = RealisticLlamaModel(config, use_surgical=False)
            surgical_model = RealisticLlamaModel(config, use_surgical=True)

            # Copy weights for fair comparison
            print("Copying weights...")
            surgical_model.load_state_dict(baseline_model.state_dict())

            # Benchmark both versions
            baseline_time, baseline_memory = benchmark_model_with_memory(
                baseline_model, input_ids, "Baseline (All Eager)"
            )

            surgical_time, surgical_memory = benchmark_model_with_memory(
                surgical_model, input_ids, "Surgical (Compiled Tensor Ops)"
            )

            # Verify correctness
            print(f"\nVerifying correctness...")
            with torch.no_grad():
                baseline_output = baseline_model(input_ids)
                surgical_output = surgical_model(input_ids)

                output_close = torch.allclose(baseline_output, surgical_output, rtol=1e-3, atol=1e-3)
                print(f"Outputs match: {output_close}")

                if not output_close:
                    max_diff = torch.abs(baseline_output - surgical_output).max()
                    print(f"Max difference: {max_diff:.6f}")

            # Calculate results
            speedup = baseline_time / surgical_time
            improvement = (speedup - 1) * 100

            print(f"\n{'='*80}")
            print(f"REALISTIC RESULTS: {model_size.upper()}")
            print(f"{'='*80}")
            print(f"Model Parameters: ~{total_params/1e6:.1f}M")
            print(f"Baseline time:    {baseline_time:.3f} ms")
            print(f"Surgical time:    {surgical_time:.3f} ms")
            print(f"Speedup:          {speedup:.3f}x")
            print(f"Improvement:      {improvement:+.2f}%")

            # Realistic compute analysis
            linear_flops = config["num_layers"] * (
                config["hidden_size"] * config["ffn_hidden_size"] * 3 +
                config["hidden_size"] * config["hidden_size"] * 4
            ) * batch_size * seq_len * 2

            tensor_flops = config["num_layers"] * config["num_heads"] * batch_size * seq_len * (
                config["hidden_size"] // config["num_heads"]
            ) * seq_len * 4  # Attention + RoPE

            linear_percentage = linear_flops / (linear_flops + tensor_flops) * 100
            tensor_percentage = 100 - linear_percentage

            print(f"\nRealistic Compute Analysis:")
            print(f"  FP8 Linear layers: ~{linear_percentage:.1f}% of compute (stays eager)")
            print(f"  Tensor operations: ~{tensor_percentage:.1f}% of compute (compiled)")

            if speedup >= 1.05:
                print(f"✅ EXCELLENT: {speedup:.3f}x speedup at production scale!")
                print(f"   This justifies surgical torch.compile for {model_size.upper()} models")
            elif speedup >= 1.02:
                print(f"✅ GOOD: {speedup:.3f}x speedup worthwhile for production")
            else:
                print(f"⚠️  LIMITED: {speedup:.3f}x speedup may not justify complexity")

    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run realistic surgical torch.compile benchmark"""
    print("Realistic Surgical torch.compile Benchmark")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch version: {torch.__version__}")

    # Test different model sizes
    model_sizes = ["1b", "3b", "8b"]

    for model_size in model_sizes:
        try:
            # Adjust batch size based on model size to fit in memory
            batch_size = {"1b": 4, "3b": 2, "8b": 1}[model_size]
            run_realistic_benchmark(model_size, batch_size=batch_size)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ SKIPPED {model_size}: Out of memory")
                break
            else:
                raise e
        except KeyboardInterrupt:
            print(f"\n⚠️  INTERRUPTED at {model_size}")
            break

    print(f"\n{'='*80}")
    print("REALISTIC ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print("Key insights:")
    print("- Uses TE FP8 linear layers for major compute (85-90%)")
    print("- Compiles only tensor operations (10-15%)")
    print("- Avoids TE attention backend complexity")
    print("- Provides realistic production performance estimates")

if __name__ == "__main__":
    main()