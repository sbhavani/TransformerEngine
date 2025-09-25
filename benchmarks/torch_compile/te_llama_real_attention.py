#!/usr/bin/env python3

"""
Real TE Attention Benchmark - Fair Comparison
This provides a fair comparison between actual TE attention mechanisms
and capturable wrappers, rather than comparing PyTorch attention variants.

Comparison Matrix:
1. Baseline: Real TE attention (eager)
2. TE + cuDNN: TE linear layers + cuDNN SDPA
3. TE + Capturable: TE linear + capturable attention preprocessing
4. Full Capturable: Attempts to make TE attention itself capturable
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
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.jit import no_torch_dynamo

warnings.filterwarnings("ignore")

# Test configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TRIALS = 20
WARMUP_TRIALS = 5

# H100 optimized settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print(f"Using GPU: {torch.cuda.get_device_name()}")

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

# =============================================================================
# CAPTURABLE TENSOR OPERATIONS (from successful patterns)
# =============================================================================

def qkv_reshape_split(qkv: torch.Tensor, num_heads: int, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reshape and split QKV tensor - proven to be capturable"""
    batch, seq_len, _ = qkv.shape

    # Reshape to [batch, seq_len, 3, num_heads, head_dim]
    qkv_reshaped = qkv.view(batch, seq_len, 3, num_heads, head_dim)

    # Split and transpose: [batch, num_heads, seq_len, head_dim]
    q, k, v = qkv_reshaped.unbind(dim=2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    return q, k, v

def apply_rope(q: torch.Tensor, k: torch.Tensor, pe: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding - proven to be capturable"""
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

def attention_combine(attn_out: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """Combine attention output - proven to be capturable"""
    batch, heads, seq_len, head_dim = attn_out.shape

    # Transpose and reshape back
    attn_out = attn_out.transpose(1, 2)
    attn_out = attn_out.contiguous().view(batch, seq_len, hidden_size)

    return attn_out

# Compiled versions
qkv_reshape_split_compiled = torch.compile(qkv_reshape_split, mode="default")
apply_rope_compiled = torch.compile(apply_rope, mode="max-autotune")
attention_combine_compiled = torch.compile(attention_combine, mode="default")

class RealTEAttentionLayer(nn.Module):
    """
    BASELINE: Real TE attention layer using actual TE MultiheadAttention
    This is what we should be comparing against!
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        hidden_size = config["hidden_size"]
        ffn_hidden_size = config["ffn_hidden_size"]
        num_heads = config["num_heads"]

        # Input layer norm
        self.input_layernorm = nn.RMSNorm(hidden_size, device=DEVICE, dtype=torch.bfloat16)

        # REAL TE ATTENTION MODULE
        self.self_attention = te.MultiheadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            bias=False,
            attention_dropout=0.0,
            fuse_qkv_params=True,
            device=DEVICE,
            dtype=torch.bfloat16,
            params_dtype=torch.bfloat16
        )

        # Post attention layer norm
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, device=DEVICE, dtype=torch.bfloat16)

        # TE Fused LayerNorm + MLP
        self.mlp = te.LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            eps=1e-5,
            bias=False,
            normalization="RMSNorm",
            activation="swiglu",
            device=DEVICE,
            params_dtype=torch.bfloat16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with REAL TE attention"""
        residual = x

        # Pre-attention norm
        x = self.input_layernorm(x)

        # REAL TE ATTENTION (this is what we want to make capturable)
        with no_torch_dynamo():
            attn_out = self.self_attention(x)

        # Residual connection
        x = residual + attn_out

        # MLP with fused LayerNorm
        mlp_out = self.mlp(x)

        return x + mlp_out

class HybridTELayer(nn.Module):
    """
    HYBRID: TE linear layers + cuDNN SDPA (our current best approach)
    """

    def __init__(self, config: dict, use_compilation: bool = False):
        super().__init__()
        self.config = config
        self.use_compilation = use_compilation

        hidden_size = config["hidden_size"]
        ffn_hidden_size = config["ffn_hidden_size"]
        num_heads = config["num_heads"]
        self.head_dim = hidden_size // num_heads

        # TE Fused LayerNorm + Linear layers
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

        # TE Fused LayerNorm + MLP
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

        # Choose compilation strategy
        if use_compilation:
            self.qkv_split_fn = qkv_reshape_split_compiled
            self.rope_fn = apply_rope_compiled
            self.combine_fn = attention_combine_compiled
        else:
            self.qkv_split_fn = qkv_reshape_split
            self.rope_fn = apply_rope
            self.combine_fn = attention_combine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with TE linear + cuDNN attention"""
        residual = x

        # Fused LayerNorm + QKV projection (TE optimized)
        qkv = self.qkv_norm_linear(x)

        # Reshape and split (compiled or not)
        q, k, v = self.qkv_split_fn(qkv, self.config["num_heads"], self.head_dim)

        # Apply RoPE (compiled or not)
        q, k = self.rope_fn(q, k, self.pe)

        # cuDNN SDPA (highly optimized)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True
        )

        # Combine attention output (compiled or not)
        attn_out = self.combine_fn(attn_out, self.config["hidden_size"])

        # Output projection (TE optimized)
        attn_out = self.out_proj(attn_out)

        # Residual connection
        x = residual + attn_out

        # Fused LayerNorm + MLP (TE optimized)
        mlp_out = self.mlp_norm_fused(x)

        return x + mlp_out

class CaptureCompatibleTEAttentionLayer(nn.Module):
    """
    EXPERIMENTAL: Attempt to make TE attention capturable
    This wraps TE attention with capturable preprocessing/postprocessing
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        hidden_size = config["hidden_size"]
        ffn_hidden_size = config["ffn_hidden_size"]
        num_heads = config["num_heads"]

        # Input layer norm (separate for capturable preprocessing)
        self.input_layernorm = nn.RMSNorm(hidden_size, device=DEVICE, dtype=torch.bfloat16)

        # TE attention module (will be kept eager)
        self.te_attention = te.MultiheadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            bias=False,
            attention_dropout=0.0,
            fuse_qkv_params=True,
            device=DEVICE,
            dtype=torch.bfloat16,
            params_dtype=torch.bfloat16
        )

        # Post attention layer norm
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, device=DEVICE, dtype=torch.bfloat16)

        # TE Fused MLP
        self.mlp = te.LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            eps=1e-5,
            bias=False,
            normalization="RMSNorm",
            activation="swiglu",
            device=DEVICE,
            params_dtype=torch.bfloat16
        )

        # Compiled preprocessing/postprocessing
        self.input_prep = torch.compile(self._input_preprocessing, mode="default")
        self.output_prep = torch.compile(self._output_postprocessing, mode="default")

    def _input_preprocessing(self, x: torch.Tensor) -> torch.Tensor:
        """Capturable input preprocessing"""
        # Any tensor operations that can be compiled before TE attention
        return x  # For now, just pass through

    def _output_postprocessing(self, x: torch.Tensor) -> torch.Tensor:
        """Capturable output postprocessing"""
        # Any tensor operations that can be compiled after TE attention
        return x  # For now, just pass through

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with capturable TE attention wrapper"""
        residual = x

        # Capturable preprocessing
        x = self.input_prep(x)

        # Pre-attention norm
        x = self.input_layernorm(x)

        # TE attention (stays eager - this is the bottleneck we want to optimize)
        with no_torch_dynamo():
            attn_out = self.te_attention(x)

        # Capturable postprocessing
        attn_out = self.output_prep(attn_out)

        # Residual connection
        x = residual + attn_out

        # MLP with fused LayerNorm
        mlp_out = self.mlp(x)

        return x + mlp_out

class RealTEBenchmarkModel(nn.Module):
    """Model for real TE attention benchmarking"""

    def __init__(self, config: dict, layer_type: str = "real_te"):
        super().__init__()
        self.config = config

        # Embedding layer
        self.embed_tokens = nn.Embedding(
            config["vocab_size"], config["hidden_size"], device=DEVICE, dtype=torch.bfloat16
        )

        # Transformer layers
        if layer_type == "real_te":
            layer_class = RealTEAttentionLayer
            layer_args = [config]
        elif layer_type == "hybrid_eager":
            layer_class = HybridTELayer
            layer_args = [config, False]  # use_compilation=False
        elif layer_type == "hybrid_compiled":
            layer_class = HybridTELayer
            layer_args = [config, True]   # use_compilation=True
        elif layer_type == "capturable_te":
            layer_class = CaptureCompatibleTEAttentionLayer
            layer_args = [config]
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}")

        self.layers = nn.ModuleList([
            layer_class(*layer_args)
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

    # Set model to eval mode
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_TRIALS):
            try:
                _ = model(input_ids)
            except Exception as e:
                print(f"  Error during warmup: {e}")
                return float('inf'), {}

    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()

    # Reset for actual benchmark
    torch.cuda.reset_peak_memory_stats()

    # Actual benchmark
    times = []
    with torch.no_grad():
        for _ in range(NUM_TRIALS):
            try:
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = model(input_ids)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
            except Exception as e:
                print(f"  Error during trial: {e}")
                return float('inf'), {}

    if not times:
        return float('inf'), {}

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

def run_real_te_comparison(model_size: str = "1b", batch_size: int = 1):
    """Compare real TE attention vs hybrid approaches"""
    print(f"\n{'='*80}")
    print(f"REAL TE ATTENTION BENCHMARK: {model_size.upper()}")
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

    # Estimate parameters
    total_params = (
        config["hidden_size"] * config["ffn_hidden_size"] * 3 +
        config["hidden_size"] * config["hidden_size"] * 4 +
        config["hidden_size"] * 2
    ) * config["num_layers"]

    print(f"  Estimated params: {total_params / 1e6:.1f}M")

    # Setup FP8
    fp8_recipe = setup_fp8_recipe()

    # Initialize FP8 state manager for H100
    FP8GlobalStateManager.reset()

    # Create input tensors
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=DEVICE)

    try:
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            print(f"\nCreating models...")

            # Create different model variants with error handling
            models = {}
            model_creation_order = ["hybrid_eager", "hybrid_compiled", "real_te", "capturable_te"]

            for model_type in model_creation_order:
                try:
                    print(f"  Creating {model_type} model...")
                    models[model_type] = RealTEBenchmarkModel(config, layer_type=model_type)
                    models[model_type].eval()  # Set to eval mode
                except Exception as e:
                    print(f"  ❌ Failed to create {model_type}: {e}")
                    continue

            # Verify we have at least some models
            if not models:
                print("❌ No models could be created!")
                return

            print(f"  Successfully created {len(models)} model variants")

            # Benchmark all variants
            results = {}
            variant_descriptions = {
                "real_te": "Real TE Attention (Baseline)",
                "hybrid_eager": "TE Linear + cuDNN SDPA (Eager)",
                "hybrid_compiled": "TE Linear + cuDNN SDPA (Compiled)",
                "capturable_te": "Capturable TE Attention Wrapper"
            }

            for variant_name, model in models.items():
                try:
                    description = variant_descriptions.get(variant_name, variant_name)
                    results[variant_name] = benchmark_model_with_memory(model, input_ids, description)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  ❌ {variant_name}: Out of GPU memory")
                        torch.cuda.empty_cache()
                    else:
                        print(f"  ❌ {variant_name}: Runtime error - {e}")
                    results[variant_name] = (float('inf'), {})
                except Exception as e:
                    print(f"  ❌ {variant_name}: Unexpected error - {e}")
                    results[variant_name] = (float('inf'), {})

            # Filter out failed results
            valid_results = {k: v for k, v in results.items() if v[0] != float('inf')}

            if not valid_results:
                print("❌ All benchmarks failed!")
                return

            # Calculate speedups
            if "real_te" in valid_results:
                baseline_time = valid_results["real_te"][0]
                baseline_name = "real_te"
            else:
                # Fallback to first valid result
                baseline_name = list(valid_results.keys())[0]
                baseline_time = valid_results[baseline_name][0]

            print(f"\n{'='*80}")
            print(f"REAL TE COMPARISON RESULTS: {model_size.upper()}")
            print(f"{'='*80}")
            print(f"Baseline: {variant_descriptions[baseline_name]}")

            for variant_name, (time_ms, memory_stats) in valid_results.items():
                speedup = baseline_time / time_ms
                improvement = (speedup - 1) * 100
                description = variant_descriptions[variant_name]

                status = "BASELINE" if variant_name == baseline_name else f"{speedup:.3f}x"
                print(f"{description:35s}: {time_ms:7.3f} ms | {status:>8s} | {improvement:+6.2f}%")

            # Analysis
            best_variant = min(valid_results.items(), key=lambda x: x[1][0])
            best_speedup = baseline_time / best_variant[1][0] if baseline_name != best_variant[0] else 1.0

            print(f"\n{'='*25} ANALYSIS {'='*25}")
            print(f"Best performing variant: {variant_descriptions[best_variant[0]]}")

            if best_variant[0] != baseline_name:
                best_improvement = (best_speedup - 1) * 100
                print(f"Best speedup: {best_speedup:.3f}x ({best_improvement:+.2f}%)")

                if best_speedup >= 1.10:
                    print(f"✅ EXCELLENT: Significant improvement over real TE attention!")
                elif best_speedup >= 1.05:
                    print(f"✅ GOOD: Measurable improvement over real TE attention")
                else:
                    print(f"⚠️  LIMITED: Minimal improvement over real TE attention")
            else:
                print(f"📊 Real TE attention is the fastest approach")

            print(f"\nKey insights:")
            if "hybrid_compiled" in valid_results and "hybrid_eager" in valid_results:
                hybrid_speedup = valid_results["hybrid_eager"][0] / valid_results["hybrid_compiled"][0]
                print(f"- Compilation benefit on hybrid approach: {hybrid_speedup:.3f}x")

            if "real_te" in valid_results and "hybrid_compiled" in valid_results:
                te_vs_hybrid = valid_results["real_te"][0] / valid_results["hybrid_compiled"][0]
                if te_vs_hybrid > 1.05:
                    print(f"- TE attention is {te_vs_hybrid:.3f}x slower than hybrid approach")
                elif te_vs_hybrid < 0.95:
                    print(f"- TE attention is {1/te_vs_hybrid:.3f}x faster than hybrid approach")
                else:
                    print(f"- TE attention and hybrid approach have similar performance")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"❌ BENCHMARK FAILED: GPU out of memory")
            print(f"  Try reducing batch size or model size")
        else:
            print(f"❌ BENCHMARK FAILED: {e}")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run real TE attention benchmark"""
    print("Real TE Attention vs Capturable Approaches Benchmark")
    print("=" * 80)
    print("Fair comparison: Real TE attention vs capturable wrappers")
    print("Goal: Understand the performance baseline and optimization potential")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch version: {torch.__version__}")

    # Test real TE on different model sizes (H100 optimized)
    model_sizes = ["small", "1b", "3b", "8b"]

    for model_size in model_sizes:
        try:
            # Adjust batch size based on model size to fit in H100 memory
            batch_size = {"small": 8, "1b": 4, "3b": 2, "8b": 1}[model_size]
            print(f"\nℹ️  Starting {model_size} benchmark with batch_size={batch_size}")
            run_real_te_comparison(model_size, batch_size=batch_size)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n❌ SKIPPED {model_size}: Out of memory")
                print(f"   Try running with smaller batch size or model size")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
        except KeyboardInterrupt:
            print(f"\n⚠️  INTERRUPTED at {model_size}")
            break

    print(f"\n{'='*80}")
    print("REAL TE ATTENTION BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print("This benchmark provides fair comparison between:")
    print("- Real TE MultiheadAttention (baseline)")
    print("- TE Linear + cuDNN SDPA hybrid approaches")
    print("- Capturable TE attention wrappers")
    print("")
    print("Results show the true performance characteristics and")
    print("optimization potential for TE attention mechanisms.")

if __name__ == "__main__":
    main()