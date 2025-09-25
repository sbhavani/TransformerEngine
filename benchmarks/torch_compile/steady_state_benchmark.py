#!/usr/bin/env python3

"""
Steady-State Performance Benchmark
Properly separates compilation time from execution time to measure true
algorithmic performance differences without contamination from first-run costs.
"""

import torch
import torch.nn as nn
import time
import math
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

DEVICE = "cuda"

# Dramatically increased warmup for proper compilation amortization
COMPILATION_WARMUP = 100  # For first-time compilation
STEADY_STATE_WARMUP = 20  # For cache warming after compilation
NUM_TRIALS = 50          # More trials for statistical significance

def setup_fp8_recipe():
    return DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
    )

def create_rotary_pos_emb(seq_len: int, head_dim: int, device: str):
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) *
                         -(math.log(10000.0) / head_dim))
    pe = torch.zeros(seq_len, head_dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def apply_rope(q, k, pe):
    batch, heads, seq_len, head_dim = q.shape
    q_rot = q.view(batch, heads, seq_len, head_dim // 2, 2)
    k_rot = k.view(batch, heads, seq_len, head_dim // 2, 2)

    cos = pe[:seq_len, 0::2].unsqueeze(0).unsqueeze(0).to(q.dtype)
    sin = pe[:seq_len, 1::2].unsqueeze(0).unsqueeze(0).to(q.dtype)

    q_cos = q_rot[..., 0] * cos - q_rot[..., 1] * sin
    q_sin = q_rot[..., 0] * sin + q_rot[..., 1] * cos
    k_cos = k_rot[..., 0] * cos - k_rot[..., 1] * sin
    k_sin = k_rot[..., 0] * sin + k_rot[..., 1] * cos

    q_out = torch.stack([q_cos, q_sin], dim=-1).view(batch, heads, seq_len, head_dim)
    k_out = torch.stack([k_cos, k_sin], dim=-1).view(batch, heads, seq_len, head_dim)
    return q_out, k_out

def qkv_reshape_split(qkv, num_heads, head_dim):
    batch, seq_len, _ = qkv.shape
    qkv_reshaped = qkv.view(batch, seq_len, 3, num_heads, head_dim)
    q, k, v = qkv_reshaped.unbind(dim=2)
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

def attention_combine(attn_out, hidden_size):
    batch, heads, seq_len, head_dim = attn_out.shape
    return attn_out.transpose(1, 2).contiguous().view(batch, seq_len, hidden_size)

# Create compiled versions
apply_rope_compiled = torch.compile(apply_rope, mode="max-autotune")
qkv_reshape_split_compiled = torch.compile(qkv_reshape_split, mode="default")
attention_combine_compiled = torch.compile(attention_combine, mode="default")

class HybridOptimalLayer(nn.Module):
    """Our proven optimal hybrid approach"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.head_dim = hidden_size // self.num_heads

        # TE optimized components
        self.qkv_norm_linear = te.LayerNormLinear(
            hidden_size, 3 * hidden_size,
            eps=1e-5, bias=False, normalization="RMSNorm"
        )
        self.o_proj = te.Linear(hidden_size, hidden_size, bias=False)
        self.mlp = te.LayerNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=config["ffn_hidden_size"],
            eps=1e-5, bias=False,
            normalization="RMSNorm", activation="swiglu"
        )

        # RoPE embeddings
        self.pe = create_rotary_pos_emb(config["seq_len"], self.head_dim, DEVICE)

    def forward(self, x):
        residual = x
        batch, seq_len, hidden = x.shape

        # TE fused norm + QKV
        qkv = self.qkv_norm_linear(x)

        # Tensor reshaping (not compiled for baseline)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Manual RoPE (not compiled for baseline)
        q, k = apply_rope(q, k, self.pe)

        # cuDNN SDPA (our proven best attention)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )

        # Combine output (not compiled for baseline)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        attn_output = self.o_proj(attn_output)
        x = residual + attn_output

        # TE fused MLP
        mlp_output = self.mlp(x)
        return x + mlp_output

class HybridCompiledLayer(nn.Module):
    """Our optimal hybrid + surgical compilation"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.head_dim = hidden_size // self.num_heads

        # TE optimized components (same as baseline)
        self.qkv_norm_linear = te.LayerNormLinear(
            hidden_size, 3 * hidden_size,
            eps=1e-5, bias=False, normalization="RMSNorm"
        )
        self.o_proj = te.Linear(hidden_size, hidden_size, bias=False)
        self.mlp = te.LayerNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=config["ffn_hidden_size"],
            eps=1e-5, bias=False,
            normalization="RMSNorm", activation="swiglu"
        )

        # RoPE embeddings
        self.pe = create_rotary_pos_emb(config["seq_len"], self.head_dim, DEVICE)

    def forward(self, x):
        residual = x
        batch, seq_len, hidden = x.shape

        # TE fused norm + QKV (same)
        qkv = self.qkv_norm_linear(x)

        # COMPILED tensor reshaping
        q, k, v = qkv_reshape_split_compiled(qkv, self.num_heads, self.head_dim)

        # COMPILED RoPE
        q, k = apply_rope_compiled(q, k, self.pe)

        # cuDNN SDPA (same - still optimal)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )

        # COMPILED combine output
        attn_output = attention_combine_compiled(attn_output, hidden)
        attn_output = self.o_proj(attn_output)
        x = residual + attn_output

        # TE fused MLP (same)
        mlp_output = self.mlp(x)
        return x + mlp_output

class TEAttentionLayer(nn.Module):
    """TE native attention for comparison"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config["hidden_size"]

        self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-5)

        # TE MultiheadAttention
        self.self_attn = te.MultiheadAttention(
            hidden_size=hidden_size,
            num_attention_heads=config["num_heads"],
            bias=False,
            fuse_qkv_params=True
        )

        # TE fused MLP
        self.mlp = te.LayerNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=config["ffn_hidden_size"],
            eps=1e-5, bias=False,
            normalization="RMSNorm", activation="swiglu"
        )

    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        attn_output = self.self_attn(x)
        x = residual + attn_output
        mlp_output = self.mlp(x)
        return x + mlp_output

class SimpleModel(nn.Module):
    def __init__(self, config, layer_class, name="Model"):
        super().__init__()
        self.config = config
        self.name = name

        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([
            layer_class(config) for _ in range(config["num_layers"])
        ])
        self.norm = nn.RMSNorm(config["hidden_size"], eps=1e-5)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

def benchmark_with_compilation_separation(model, input_ids, name):
    """Benchmark with proper compilation time separation"""
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {name}")
    print(f"{'='*80}")

    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"Phase 1: Compilation warmup ({COMPILATION_WARMUP} iterations)")

    # Phase 1: Compilation warmup (measure compilation overhead)
    compilation_start = time.perf_counter()

    with torch.no_grad():
        for i in range(COMPILATION_WARMUP):
            if i == 0:
                print("  First compilation run...")
            elif i == 10:
                print("  Compilation stabilizing...")
            elif i == 50:
                print("  Deep compilation warmup...")

            _ = model(input_ids)
            torch.cuda.synchronize()

    compilation_time = (time.perf_counter() - compilation_start) * 1000
    print(f"  Total compilation + warmup time: {compilation_time:.1f} ms")
    print(f"  Average per iteration during compilation: {compilation_time/COMPILATION_WARMUP:.3f} ms")

    # Clear memory and reset
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"\nPhase 2: Steady-state warmup ({STEADY_STATE_WARMUP} iterations)")

    # Phase 2: Steady-state warmup (post-compilation cache warming)
    with torch.no_grad():
        for _ in range(STEADY_STATE_WARMUP):
            _ = model(input_ids)
            torch.cuda.synchronize()

    print(f"\nPhase 3: Steady-state measurement ({NUM_TRIALS} trials)")

    # Phase 3: Steady-state measurement
    times = []
    with torch.no_grad():
        for i in range(NUM_TRIALS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(input_ids)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

            if i == 0:
                print("  First steady-state measurement...")
            elif i == NUM_TRIALS // 2:
                print("  Halfway through measurements...")

    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = (sum([(t - avg_time) ** 2 for t in times]) / len(times)) ** 0.5

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    print(f"\n📊 STEADY-STATE RESULTS:")
    print(f"  Average: {avg_time:.3f} ms")
    print(f"  Min: {min_time:.3f} ms")
    print(f"  Max: {max_time:.3f} ms")
    print(f"  Std Dev: {std_time:.3f} ms")
    print(f"  Peak Memory: {peak_memory:.2f} GB")
    print(f"  Compilation overhead: {compilation_time:.1f} ms (one-time cost)")

    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_time': std_time,
        'compilation_time': compilation_time,
        'peak_memory': peak_memory
    }

def main():
    print("=== STEADY-STATE PERFORMANCE BENCHMARK ===")
    print("Proper separation of compilation time from algorithmic performance")
    print(f"GPU: {torch.cuda.get_device_name()}")

    # Test configuration - realistic but manageable
    config = {
        "vocab_size": 32000,
        "hidden_size": 2048,
        "ffn_hidden_size": 5504,
        "num_layers": 8,  # Reduced for focused comparison
        "num_heads": 16,
        "seq_len": 2048,
    }

    batch_size = 2
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, config["seq_len"]), device=DEVICE)

    print(f"\nConfiguration:")
    print(f"  Model size: {config['num_layers']} layers, {config['hidden_size']} hidden")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {config['seq_len']}")
    print(f"  Compilation warmup: {COMPILATION_WARMUP} iterations")
    print(f"  Steady-state trials: {NUM_TRIALS}")

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):

        # Test models
        models_to_test = [
            ("hybrid_baseline", HybridOptimalLayer, "Hybrid Baseline (No Compilation)"),
            ("hybrid_compiled", HybridCompiledLayer, "Hybrid + Surgical torch.compile"),
            ("te_attention", TEAttentionLayer, "TE MultiheadAttention"),
        ]

        results = {}

        for model_key, layer_class, description in models_to_test:
            try:
                print(f"\n{'='*100}")
                print(f"CREATING: {description}")
                print(f"{'='*100}")

                model = SimpleModel(config, layer_class, description).to(DEVICE)
                print(f"✅ Model created successfully")

                results[model_key] = benchmark_with_compilation_separation(model, input_ids, description)

                # Clean up
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"❌ {description} failed: {e}")
                results[model_key] = None

        # Final Analysis
        print(f"\n{'='*100}")
        print("FINAL STEADY-STATE COMPARISON")
        print(f"{'='*100}")

        valid_results = {k: v for k, v in results.items() if v is not None}

        if len(valid_results) >= 2:
            print(f"\n📊 Steady-State Performance Ranking:")

            # Sort by steady-state performance
            sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['avg_time'])

            for i, (model_key, result) in enumerate(sorted_results, 1):
                model_name = {
                    'hybrid_baseline': 'Hybrid Baseline (No Compilation)',
                    'hybrid_compiled': 'Hybrid + Surgical torch.compile',
                    'te_attention': 'TE MultiheadAttention'
                }[model_key]

                print(f"{i}. {model_name}")
                print(f"   Steady-state: {result['avg_time']:.3f} ms (±{result['std_time']:.2f})")
                print(f"   Compilation: {result['compilation_time']:.1f} ms (one-time)")

            # Head-to-head comparison
            if len(sorted_results) >= 2:
                fastest = sorted_results[0]
                print(f"\n🎯 DETAILED COMPARISON (vs fastest: {fastest[0]})")

                for model_key, result in sorted_results:
                    if model_key != fastest[0]:
                        slowdown = result['avg_time'] / fastest[1]['avg_time']
                        model_name = {
                            'hybrid_baseline': 'Hybrid Baseline',
                            'hybrid_compiled': 'Hybrid + torch.compile',
                            'te_attention': 'TE MultiheadAttention'
                        }[model_key]

                        print(f"  {model_name}: {slowdown:.3f}x slower ({result['avg_time']:.3f} vs {fastest[1]['avg_time']:.3f} ms)")

            # Surgical compilation benefit analysis
            if 'hybrid_baseline' in valid_results and 'hybrid_compiled' in valid_results:
                baseline = valid_results['hybrid_baseline']['avg_time']
                compiled = valid_results['hybrid_compiled']['avg_time']
                speedup = baseline / compiled
                improvement = (speedup - 1) * 100

                print(f"\n⚡ SURGICAL COMPILATION ANALYSIS:")
                print(f"  Baseline hybrid: {baseline:.3f} ms")
                print(f"  Compiled hybrid: {compiled:.3f} ms")
                print(f"  Speedup: {speedup:.3f}x ({improvement:+.1f}%)")

                if speedup > 1.10:
                    print(f"  ✅ SIGNIFICANT surgical compilation benefit!")
                elif speedup > 1.05:
                    print(f"  ✅ MEANINGFUL surgical compilation benefit")
                else:
                    print(f"  ⚖️  MARGINAL surgical compilation benefit")

        else:
            print("❌ Insufficient successful benchmarks for comparison")

        print(f"\n🎯 CONCLUSION:")
        print(f"This benchmark properly separates one-time compilation costs")
        print(f"from steady-state algorithmic performance differences.")

if __name__ == "__main__":
    main()