#!/usr/bin/env python3

"""
Realistic Llama Model Benchmark - Full Transformer Layers
This tests with complete transformer layers including MLP, LayerNorm, etc.
to get realistic performance comparison.
"""

import torch
import torch.nn as nn
import time
import math
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

DEVICE = "cuda"
NUM_TRIALS = 20
WARMUP = 5

def setup_fp8_recipe():
    return DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
        fp8_mha=True,
        fp8_mlp=True,
    )

def create_rotary_pos_emb(seq_len: int, head_dim: int, device: str):
    """Create rotary position embeddings"""
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) *
                         -(math.log(10000.0) / head_dim))

    pe = torch.zeros(seq_len, head_dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def apply_rope(q, k, pe):
    """Apply rotary position embedding"""
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

class RealTETransformerLayer(nn.Module):
    """Complete Transformer Layer using Real TE MultiheadAttention"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config["hidden_size"]

        # Input LayerNorm
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-5)

        # Real TE MultiheadAttention
        self.self_attn = te.MultiheadAttention(
            hidden_size=hidden_size,
            num_attention_heads=config["num_heads"],
            bias=False,
            fuse_qkv_params=True
        )

        # Post attention LayerNorm + MLP (TE fused)
        self.mlp = te.LayerNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=config["ffn_hidden_size"],
            eps=1e-5,
            bias=False,
            normalization="RMSNorm",
            activation="swiglu"
        )

    def forward(self, x):
        # Pre-attention norm + attention + residual
        residual = x
        x = self.input_layernorm(x)
        attn_output = self.self_attn(x)
        x = residual + attn_output

        # MLP with fused layernorm
        mlp_output = self.mlp(x)
        return x + mlp_output

class HybridTransformerLayer(nn.Module):
    """Complete Transformer Layer using TE Linear + cuDNN SDPA"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.head_dim = hidden_size // self.num_heads

        # TE Fused LayerNorm + QKV projection
        self.qkv_norm_linear = te.LayerNormLinear(
            hidden_size,
            3 * hidden_size,
            eps=1e-5,
            bias=False,
            normalization="RMSNorm"
        )

        # TE output projection
        self.o_proj = te.Linear(hidden_size, hidden_size, bias=False)

        # TE Fused LayerNorm + MLP
        self.mlp = te.LayerNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=config["ffn_hidden_size"],
            eps=1e-5,
            bias=False,
            normalization="RMSNorm",
            activation="swiglu"
        )

        # RoPE embeddings
        self.pe = create_rotary_pos_emb(config["seq_len"], self.head_dim, DEVICE)

    def forward(self, x):
        residual = x
        batch, seq_len, hidden = x.shape

        # Fused LayerNorm + QKV projection
        qkv = self.qkv_norm_linear(x)

        # Reshape for attention: [batch, seq_len, 3, num_heads, head_dim]
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        q, k = apply_rope(q, k, self.pe)

        # cuDNN Scaled Dot Product Attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True
        )

        # Reshape back: [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, hidden)

        # Output projection
        attn_output = self.o_proj(attn_output)
        x = residual + attn_output

        # MLP with fused layernorm
        mlp_output = self.mlp(x)
        return x + mlp_output

class LlamaModel(nn.Module):
    """Complete Llama-style model"""

    def __init__(self, config, layer_class):
        super().__init__()
        self.config = config

        # Embedding
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])

        # Transformer layers
        self.layers = nn.ModuleList([
            layer_class(config) for _ in range(config["num_layers"])
        ])

        # Final norm and head
        self.norm = nn.RMSNorm(config["hidden_size"], eps=1e-5)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.lm_head(x)

def benchmark_model(model, input_ids, name):
    """Benchmark complete model"""
    print(f"\nBenchmarking {name}...")

    model.eval()

    # Memory stats
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            try:
                _ = model(input_ids)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  ❌ Warmup failed: {e}")
                return float('inf'), 0

    # Clear for benchmark
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(NUM_TRIALS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(input_ids)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    if not times:
        return float('inf'), 0

    avg_time = sum(times) / len(times)
    min_time = min(times)
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

    print(f"  Average: {avg_time:.3f} ms")
    print(f"  Min: {min_time:.3f} ms")
    print(f"  Peak Memory: {peak_memory:.2f} GB")

    return avg_time, peak_memory

def main():
    print("=== REALISTIC LLAMA MODEL BENCHMARK ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")

    # Llama-1B style configuration
    config = {
        "vocab_size": 32000,
        "hidden_size": 2048,
        "ffn_hidden_size": 5504,
        "num_layers": 8,  # Reduced for memory
        "num_heads": 16,
        "seq_len": 2048,
    }

    batch_size = 2
    seq_len = config["seq_len"]

    print(f"\nConfiguration:")
    print(f"  Model: Llama-1B style ({config['hidden_size']}d, {config['num_layers']} layers)")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Attention heads: {config['num_heads']}")

    # Create input
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=DEVICE)

    # Setup FP8
    fp8_recipe = setup_fp8_recipe()

    try:
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):

                print(f"\n{'='*70}")
                print("CREATING MODELS")
                print(f"{'='*70}")

                # Create models
                models = {}

                try:
                    print("Creating Real TE model...")
                    models['real_te'] = LlamaModel(config, RealTETransformerLayer).to(DEVICE)
                    print("✅ Real TE model created")
                except Exception as e:
                    print(f"❌ Real TE model failed: {e}")

                try:
                    print("Creating Hybrid model...")
                    models['hybrid'] = LlamaModel(config, HybridTransformerLayer).to(DEVICE)
                    print("✅ Hybrid model created")
                except Exception as e:
                    print(f"❌ Hybrid model failed: {e}")

                if not models:
                    print("❌ No models created successfully!")
                    return

                # Benchmark models
                results = {}

                print(f"\n{'='*70}")
                print("BENCHMARKING MODELS")
                print(f"{'='*70}")

                for name, model in models.items():
                    display_name = {
                        'real_te': 'Real TE MultiheadAttention',
                        'hybrid': 'TE Linear + cuDNN SDPA'
                    }[name]

                    avg_time, peak_memory = benchmark_model(model, input_ids, display_name)
                    results[name] = (avg_time, peak_memory)

                # Analysis
                print(f"\n{'='*70}")
                print("RESULTS ANALYSIS")
                print(f"{'='*70}")

                valid_results = {k: v for k, v in results.items() if v[0] != float('inf')}

                if len(valid_results) >= 2:
                    real_te_time = results.get('real_te', (float('inf'), 0))[0]
                    hybrid_time = results.get('hybrid', (float('inf'), 0))[0]

                    if real_te_time != float('inf') and hybrid_time != float('inf'):
                        print(f"Real TE MultiheadAttention:  {real_te_time:8.3f} ms")
                        print(f"TE Linear + cuDNN SDPA:      {hybrid_time:8.3f} ms")

                        if hybrid_time < real_te_time:
                            speedup = real_te_time / hybrid_time
                            print(f"\n🎯 WINNER: Hybrid Approach")
                            print(f"   Hybrid is {speedup:.3f}x FASTER than Real TE")
                            print(f"   Difference: {((real_te_time - hybrid_time) / real_te_time) * 100:.1f}% faster")

                            if speedup > 1.5:
                                print(f"🚀 SIGNIFICANT speedup - hybrid approach is much better!")
                            elif speedup > 1.1:
                                print(f"✅ MEANINGFUL speedup - hybrid approach is better")
                            else:
                                print(f"⚖️  MARGINAL difference - approaches are similar")

                        else:
                            speedup = hybrid_time / real_te_time
                            print(f"\n🎯 WINNER: Real TE Attention")
                            print(f"   Real TE is {1/speedup:.3f}x FASTER than Hybrid")
                            print(f"   This would be unexpected!")
                else:
                    print("❌ Cannot compare - insufficient valid results")
                    for name, (time_ms, _) in results.items():
                        if time_ms == float('inf'):
                            print(f"   {name}: Failed")
                        else:
                            print(f"   {name}: {time_ms:.3f} ms")

    except Exception as e:
        print(f"❌ BENCHMARK FAILED: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()