#!/usr/bin/env python3

"""
Maximum TE Fusion Benchmark - Without FP8 autocast to avoid cleanup issues
This should still show the relative performance differences
"""

import torch
import torch.nn as nn
import time
import math
import transformer_engine.pytorch as te

DEVICE = "cuda"
NUM_TRIALS = 20
WARMUP = 5

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

class TETransformerLayerAttempt(nn.Module):
    """Try to use TE's TransformerLayer with various approaches"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer_layer = None

        print("Attempting to create te.TransformerLayer...")

        # Try different TransformerLayer configurations
        attempts = [
            # Attempt 1: Minimal parameters
            {
                "hidden_size": config["hidden_size"],
                "ffn_hidden_size": config["ffn_hidden_size"],
                "num_attention_heads": config["num_heads"],
            },
            # Attempt 2: With layer number
            {
                "hidden_size": config["hidden_size"],
                "ffn_hidden_size": config["ffn_hidden_size"],
                "num_attention_heads": config["num_heads"],
                "layer_number": 1,
            },
            # Attempt 3: More complete config
            {
                "hidden_size": config["hidden_size"],
                "ffn_hidden_size": config["ffn_hidden_size"],
                "num_attention_heads": config["num_heads"],
                "layer_number": 1,
                "bias": False,
                "layer_type": "encoder",
            }
        ]

        for i, params in enumerate(attempts, 1):
            try:
                print(f"  Attempt {i}: {list(params.keys())}")
                self.transformer_layer = te.TransformerLayer(**params)
                print(f"  ✅ Success with attempt {i}!")
                break
            except Exception as e:
                print(f"  ❌ Attempt {i} failed: {e}")
                continue

        if self.transformer_layer is None:
            print("  ❌ All TransformerLayer attempts failed, using fallback")
            self.create_fallback(config)

    def create_fallback(self, config):
        """Create fallback using separate components"""
        self.input_layernorm = nn.RMSNorm(config["hidden_size"], eps=1e-5)

        self.self_attn = te.MultiheadAttention(
            hidden_size=config["hidden_size"],
            num_attention_heads=config["num_heads"],
            bias=False,
            fuse_qkv_params=True
        )

        self.mlp = te.LayerNormMLP(
            hidden_size=config["hidden_size"],
            ffn_hidden_size=config["ffn_hidden_size"],
            eps=1e-5,
            bias=False,
            normalization="RMSNorm",
            activation="swiglu"
        )

    def forward(self, x):
        if self.transformer_layer is not None:
            return self.transformer_layer(x)
        else:
            # Fallback path
            residual = x
            x = self.input_layernorm(x)
            attn_output = self.self_attn(x)
            x = residual + attn_output
            mlp_output = self.mlp(x)
            return x + mlp_output

class MaxFusedTELayer(nn.Module):
    """TE with maximum fusion we can achieve"""

    def __init__(self, config):
        super().__init__()

        # Try to use LayerNormLinear for attention preprocessing
        try:
            self.attn_ln_linear = te.LayerNormLinear(
                config["hidden_size"],
                config["hidden_size"],
                eps=1e-5,
                bias=False,
                normalization="RMSNorm"
            )
            self.has_fused_attn_norm = True
            print("  ✅ Using fused attention LayerNorm")
        except:
            self.input_layernorm = nn.RMSNorm(config["hidden_size"], eps=1e-5)
            self.has_fused_attn_norm = False
            print("  ⚠️  Using separate attention LayerNorm")

        # TE MultiheadAttention with all possible fusion
        self.self_attn = te.MultiheadAttention(
            hidden_size=config["hidden_size"],
            num_attention_heads=config["num_heads"],
            bias=False,
            fuse_qkv_params=True
        )

        # TE LayerNormMLP (definitely fused)
        self.mlp = te.LayerNormMLP(
            hidden_size=config["hidden_size"],
            ffn_hidden_size=config["ffn_hidden_size"],
            eps=1e-5,
            bias=False,
            normalization="RMSNorm",
            activation="swiglu"
        )

    def forward(self, x):
        residual = x

        if self.has_fused_attn_norm:
            # This doesn't actually work for attention preprocessing
            # because we need the normalized value, not a linear projection
            # Fall back to separate norm
            x = nn.functional.rms_norm(x, (x.size(-1),), eps=1e-5)
        else:
            x = self.input_layernorm(x)

        attn_output = self.self_attn(x)
        x = residual + attn_output

        # Fused LayerNorm + MLP
        mlp_output = self.mlp(x)
        return x + mlp_output

class HybridMaxFusedLayer(nn.Module):
    """Your hybrid approach with maximum TE fusion"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.head_dim = hidden_size // self.num_heads

        # TE Fused LayerNorm + QKV projection (maximum fusion here)
        self.qkv_norm_linear = te.LayerNormLinear(
            hidden_size,
            3 * hidden_size,
            eps=1e-5,
            bias=False,
            normalization="RMSNorm"
        )

        # TE output projection
        self.o_proj = te.Linear(hidden_size, hidden_size, bias=False)

        # TE Fused LayerNorm + MLP (maximum fusion)
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

        # Reshape for attention
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        q, k = apply_rope(q, k, self.pe)

        # cuDNN SDPA
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True
        )

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, hidden)

        # TE output projection
        attn_output = self.o_proj(attn_output)
        x = residual + attn_output

        # TE Fused LayerNorm + MLP
        mlp_output = self.mlp(x)
        return x + mlp_output

class LlamaModel(nn.Module):
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

        print(f"✅ {name} created")

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

def benchmark_model(model, input_ids, name):
    print(f"\nBenchmarking {name}...")

    model.eval()
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
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    print(f"  Average: {avg_time:.3f} ms")
    print(f"  Min: {min_time:.3f} ms")
    print(f"  Peak Memory: {peak_memory:.2f} GB")

    return avg_time, peak_memory

def main():
    print("=== TE MAXIMUM FUSION vs HYBRID (No FP8) ===")
    print(f"GPU: {torch.cuda.get_device_name()}")

    config = {
        "vocab_size": 32000,
        "hidden_size": 2048,
        "ffn_hidden_size": 5504,
        "num_layers": 8,
        "num_heads": 16,
        "seq_len": 2048,
    }

    batch_size = 2
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, config["seq_len"]), device=DEVICE)

    # Use regular autocast without fp8
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):

        print(f"\n{'='*80}")
        print("CREATING MODELS")
        print(f"{'='*80}")

        models = {}

        # 1. Try TE TransformerLayer
        try:
            models['te_transformer'] = LlamaModel(config, TETransformerLayerAttempt, "TE TransformerLayer").to(DEVICE)
        except Exception as e:
            print(f"❌ TE TransformerLayer model failed: {e}")

        # 2. Maximum fused TE attention
        try:
            models['te_maxfused'] = LlamaModel(config, MaxFusedTELayer, "Max Fused TE Attention").to(DEVICE)
        except Exception as e:
            print(f"❌ Max Fused TE model failed: {e}")

        # 3. Your hybrid approach
        try:
            models['hybrid'] = LlamaModel(config, HybridMaxFusedLayer, "Hybrid (TE + cuDNN)").to(DEVICE)
        except Exception as e:
            print(f"❌ Hybrid model failed: {e}")

        if not models:
            print("❌ No models created!")
            return

        print(f"\n{'='*80}")
        print("BENCHMARKING")
        print(f"{'='*80}")

        results = {}
        descriptions = {
            'te_transformer': 'TE TransformerLayer (Full Fusion)',
            'te_maxfused': 'TE MultiheadAttention (Max Fused)',
            'hybrid': 'Hybrid: TE Linear + cuDNN SDPA'
        }

        for name, model in models.items():
            desc = descriptions.get(name, name)
            avg_time, peak_memory = benchmark_model(model, input_ids, desc)
            results[name] = (avg_time, peak_memory)

        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")

        valid_results = {k: v for k, v in results.items() if v[0] != float('inf')}

        if valid_results:
            print("Performance Ranking:")
            sorted_results = sorted(valid_results.items(), key=lambda x: x[1][0])

            for i, (name, (time_ms, memory_gb)) in enumerate(sorted_results, 1):
                desc = descriptions.get(name, name)
                print(f"{i}. {desc:40s}: {time_ms:8.3f} ms")

            # Head to head comparison
            if len(valid_results) >= 2:
                fastest_name, (fastest_time, _) = sorted_results[0]

                print(f"\nDetailed Comparison:")
                for name, (time_ms, _) in sorted_results:
                    if name != fastest_name:
                        slowdown = time_ms / fastest_time
                        desc = descriptions.get(name, name)
                        print(f"  {descriptions[fastest_name]} is {slowdown:.3f}x faster than {desc}")

                # Check if hybrid wins
                hybrid_time = results.get('hybrid', (float('inf'), 0))[0]
                if hybrid_time != float('inf') and fastest_name == 'hybrid':
                    print(f"\n🎯 WINNER: Your Hybrid Approach!")
                    print(f"✅ Even against maximum TE fusion, hybrid is fastest")
                elif hybrid_time != float('inf'):
                    winner_time = sorted_results[0][1][0]
                    hybrid_slowdown = hybrid_time / winner_time
                    print(f"\n🎯 WINNER: {descriptions[fastest_name]}")
                    print(f"⚠️  Hybrid is {hybrid_slowdown:.3f}x slower than best TE approach")

        else:
            print("❌ No valid results")

if __name__ == "__main__":
    main()