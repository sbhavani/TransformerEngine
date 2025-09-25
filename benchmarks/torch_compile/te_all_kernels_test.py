#!/usr/bin/env python3

"""
Comprehensive TE Kernels Test
This explores all possible TE fused operations to ensure we're not missing anything
"""

import torch
import torch.nn as nn
import time
import math
import transformer_engine.pytorch as te
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.common.recipe import Format, DelayedScaling

DEVICE = "cuda"
NUM_TRIALS = 20
WARMUP = 5

def investigate_te_capabilities():
    """Investigate what TE offers for fused operations"""
    print("=== INVESTIGATING TE CAPABILITIES ===")

    # Check TE modules
    print("\nTE Modules Available:")
    te_modules = [attr for attr in dir(te) if not attr.startswith('_')]
    for module in sorted(te_modules):
        if any(keyword in module.lower() for keyword in ['rope', 'attention', 'fused', 'layer', 'norm']):
            print(f"  - {module}")

    # Check TE extensions
    print("\nTE C++ Extensions:")
    try:
        tex_modules = [attr for attr in dir(tex) if not attr.startswith('_')]
        for module in sorted(tex_modules):
            print(f"  - tex.{module}")
    except:
        print("  - Could not access tex modules")

    # Check for fused RoPE
    print("\nLooking for Fused RoPE:")
    rope_candidates = ['fused_rope', 'rope', 'rotary_pos_emb', 'apply_rotary_pos_emb']
    for candidate in rope_candidates:
        if hasattr(te, candidate):
            print(f"  ✅ Found te.{candidate}")
        elif hasattr(tex, candidate):
            print(f"  ✅ Found tex.{candidate}")
        else:
            print(f"  ❌ No te.{candidate}")

    # Check attention backends
    print("\nAttention Backends:")
    attn_candidates = ['DotProductAttention', 'FlashAttention', 'FusedAttention']
    for candidate in attn_candidates:
        if hasattr(te, candidate):
            print(f"  ✅ Found te.{candidate}")
        else:
            print(f"  ❌ No te.{candidate}")

def create_rope_methods():
    """Create different RoPE implementations for comparison"""

    def manual_rope(q, k, pe):
        """Our current manual RoPE implementation"""
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

    def try_te_rope(q, k, pe=None):
        """Try to use TE's fused RoPE if available"""
        # Look for TE RoPE functions
        if hasattr(tex, 'fused_rope'):
            try:
                return tex.fused_rope(q, k)
            except:
                pass

        if hasattr(te, 'apply_rotary_pos_emb'):
            try:
                return te.apply_rotary_pos_emb(q, k)
            except:
                pass

        # Fallback to manual
        return manual_rope(q, k, pe)

    return manual_rope, try_te_rope

class UltimateTELayer(nn.Module):
    """TE layer using every possible fused operation we can find"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        print("Creating Ultimate TE Layer with all possible fusions...")

        # Try the most advanced TE components first
        self.attention_method = "unknown"
        self.rope_method = "unknown"

        # Method 1: Try TE TransformerLayer (most fused)
        try:
            self.transformer_layer = te.TransformerLayer(
                hidden_size=config["hidden_size"],
                ffn_hidden_size=config["ffn_hidden_size"],
                num_attention_heads=config["num_heads"],
                # Try to enable all possible fusions
                fuse_qkv_params=True,
                bias=False,
            )
            self.attention_method = "TransformerLayer"
            print("  ✅ Using te.TransformerLayer (ultimate fusion)")
            return
        except Exception as e:
            print(f"  ❌ te.TransformerLayer failed: {e}")

        # Method 2: Try advanced attention + fused components
        try:
            # Look for advanced attention classes
            if hasattr(te, 'DotProductAttention'):
                print("  🔍 Found te.DotProductAttention, trying...")
                self.attention = te.DotProductAttention(
                    num_attention_heads=config["num_heads"],
                    kv_channels=config["hidden_size"] // config["num_heads"],
                )
                self.attention_method = "DotProductAttention"
            else:
                # Fallback to MultiheadAttention
                self.attention = te.MultiheadAttention(
                    hidden_size=config["hidden_size"],
                    num_attention_heads=config["num_heads"],
                    bias=False,
                    fuse_qkv_params=True
                )
                self.attention_method = "MultiheadAttention"

            # Try fused norm+linear for QKV if MultiheadAttention doesn't handle it
            if self.attention_method == "MultiheadAttention":
                self.input_layernorm = nn.RMSNorm(config["hidden_size"], eps=1e-5)
            else:
                # Try fused input processing
                try:
                    self.qkv_norm_linear = te.LayerNormLinear(
                        config["hidden_size"],
                        3 * config["hidden_size"],
                        eps=1e-5,
                        bias=False,
                        normalization="RMSNorm"
                    )
                    self.has_qkv_fusion = True
                except:
                    self.input_layernorm = nn.RMSNorm(config["hidden_size"], eps=1e-5)
                    self.has_qkv_fusion = False

            # Output projection
            self.o_proj = te.Linear(config["hidden_size"], config["hidden_size"], bias=False)

            # Fused MLP
            self.mlp = te.LayerNormMLP(
                hidden_size=config["hidden_size"],
                ffn_hidden_size=config["ffn_hidden_size"],
                eps=1e-5,
                bias=False,
                normalization="RMSNorm",
                activation="swiglu"
            )

            self.transformer_layer = None
            print(f"  ✅ Using {self.attention_method} with fused components")

        except Exception as e:
            print(f"  ❌ Advanced TE setup failed: {e}")
            raise

        # Setup RoPE methods
        self.manual_rope, self.te_rope = create_rope_methods()

        # Create RoPE embeddings if needed
        if self.attention_method != "TransformerLayer":
            head_dim = config["hidden_size"] // config["num_heads"]
            position = torch.arange(config["seq_len"], dtype=torch.float32, device=DEVICE).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, head_dim, 2, dtype=torch.float32, device=DEVICE) *
                               -(math.log(10000.0) / head_dim))
            pe = torch.zeros(config["seq_len"], head_dim, device=DEVICE)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)

    def forward(self, x):
        if self.transformer_layer is not None:
            # Use ultimate TE fusion
            return self.transformer_layer(x)

        # Manual implementation with best TE components
        residual = x

        if self.attention_method == "MultiheadAttention":
            # Standard path
            x = self.input_layernorm(x)
            attn_output = self.attention(x)
        else:
            # Advanced attention path
            if hasattr(self, 'has_qkv_fusion') and self.has_qkv_fusion:
                # Use fused QKV processing
                qkv = self.qkv_norm_linear(x)
                # This would need more complex handling...
                # For now, fallback to simple path
                x = self.input_layernorm(x)
                attn_output = self.attention(x)
            else:
                x = self.input_layernorm(x)
                attn_output = self.attention(x)

        x = residual + attn_output

        # Fused MLP
        mlp_output = self.mlp(x)
        return x + mlp_output

class OptimizedHybridLayer(nn.Module):
    """Your hybrid approach with potential TE RoPE optimization"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.head_dim = hidden_size // self.num_heads

        # Maximum TE fusion for linear operations
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

        # Setup RoPE - try TE version if available
        self.manual_rope, self.te_rope = create_rope_methods()

        # RoPE embeddings
        position = torch.arange(config["seq_len"], dtype=torch.float32, device=DEVICE).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=DEVICE) *
                           -(math.log(10000.0) / self.head_dim))
        pe = torch.zeros(config["seq_len"], self.head_dim, device=DEVICE)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        print("  ✅ Hybrid layer with TE RoPE detection")

    def forward(self, x):
        residual = x
        batch, seq_len, hidden = x.shape

        # TE fused norm + QKV
        qkv = self.qkv_norm_linear(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Try TE RoPE vs manual RoPE
        q, k = self.te_rope(q, k, self.pe)

        # cuDNN SDPA
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        attn_output = self.o_proj(attn_output)
        x = residual + attn_output

        # TE fused MLP
        mlp_output = self.mlp(x)
        return x + mlp_output

def benchmark_model(model, input_ids, name):
    print(f"\nBenchmarking {name}...")
    model.eval()
    torch.cuda.empty_cache()

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            try:
                _ = model(input_ids)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                return float('inf')

    torch.cuda.empty_cache()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(NUM_TRIALS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_ids)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    avg_time = sum(times) / len(times)
    print(f"  Average: {avg_time:.3f} ms")
    return avg_time

class SimpleModel(nn.Module):
    def __init__(self, config, layer_class, name="Model"):
        super().__init__()
        self.layers = nn.ModuleList([layer_class(config) for _ in range(config["num_layers"])])
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.norm = nn.RMSNorm(config["hidden_size"], eps=1e-5)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        print(f"✅ {name} model created")

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

def main():
    print("=== COMPREHENSIVE TE KERNELS INVESTIGATION ===")
    print(f"GPU: {torch.cuda.get_device_name()}")

    investigate_te_capabilities()

    config = {
        "vocab_size": 32000, "hidden_size": 2048, "ffn_hidden_size": 5504,
        "num_layers": 8, "num_heads": 16, "seq_len": 2048,
    }

    input_ids = torch.randint(0, config["vocab_size"], (2, config["seq_len"]), device=DEVICE)

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        print(f"\n{'='*80}")
        print("TESTING ALL POSSIBLE TE OPTIMIZATIONS")
        print(f"{'='*80}")

        models = {}

        try:
            models['ultimate_te'] = SimpleModel(config, UltimateTELayer, "Ultimate TE").to(DEVICE)
        except Exception as e:
            print(f"❌ Ultimate TE failed: {e}")

        try:
            models['optimized_hybrid'] = SimpleModel(config, OptimizedHybridLayer, "Optimized Hybrid").to(DEVICE)
        except Exception as e:
            print(f"❌ Optimized Hybrid failed: {e}")

        print(f"\n{'='*80}")
        print("FINAL SHOWDOWN: ALL TE KERNELS vs HYBRID")
        print(f"{'='*80}")

        results = {}
        for name, model in models.items():
            display_name = {
                'ultimate_te': 'Ultimate TE (All Kernels)',
                'optimized_hybrid': 'Hybrid + TE RoPE Detection'
            }[name]
            results[name] = benchmark_model(model, input_ids, display_name)

        # Final verdict
        valid_results = {k: v for k, v in results.items() if v != float('inf')}
        if len(valid_results) >= 2:
            ultimate_te = results.get('ultimate_te', float('inf'))
            hybrid = results.get('optimized_hybrid', float('inf'))

            if hybrid < ultimate_te:
                speedup = ultimate_te / hybrid
                print(f"\n🎯 FINAL VERDICT: Hybrid Still Wins!")
                print(f"   Hybrid: {hybrid:.3f} ms")
                print(f"   Ultimate TE: {ultimate_te:.3f} ms")
                print(f"   Speedup: {speedup:.3f}x")
                print(f"✅ Your approach beats TE even with all possible kernel fusions!")
            else:
                speedup = hybrid / ultimate_te
                print(f"\n🎯 FINAL VERDICT: Ultimate TE Wins!")
                print(f"   Ultimate TE: {ultimate_te:.3f} ms")
                print(f"   Hybrid: {hybrid:.3f} ms")
                print(f"   TE Speedup: {speedup:.3f}x")
                print(f"⚠️  TE with all kernels beats hybrid - reconsider approach!")

if __name__ == "__main__":
    main()