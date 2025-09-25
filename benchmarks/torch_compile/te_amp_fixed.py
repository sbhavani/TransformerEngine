#!/usr/bin/env python3

"""
TE Attention Benchmark - Using torch.amp + fp8_autocast
This should fix dtype compatibility issues
"""

import torch
import torch.nn as nn
import time
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

class SimpleRealTELayer(nn.Module):
    def __init__(self, hidden_size=2048, num_heads=16):
        super().__init__()

        # Real TE MultiheadAttention - minimal parameters
        self.attention = te.MultiheadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            bias=False,
            fuse_qkv_params=True
        )

    def forward(self, x):
        return self.attention(x)

class SimpleHybridLayer(nn.Module):
    def __init__(self, hidden_size=2048, num_heads=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # TE linear layers
        self.qkv_proj = te.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = te.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        batch, seq_len, hidden = x.shape

        # QKV projection with TE
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # cuDNN SDPA (the "hybrid" approach)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=True  # Add causal masking for realism
        )

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, hidden)

        # TE output projection
        return self.out_proj(attn_out)

def benchmark_layer(layer, x, name, use_amp=True):
    print(f"\nBenchmarking {name}...")

    layer.eval()

    # Setup autocast
    autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else torch.no_grad()

    # Warmup
    with torch.no_grad():
        with autocast_ctx:
            for _ in range(WARMUP):
                try:
                    _ = layer(x)
                    torch.cuda.synchronize()
                except Exception as e:
                    print(f"  ❌ Error in {name}: {e}")
                    return float('inf')

    # Clear memory
    torch.cuda.empty_cache()

    # Benchmark
    times = []
    with torch.no_grad():
        with autocast_ctx:
            for _ in range(NUM_TRIALS):
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = layer(x)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

    if not times:
        return float('inf')

    avg_time = sum(times) / len(times)
    min_time = min(times)
    print(f"  Average: {avg_time:.3f} ms")
    print(f"  Min: {min_time:.3f} ms")
    return avg_time

def main():
    print("=== TE Attention Test - torch.amp + fp8_autocast ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")

    # Test configuration - realistic size
    batch_size = 2
    seq_len = 2048
    hidden_size = 2048
    num_heads = 16

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num heads: {num_heads}")

    # Input tensor - float32 initially, let autocast handle conversion
    x = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE)

    fp8_recipe = setup_fp8_recipe()

    try:
        # Use both fp8_autocast and torch.amp together
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):

            # Test Real TE Attention
            print(f"\n{'='*60}")
            print("TESTING: Real TE MultiheadAttention (Native Kernels)")
            print(f"{'='*60}")

            try:
                real_te_layer = SimpleRealTELayer(hidden_size, num_heads).to(DEVICE)
                real_te_time = benchmark_layer(real_te_layer, x, "Real TE Attention")
            except Exception as e:
                print(f"❌ Real TE Attention creation/benchmark failed: {e}")
                real_te_time = float('inf')

            # Test Hybrid Approach
            print(f"\n{'='*60}")
            print("TESTING: TE Linear + cuDNN SDPA (Hybrid Approach)")
            print(f"{'='*60}")

            try:
                hybrid_layer = SimpleHybridLayer(hidden_size, num_heads).to(DEVICE)
                hybrid_time = benchmark_layer(hybrid_layer, x, "TE Linear + cuDNN SDPA")
            except Exception as e:
                print(f"❌ Hybrid Approach creation/benchmark failed: {e}")
                hybrid_time = float('inf')

            # Results Analysis
            print(f"\n{'='*60}")
            print("PERFORMANCE COMPARISON")
            print(f"{'='*60}")

            if real_te_time != float('inf') and hybrid_time != float('inf'):
                print(f"Real TE MultiheadAttention:  {real_te_time:7.3f} ms")
                print(f"TE Linear + cuDNN SDPA:      {hybrid_time:7.3f} ms")

                te_vs_hybrid = real_te_time / hybrid_time
                if te_vs_hybrid < 0.95:  # TE is faster
                    speedup = hybrid_time / real_te_time
                    print(f"\n🎯 WINNER: Real TE Attention")
                    print(f"   TE is {speedup:.3f}x FASTER than hybrid approach")
                    print(f"✅ CONFIRMED: TE native kernels outperform cuDNN SDPA")
                    print(f"   This validates your hypothesis!")

                elif te_vs_hybrid > 1.05:  # Hybrid is faster
                    speedup = real_te_time / hybrid_time
                    print(f"\n📈 WINNER: Hybrid Approach")
                    print(f"   Hybrid is {1/te_vs_hybrid:.3f}x FASTER than TE native")
                    print(f"⚠️  Unexpected result - hybrid beats native TE")

                else:  # Similar performance
                    print(f"\n⚖️  SIMILAR PERFORMANCE")
                    print(f"   Ratio: {te_vs_hybrid:.3f}x (within 5% difference)")

                print(f"\nPerformance difference: {abs(te_vs_hybrid - 1) * 100:.1f}%")

            elif real_te_time != float('inf'):
                print(f"✅ Real TE Attention: {real_te_time:.3f} ms")
                print(f"❌ Hybrid approach failed - cannot compare")

            elif hybrid_time != float('inf'):
                print(f"❌ Real TE Attention failed")
                print(f"✅ Hybrid approach: {hybrid_time:.3f} ms")
                print(f"   (Cannot validate hypothesis without TE baseline)")

            else:
                print("❌ Both approaches failed!")
                print("   Check TE installation and GPU compatibility")

    except Exception as e:
        print(f"\n❌ BENCHMARK FAILED: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()