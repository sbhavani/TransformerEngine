#!/usr/bin/env python3

"""
Minimal Working TE Attention Benchmark
Simplified to just get TE attention working first
"""

import torch
import torch.nn as nn
import time
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

DEVICE = "cuda"
NUM_TRIALS = 10
WARMUP = 3

def setup_fp8_recipe():
    return DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
    )

class SimpleRealTELayer(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16):
        super().__init__()

        # Try minimal TE attention parameters
        self.attention = te.MultiheadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads
        )

    def forward(self, x):
        return self.attention(x)

class SimpleHybridLayer(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # TE linear for QKV
        self.qkv_proj = te.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = te.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch, seq_len, hidden = x.shape

        # QKV projection with TE
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # cuDNN attention
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, hidden)

        # TE output projection
        return self.out_proj(attn_out)

def benchmark_layer(layer, x, name):
    print(f"\nBenchmarking {name}...")

    layer.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            try:
                _ = layer(x)
            except Exception as e:
                print(f"  ❌ Error in {name}: {e}")
                return float('inf')

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(NUM_TRIALS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = layer(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    avg_time = sum(times) / len(times)
    print(f"  Average: {avg_time:.3f} ms")
    return avg_time

def main():
    print("=== Minimal TE Attention Test ===")
    print(f"GPU: {torch.cuda.get_device_name()}")

    # Small test case
    batch_size = 4
    seq_len = 512
    hidden_size = 1024
    num_heads = 16

    # Input tensor
    x = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE, dtype=torch.bfloat16)

    fp8_recipe = setup_fp8_recipe()

    try:
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):

            # Test Real TE Attention
            print("\n--- Testing Real TE Attention ---")
            try:
                real_te_layer = SimpleRealTELayer(hidden_size, num_heads).to(DEVICE)
                real_te_time = benchmark_layer(real_te_layer, x, "Real TE Attention")
                print(f"✅ Real TE Attention: {real_te_time:.3f} ms")
            except Exception as e:
                print(f"❌ Real TE Attention failed: {e}")
                real_te_time = float('inf')

            # Test Hybrid Approach
            print("\n--- Testing Hybrid Approach ---")
            try:
                hybrid_layer = SimpleHybridLayer(hidden_size, num_heads).to(DEVICE)
                hybrid_time = benchmark_layer(hybrid_layer, x, "TE Linear + cuDNN SDPA")
                print(f"✅ Hybrid Approach: {hybrid_time:.3f} ms")
            except Exception as e:
                print(f"❌ Hybrid Approach failed: {e}")
                hybrid_time = float('inf')

            # Comparison
            print(f"\n=== RESULTS ===")
            if real_te_time != float('inf') and hybrid_time != float('inf'):
                speedup = hybrid_time / real_te_time
                if speedup > 1.05:
                    print(f"🎯 Real TE is {speedup:.3f}x FASTER than hybrid approach")
                    print("✅ CONFIRMED: TE native attention beats compilation")
                elif speedup < 0.95:
                    print(f"📈 Hybrid is {1/speedup:.3f}x FASTER than Real TE")
                    print("⚠️  Unexpected: Hybrid beats native TE")
                else:
                    print(f"⚖️  Similar performance: {speedup:.3f}x")
            else:
                print("❌ Could not compare - one or both approaches failed")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()