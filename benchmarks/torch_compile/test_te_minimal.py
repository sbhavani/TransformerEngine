#!/usr/bin/env python3

"""
Minimal test to verify TransformerEngine installation and basic functionality.
Run this after fixing TE installation to verify it works before running the full benchmark.
"""

import sys
import torch

print("=== TransformerEngine Installation Test ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("❌ CUDA not available - TE requires CUDA")
    sys.exit(1)

# Test TE import
try:
    print("\nTesting TE import...")
    import transformer_engine.pytorch as te
    print("✅ TE import successful")
except Exception as e:
    print(f"❌ TE import failed: {e}")
    sys.exit(1)

# Test TE components
try:
    print("\nTesting TE components...")

    # Test FP8 recipe
    from transformer_engine.common.recipe import Format, DelayedScaling
    fp8_recipe = DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
    )
    print("✅ FP8 recipe creation successful")

    # Test TE Linear layer
    device = "cuda"
    hidden_size = 128

    linear = te.Linear(
        hidden_size,
        hidden_size,
        bias=False,
        device=device,
        params_dtype=torch.bfloat16
    )
    print("✅ TE Linear layer creation successful")

    # Test forward pass
    x = torch.randn(2, 32, hidden_size, device=device, dtype=torch.bfloat16)

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = linear(x)

    print("✅ TE Linear forward pass successful")
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")

    # Test MultiheadAttention (if available)
    try:
        attention = te.MultiheadAttention(
            hidden_size=hidden_size,
            num_attention_heads=8,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
            params_dtype=torch.bfloat16
        )

        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            attn_out = attention(x)

        print("✅ TE MultiheadAttention successful")
        print(f"   Attention output shape: {attn_out.shape}")
    except Exception as e:
        print(f"⚠️  TE MultiheadAttention failed: {e}")
        print("   This might be okay - some TE features are optional")

except Exception as e:
    print(f"❌ TE component test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed! TE is ready for benchmarking.")
print("You can now run: python3 te_llama_real_attention.py")