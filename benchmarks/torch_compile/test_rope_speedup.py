#!/usr/bin/env python3

"""
Focused test on RoPE (Rotary Position Embedding) operations - the most promising
candidate for torch.compile speedup in TransformerEngine.
"""

import time
import torch
import torch.nn.functional as F
import math

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
SEQ_LEN = 2048
NUM_HEADS = 32
HEAD_DIM = 128
NUM_TRIALS = 100
WARMUP = 20

def create_rope_embeddings(seq_len: int, head_dim: int):
    """Create RoPE embeddings"""
    position = torch.arange(seq_len, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, head_dim, 2, dtype=torch.float32, device=DEVICE) *
                         -(math.log(10000.0) / head_dim))

    pe = torch.zeros(seq_len, head_dim, device=DEVICE)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

def apply_rope_eager(q: torch.Tensor, k: torch.Tensor, pe: torch.Tensor):
    """Apply RoPE in eager mode (current TE pattern)"""
    seq_len = q.size(2)

    # Reshape for RoPE application
    q_rope = q.view(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM // 2, 2)
    k_rope = k.view(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM // 2, 2)

    # Split PE into cos and sin
    cos = pe[:seq_len, 0::2].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
    sin = pe[:seq_len, 1::2].unsqueeze(0).unsqueeze(0)

    # Apply rotation
    q_cos = q_rope[..., 0] * cos - q_rope[..., 1] * sin
    q_sin = q_rope[..., 0] * sin + q_rope[..., 1] * cos

    k_cos = k_rope[..., 0] * cos - k_rope[..., 1] * sin
    k_sin = k_rope[..., 0] * sin + k_rope[..., 1] * cos

    # Recombine
    q_rotated = torch.stack([q_cos, q_sin], dim=-1).view(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM)
    k_rotated = torch.stack([k_cos, k_sin], dim=-1).view(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM)

    return q_rotated, k_rotated

@torch.compile(mode="max-autotune")
def apply_rope_compiled(q: torch.Tensor, k: torch.Tensor, pe: torch.Tensor):
    """Apply RoPE with torch.compile"""
    seq_len = q.size(2)

    # Reshape for RoPE application
    q_rope = q.view(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM // 2, 2)
    k_rope = k.view(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM // 2, 2)

    # Split PE into cos and sin
    cos = pe[:seq_len, 0::2].unsqueeze(0).unsqueeze(0)
    sin = pe[:seq_len, 1::2].unsqueeze(0).unsqueeze(0)

    # Apply rotation
    q_cos = q_rope[..., 0] * cos - q_rope[..., 1] * sin
    q_sin = q_rope[..., 0] * sin + q_rope[..., 1] * cos

    k_cos = k_rope[..., 0] * cos - k_rope[..., 1] * sin
    k_sin = k_rope[..., 0] * sin + k_rope[..., 1] * cos

    # Recombine
    q_rotated = torch.stack([q_cos, q_sin], dim=-1).view(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM)
    k_rotated = torch.stack([k_cos, k_sin], dim=-1).view(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM)

    return q_rotated, k_rotated

def benchmark_rope():
    """Benchmark RoPE operations"""
    print("RoPE (Rotary Position Embedding) Benchmark")
    print("="*50)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Number of heads: {NUM_HEADS}")
    print(f"Head dimension: {HEAD_DIM}")

    # Setup test data
    q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE)
    k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE)
    pe = create_rope_embeddings(SEQ_LEN, HEAD_DIM)

    # Warmup both functions
    print("\nWarming up...")
    for _ in range(WARMUP):
        _ = apply_rope_eager(q, k, pe)
        _ = apply_rope_compiled(q, k, pe)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # Benchmark eager mode
    print("Benchmarking eager mode...")
    eager_times = []
    for _ in range(NUM_TRIALS):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        q_eager, k_eager = apply_rope_eager(q, k, pe)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        eager_times.append((end - start) * 1000)

    eager_avg = sum(eager_times) / len(eager_times)
    eager_min = min(eager_times)

    # Benchmark compiled mode
    print("Benchmarking compiled mode...")
    compiled_times = []
    for _ in range(NUM_TRIALS):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        q_compiled, k_compiled = apply_rope_compiled(q, k, pe)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        compiled_times.append((end - start) * 1000)

    compiled_avg = sum(compiled_times) / len(compiled_times)
    compiled_min = min(compiled_times)

    # Verify correctness
    print("\nCorrectness check...")
    q_close = torch.allclose(q_eager, q_compiled, rtol=1e-4, atol=1e-4)
    k_close = torch.allclose(k_eager, k_compiled, rtol=1e-4, atol=1e-4)
    print(f"Query tensors match: {q_close}")
    print(f"Key tensors match: {k_close}")

    if not q_close:
        q_diff = torch.abs(q_eager - q_compiled).max()
        print(f"Max query difference: {q_diff}")

    if not k_close:
        k_diff = torch.abs(k_eager - k_compiled).max()
        print(f"Max key difference: {k_diff}")

    # Results
    print(f"\nResults:")
    print(f"Eager mode:     {eager_avg:.3f} ms (min: {eager_min:.3f} ms)")
    print(f"Compiled mode:  {compiled_avg:.3f} ms (min: {compiled_min:.3f} ms)")

    speedup = eager_avg / compiled_avg
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Improvement: {((speedup - 1) * 100):.1f}%")

    return speedup

def benchmark_simple_reshape():
    """Benchmark simple tensor reshaping operations"""
    print("\n" + "="*50)
    print("Simple Reshape Operations Benchmark")
    print("="*50)

    # Test tensor similar to QKV output in TE
    x = torch.randn(SEQ_LEN, BATCH_SIZE, 3 * NUM_HEADS * HEAD_DIM, device=DEVICE)

    def eager_reshape(x):
        # Reshape [seq_len, batch, 3*heads*head_dim] -> [seq_len, batch, 3, heads, head_dim]
        reshaped = x.view(SEQ_LEN, BATCH_SIZE, 3, NUM_HEADS, HEAD_DIM)
        # Split into Q, K, V
        q, k, v = reshaped.unbind(dim=2)
        # Transpose for attention: [batch, heads, seq_len, head_dim]
        q = q.transpose(0, 1).transpose(1, 2)
        k = k.transpose(0, 1).transpose(1, 2)
        v = v.transpose(0, 1).transpose(1, 2)
        return q.contiguous(), k.contiguous(), v.contiguous()

    @torch.compile(mode="default")
    def compiled_reshape(x):
        reshaped = x.view(SEQ_LEN, BATCH_SIZE, 3, NUM_HEADS, HEAD_DIM)
        q, k, v = reshaped.unbind(dim=2)
        q = q.transpose(0, 1).transpose(1, 2)
        k = k.transpose(0, 1).transpose(1, 2)
        v = v.transpose(0, 1).transpose(1, 2)
        return q.contiguous(), k.contiguous(), v.contiguous()

    # Warmup
    for _ in range(WARMUP):
        _ = eager_reshape(x)
        _ = compiled_reshape(x)

    # Benchmark eager
    eager_times = []
    for _ in range(NUM_TRIALS):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        q_eager, k_eager, v_eager = eager_reshape(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        eager_times.append((end - start) * 1000)

    # Benchmark compiled
    compiled_times = []
    for _ in range(NUM_TRIALS):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        q_compiled, k_compiled, v_compiled = compiled_reshape(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        compiled_times.append((end - start) * 1000)

    eager_avg = sum(eager_times) / len(eager_times)
    compiled_avg = sum(compiled_times) / len(compiled_times)

    # Verify correctness
    q_close = torch.allclose(q_eager, q_compiled, rtol=1e-6)
    k_close = torch.allclose(k_eager, k_compiled, rtol=1e-6)
    v_close = torch.allclose(v_eager, v_compiled, rtol=1e-6)

    print(f"Correctness: Q={q_close}, K={k_close}, V={v_close}")
    print(f"Eager mode:    {eager_avg:.3f} ms")
    print(f"Compiled mode: {compiled_avg:.3f} ms")

    speedup = eager_avg / compiled_avg
    print(f"Speedup: {speedup:.2f}x ({((speedup - 1) * 100):.1f}% improvement)")

    return speedup

def main():
    print("TransformerEngine Surgical Compile - Focused Tests")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torch compile available: {hasattr(torch, 'compile')}")

    rope_speedup = benchmark_rope()
    reshape_speedup = benchmark_simple_reshape()

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"RoPE speedup:     {rope_speedup:.2f}x")
    print(f"Reshape speedup:  {reshape_speedup:.2f}x")
    print("\nThese results show the potential for surgical torch.compile")
    print("integration in TransformerEngine layers. Focus should be on:")
    print("1. RoPE computations (if speedup > 1.0)")
    print("2. Tensor reshaping/transposition operations")
    print("3. Operations between FP8 linear layers")

if __name__ == "__main__":
    main()