# torch.compile benchmarks

**NVIDIA H100 PCIe | PyTorch 2.8.0a0+34c6371d24.nv25.08**

## Performance Results

**Llama-1B Model (8 layers, 2048 seq len, batch size 2)**

| Approach | Time (ms) | vs Optimal | Status |
|----------|-----------|------------|---------|
| **🏆 Hybrid + torch.compile** | **15.3** | **Baseline** | ✅ **OPTIMAL** |
| 🥈 Hybrid (TE Linear + cuDNN SDPA) | 16.9 | 1.10x slower | ✅ Excellent base |
| 🥉 TE TransformerLayer | 17.3-17.5 | 1.13-1.14x slower | ⚠️ TE's best option |
| TE MultiheadAttention | 21.2 | 1.38x slower | ❌ Not production viable |

**Performance Gains:**
- **+10.0%** from surgical torch.compile (16.9 → 15.3 ms)
- **+13-14%** vs TE TransformerLayer (TE's production choice)
- **+25.4%** vs standalone TE attention (reference only)

## Architecture

**Optimal Stack:**
```python
# TE fused linear operations (best FP8 performance)
qkv = te.LayerNormLinear(hidden_size, 3 * hidden_size)(x)
mlp = te.LayerNormMLP(hidden_size, ffn_hidden_size)(x)

# Compiled tensor operations
q, k, v = torch.compile(qkv_reshape_split)(qkv, heads, head_dim)
q, k = torch.compile(apply_rope)(q, k, rope_embeddings)

# cuDNN SDPA (faster than TE attention)
attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**Why This Works:**
- TE Linear: Best FP8 matrix multiplication
- cuDNN SDPA: 25.4% faster than TE attention
- torch.compile: 10% boost on tensor ops

## Benchmark Files

### Current Production Benchmarks
- `steady_state_benchmark.py` - Definitive performance comparison (proper compilation separation)
- `te_llama_real_attention.py` - Complete TE vs hybrid comparison
- `te_all_kernels_test.py` - Comprehensive TE kernel testing

### Research Prototypes
- `te_llama_phase1.py` - Extended surgical compilation experiments
- `te_llama_phase2.py` - TE attention capturable wrappers

## Compilation Analysis

**Steady-state performance (after compilation warmup):**

| Metric | Hybrid Baseline | Hybrid + Compile | TE Attention |
|--------|-----------------|------------------|--------------|
| Average | 16.871 ms | **15.338 ms** | 21.164 ms |
| Std Dev | ±0.12 | ±0.16 | ±0.18 |
| Compilation cost | 1.8s (one-time) | 4.6s (one-time) | 2.2s (one-time) |

## Production Recommendations

### ✅ DO
- Use TE Linear + cuDNN SDPA hybrid architecture
- Apply surgical torch.compile to tensor operations only
- Keep TE linear layers and MLP in eager mode

### ❌ AVOID
- TE MultiheadAttention (25% slower than optimal)
- Compiling TE linear operations (breaks FP8 optimization)
- Pure torch.compile without TE linear layers

## Conclusion

**The optimal Transformer implementation combines:**
1. TE's industry-leading FP8 linear optimizations
2. cuDNN's superior attention kernel performance
3. torch.compile's tensor operation optimization

**Result: 13-14% faster than TE's production TransformerLayer**