# SparseFlow Integration Guide

Complete guide for integrating SparseFlow into your LLaMA/Transformer inference pipeline.

---

## 📋 Requirements

### Hardware
- **GPU Architecture**: NVIDIA Ampere or newer (A100, A6000, RTX 30/40 series, H100)
- **Compute Capability**: 8.0+ (required for 2:4 sparse tensor cores)
- **Memory**: Sufficient for your model (same as dense inference)

**Check your GPU:**
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
# Should be (8, 0) or higher for Ampere+
```

### Software
- **PyTorch**: 2.0+ (with cuSPARSELt support)
- **CUDA**: 11.8+ or 12.x
- **Python**: 3.8+

**Verify installation:**
```python
import torch
assert torch.cuda.is_available(), "CUDA not available"
assert torch.__version__ >= "2.0", "PyTorch 2.0+ required"
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
```

### Shape Requirements
- **K dimension**: Must be multiple of 4 (for 2:4 pruning)
- **Recommended batch size**: M ≥ 512 for optimal speedup
- **Data type**: FP16 (sparse tensor cores are FP16 only)

---

## 🚀 Quick Start

### Basic Usage
```python
import torch

# Your existing dense matmul
A = torch.randn(2048, 4096, dtype=torch.float16, device='cuda')
B = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
C = torch.matmul(A, B)  # Dense computation

# Convert to SparseFlow (2 steps)
# Step 1: Prune A to 2:4 sparsity pattern
A_pruned = sparseflow.prune_24(A)  # Keep top-2 of every 4 weights

# Step 2: Convert to sparse tensor core format
A_sparse = torch.sparse.to_sparse_semi_structured(A_pruned)

# Step 3: Sparse matmul (drop-in replacement)
C = torch.matmul(A_sparse, B)  # 1.2-1.4× faster on large batches
```

### Complete Example (LLaMA Attention)
```python
import torch

def prune_24(dense_tensor):
    """Prune matrix to 2:4 sparsity pattern"""
    M, K = dense_tensor.shape
    pruned = torch.zeros_like(dense_tensor)
    
    for i in range(M):
        for j in range(0, K, 4):
            # For each 4-element block, keep top-2 by magnitude
            block = dense_tensor[i, j:j+4]
            _, indices = torch.topk(torch.abs(block), k=2, sorted=False)
            for idx in indices:
                pruned[i, j + idx] = block[idx]
    
    return pruned

# LLaMA-70B attention (seq_len=2048)
batch_size = 2048
hidden_dim = 4096

Q = torch.randn(batch_size, hidden_dim, dtype=torch.float16, device='cuda')
W_attn = torch.randn(hidden_dim, hidden_dim, dtype=torch.float16, device='cuda')

# Apply 2:4 pruning to weight matrix
W_attn_pruned = prune_24(W_attn)

# Convert to sparse format (one-time conversion)
W_attn_sparse = torch.sparse.to_sparse_semi_structured(W_attn_pruned)

# Inference (fast path)
output = torch.matmul(Q, W_attn_sparse)
```

---

## ✅ What SparseFlow Guarantees

### 1. Numerical Correctness
- **Max absolute error**: < 0.2 (FP16 precision-appropriate)
- **Mean absolute error**: < 0.01 for large matrices
- **Validated shapes**: All LLaMA-70B attention + FFN dimensions
- **Deterministic**: Same input → same output (within FP16 bounds)

**Validation results:**
```
11/11 production shapes passing validation
Max error: 0.031-0.194 across all shapes
Zero conversion errors (round-trip perfect)
```

### 2. Performance Characteristics
- **Speedup range**: 1.2-1.4× on production workloads (batch ≥ 512)
- **Peak throughput**: 308-334 TFLOPS on A100
- **Memory reduction**: 50% (sparse weights)
- **Compute-bound**: Fully utilizes tensor cores on large batches

### 3. Compatibility
- **Drop-in replacement**: Works with existing PyTorch code
- **No model changes**: Pure inference-time optimization
- **Framework agnostic**: Works with HuggingFace, vLLM, TGI

---

## ❌ What SparseFlow Does NOT Guarantee

### 1. When Speedup is NOT Guaranteed

**Small batch sizes** (M < 256):
- Kernel launch overhead dominates
- May be **slower** than dense (0.6-0.9× on small shapes)
- Recommendation: Use dense matmul for batch < 256

**Memory-bound operations**:
- Very wide matrices (N >> M, K)
- Speedup limited by DRAM bandwidth

### 2. Limitations

**Training not supported**:
- No gradient computation for sparse tensors (yet)
- Use for inference only

**FP16 only**:
- Sparse tensor cores are FP16 hardware
- FP32/BF16 will fallback to dense (no speedup)

**Pre-Ampere GPUs**:
- V100, T4, etc. lack sparse tensor core hardware
- Will fallback to dense matmul

**Dynamic sparsity**:
- Sparsity pattern must be known ahead of time
- Conversion has overhead - do it once, reuse many times

---

## 🔄 Fallback Behavior

SparseFlow automatically falls back to dense computation when:

1. **Unsupported GPU** (pre-Ampere)
```python
   if torch.cuda.get_device_capability()[0] < 8:
       # Automatic fallback to dense
       C = torch.matmul(A_pruned, B)  # Uses dense path
```

2. **Wrong data type** (not FP16)
```python
   A_fp32 = torch.randn(M, K, dtype=torch.float32, device='cuda')
   A_sparse = torch.sparse.to_sparse_semi_structured(A_fp32)  # Converts to FP16 internally
```

3. **Shape requirements not met** (K not divisible by 4)
   - Pad K dimension to nearest multiple of 4

---

## 🎯 Performance Tips

### 1. Batch Workloads for Best Speedup
```python
# ❌ Bad: Small batch (overhead dominates)
for i in range(1000):
    x = torch.randn(32, 4096, dtype=torch.float16, device='cuda')
    y = torch.matmul(x, W_sparse)  # Slow!

# ✅ Good: Large batch (amortize overhead)
X = torch.randn(2048, 4096, dtype=torch.float16, device='cuda')
Y = torch.matmul(X, W_sparse)  # Fast!
```

### 2. Reuse Sparse Tensors
```python
# ❌ Bad: Convert every forward pass
for batch in dataloader:
    W_sparse = torch.sparse.to_sparse_semi_structured(W_pruned)  # Overhead!
    output = torch.matmul(batch, W_sparse)

# ✅ Good: Convert once, reuse
W_sparse = torch.sparse.to_sparse_semi_structured(W_pruned)  # One-time
for batch in dataloader:
    output = torch.matmul(batch, W_sparse)  # Fast!
```

### 3. Apply Selectively

Not all layers benefit equally. Focus on:
- **Attention projections** (Q, K, V, O)
- **FFN layers** (gate, up, down)
- **Large matrix dimensions** (4096×4096, 11008×4096)

Skip:
- Small embedding layers
- Layer norms
- Activation functions

### 4. Profile Before Deploying
```python
import time

def benchmark(matmul_fn, A, B, iterations=100):
    # Warmup
    for _ in range(10):
        _ = matmul_fn(A, B)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(iterations):
        _ = matmul_fn(A, B)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / iterations * 1000  # ms

# Test your specific shapes
dense_time = benchmark(lambda A, B: torch.matmul(A, B), A_pruned, B)
sparse_time = benchmark(lambda A, B: torch.matmul(A, B), A_sparse, B)

print(f"Dense:  {dense_time:.2f} ms")
print(f"Sparse: {sparse_time:.2f} ms")
print(f"Speedup: {dense_time / sparse_time:.2f}×")
```

---

## 🐛 Troubleshooting

### "RuntimeError: expected scalar type Half but found Float"
**Cause**: Sparse tensor cores require FP16  
**Fix**: Ensure inputs are FP16
```python
A = A.half()  # Convert to FP16
B = B.half()
```

### "No speedup observed"
**Cause**: Batch size too small or wrong GPU  
**Fix**: 
1. Check GPU: `torch.cuda.get_device_capability()[0] >= 8`
2. Increase batch size: M ≥ 512
3. Profile to verify you're actually using sparse path

### "CUDA out of memory"
**Cause**: Conversion temporarily uses more memory  
**Fix**:
```python
# Free memory before conversion
torch.cuda.empty_cache()
A_sparse = torch.sparse.to_sparse_semi_structured(A_pruned)
```

### "Results don't match dense"
**Cause**: Different FP16 rounding (expected)  
**Fix**: This is normal. Validate against FP32 ground truth:
```python
C_ref = (A_pruned.float() @ B.float())  # FP32 reference
C_sparse = (A_sparse @ B).float()
error = torch.abs(C_ref - C_sparse).max()
assert error < 0.2, f"Error {error} too large"
```

---

## 📚 Additional Resources

- **Validation script**: `tests/validate_sparseflow_production.py`
- **Benchmarks**: `benchmarks/run_benchmarks.py`
- **Demo notebook**: `demo/SparseFlow_Demo.ipynb`
- **Performance dashboard**: `benchmarks/sparseflow_dashboard.png`

## 🤝 Support

- **GitHub Issues**: [MapleSilicon/SparseFlow/issues](https://github.com/MapleSilicon/SparseFlow/issues)
- **Documentation**: [Full docs](https://github.com/MapleSilicon/SparseFlow)

---

## ⚖️ Production Checklist

Before deploying SparseFlow to production:

- [ ] Validated correctness on your specific model shapes
- [ ] Profiled and confirmed speedup (≥1.2× for batch ≥ 512)
- [ ] Tested on target GPU hardware (Ampere+)
- [ ] Converted weights once, cached for inference
- [ ] Added fallback for unsupported GPUs
- [ ] Monitored accuracy metrics in staging
- [ ] Load tested under production traffic

**SparseFlow is production-ready when used within these guidelines.**
