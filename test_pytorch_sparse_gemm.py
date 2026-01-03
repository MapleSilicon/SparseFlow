import torch
import time

print("="*60)
print("PYTORCH 2:4 SPARSE TENSOR CORE TEST")
print("="*60)

# Test different sizes
for M in [128, 512, 1024, 2048, 4096]:
    K, N = M, M
    
    # Create dense matrices
    A_dense = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')
    
    # Apply 2:4 sparsity to A
    A_sparse_dense = A_dense.clone()
    for i in range(0, K, 4):
        group = A_sparse_dense[:, i:i+4].abs()
        _, indices = torch.topk(group, k=2, dim=1, largest=False)
        for row in range(M):
            A_sparse_dense[row, i + indices[row, 0]] = 0
            A_sparse_dense[row, i + indices[row, 1]] = 0
    
    # Compress to sparse format
    A_compressed = torch._sparse_semi_structured_apply(
        A_sparse_dense, 
        torch.float16
    )
    
    # Warmup
    for _ in range(10):
        C_sparse = torch._cslt_sparse_mm(A_compressed, B)
    torch.cuda.synchronize()
    
    # Benchmark sparse
    start = time.perf_counter()
    for _ in range(100):
        C_sparse = torch._cslt_sparse_mm(A_compressed, B)
    torch.cuda.synchronize()
    sparse_time = time.perf_counter() - start
    
    # Benchmark dense (cuBLAS)
    for _ in range(10):
        C_dense = torch.matmul(A_dense, B)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(100):
        C_dense = torch.matmul(A_dense, B)
    torch.cuda.synchronize()
    dense_time = time.perf_counter() - start
    
    # Calculate TFLOPS
    # Sparse does 2× the K dimension but only needs 50% of data
    flops = 2 * M * N * K * 100
    sparse_tflops = flops / sparse_time / 1e12
    dense_tflops = flops / dense_time / 1e12
    
    speedup = dense_time / sparse_time
    
    print(f"\n{M}×{M}:")
    print(f"  Dense (cuBLAS):  {dense_tflops:6.2f} TFLOPS")
    print(f"  Sparse (2:4):    {sparse_tflops:6.2f} TFLOPS")
    print(f"  Speedup:         {speedup:6.2f}×")

print("\n" + "="*60)
