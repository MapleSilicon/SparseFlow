import torch
import time

print("="*60)
print("PYTORCH 2:4 SPARSE - CORRECT API")
print("="*60)

# Test compression and matmul
M, K, N = 128, 128, 128

# Create dense matrix
A_dense = torch.randn(M, K, dtype=torch.float16, device='cuda')
B = torch.randn(K, N, dtype=torch.float16, device='cuda')

# Apply 2:4 sparsity pattern
A_sparse_dense = A_dense.clone()
for i in range(0, K, 4):
    group = A_sparse_dense[:, i:i+4].abs()
    _, indices = torch.topk(group, k=2, dim=1, largest=False)
    for row in range(M):
        A_sparse_dense[row, i + indices[row, 0]] = 0
        A_sparse_dense[row, i + indices[row, 1]] = 0

sparsity = (A_sparse_dense == 0).float().mean()
print(f"Matrix: {M}×{K}")
print(f"Sparsity: {sparsity*100:.1f}%")

# Compress using CUTLASS library
print("\nCompressing with _cslt_compress...")
try:
    compressed = torch._cslt_compress(A_sparse_dense)
    print(f"✅ Compression successful!")
    print(f"Compressed type: {type(compressed)}")
    
    # Try sparse matmul
    print("\nTesting sparse matmul...")
    C_sparse = torch._cslt_sparse_mm(compressed, B)
    print(f"✅ Sparse matmul works!")
    print(f"Output shape: {C_sparse.shape}")
    
    # Verify correctness
    C_ref = torch.matmul(A_sparse_dense, B)
    error = torch.max(torch.abs(C_sparse - C_ref)).item()
    print(f"Max error vs reference: {error:.6f}")
    
    if error < 0.01:
        print("✅ CORRECTNESS VERIFIED!")
        
        # Now benchmark!
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK")
        print("="*60)
        
        for size in [512, 1024, 2048, 4096]:
            M = K = N = size
            
            A_dense = torch.randn(M, K, dtype=torch.float16, device='cuda')
            B = torch.randn(K, N, dtype=torch.float16, device='cuda')
            
            # Apply 2:4 pattern
            A_sparse_dense = A_dense.clone()
            for i in range(0, K, 4):
                group = A_sparse_dense[:, i:i+4].abs()
                _, indices = torch.topk(group, k=2, dim=1, largest=False)
                for row in range(M):
                    A_sparse_dense[row, i + indices[row, 0]] = 0
                    A_sparse_dense[row, i + indices[row, 1]] = 0
            
            compressed = torch._cslt_compress(A_sparse_dense)
            
            # Warmup
            for _ in range(10):
                _ = torch._cslt_sparse_mm(compressed, B)
            torch.cuda.synchronize()
            
            # Benchmark sparse
            start = time.perf_counter()
            for _ in range(50):
                C = torch._cslt_sparse_mm(compressed, B)
            torch.cuda.synchronize()
            sparse_time = time.perf_counter() - start
            
            # Benchmark dense
            for _ in range(10):
                _ = torch.matmul(A_dense, B)
            torch.cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(50):
                C = torch.matmul(A_dense, B)
            torch.cuda.synchronize()
            dense_time = time.perf_counter() - start
            
            flops = 2 * M * N * K * 50
            sparse_tflops = flops / sparse_time / 1e12
            dense_tflops = flops / dense_time / 1e12
            
            print(f"\n{size}×{size}:")
            print(f"  Dense:  {dense_tflops:6.2f} TFLOPS")
            print(f"  Sparse: {sparse_tflops:6.2f} TFLOPS")
            print(f"  Speedup: {sparse_tflops/dense_tflops:5.2f}×")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
