import torch
import time
import ctypes
import numpy as np

print("="*70)
print("SparseFlow Benchmark: Sparse v3 vs Dense WMMA")
print("="*70)

# Load libraries
dense_lib = ctypes.CDLL("../../wmma_dense/dense_wmma_grid.so")
sparse_lib = ctypes.CDLL("./libsparse_ref_v3.so")

# Set function signatures
dense_lib.launch_dense_wmma_fp32.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

sparse_lib.launch_sparse_ref_v3.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

def apply_24_sparsity(A):
    """Apply 2:4 sparsity pattern to matrix A"""
    M, K = A.shape
    A_pruned = A.clone()
    
    for i in range(0, K, 4):
        group = A_pruned[:, i:i+4]
        _, indices = torch.topk(torch.abs(group), k=2, dim=1, largest=False)
        for row in range(M):
            A_pruned[row, i + indices[row, 0]] = 0
            A_pruned[row, i + indices[row, 1]] = 0
    
    return A_pruned

def compress_24(A_pruned):
    """Compress 2:4 sparse matrix and generate metadata"""
    M, K = A_pruned.shape
    A_comp = torch.empty((M, K//2), dtype=torch.float16, device='cuda')
    E = torch.empty((K//4, M), dtype=torch.uint16, device='cuda')
    
    pairs = {(0,1):0, (0,2):1, (0,3):2, (1,2):3, (1,3):4, (2,3):5}
    
    A_np = A_pruned.cpu().numpy()
    E_np = np.zeros((K//4, M), dtype=np.uint16)
    A_comp_np = np.zeros((M, K//2), dtype=np.float16)
    
    for row in range(M):
        for g in range(K//4):
            dense_group = A_np[row, g*4:(g+1)*4]
            nz_indices = np.where(dense_group != 0)[0]
            
            if len(nz_indices) >= 2:
                i0, i1 = nz_indices[0], nz_indices[1]
                E_np[g, row] = pairs.get((i0, i1), 0)
                A_comp_np[row, g*2] = dense_group[i0]
                A_comp_np[row, g*2+1] = dense_group[i1]
    
    A_comp[:] = torch.from_numpy(A_comp_np).cuda()
    E[:] = torch.from_numpy(E_np).cuda()
    
    return A_comp, E.flatten()

def benchmark(M, N, K, iters=50):
    print(f"\n{'='*70}")
    print(f"Shape: {M}×{K} @ {K}×{N} = {M}×{N}")
    print(f"{'='*70}")
    
    # Create matrices
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    # Apply 2:4 sparsity to A
    A_pruned = apply_24_sparsity(A)
    A_comp, E = compress_24(A_pruned)
    
    print(f"Original A: {A.numel() * 2 / 1024**2:.2f} MB")
    print(f"Compressed: {A_comp.numel() * 2 / 1024**2:.2f} MB (50% reduction)")
    print(f"Metadata:   {E.numel() * 2 / 1024**2:.2f} MB")
    
    # Prepare outputs
    B_col = B.t().contiguous()
    C_dense = torch.empty(M, N, device='cuda', dtype=torch.float32)
    C_sparse = torch.empty(M, N, device='cuda', dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        dense_lib.launch_dense_wmma_fp32(
            A.data_ptr(), B_col.data_ptr(), C_dense.data_ptr(), M, N, K
        )
        sparse_lib.launch_sparse_ref_v3(
            A_comp.data_ptr(), E.data_ptr(), B.data_ptr(), C_sparse.data_ptr(),
            M, N, K
        )
    torch.cuda.synchronize()
    
    # Benchmark dense
    t0 = time.time()
    for _ in range(iters):
        dense_lib.launch_dense_wmma_fp32(
            A.data_ptr(), B_col.data_ptr(), C_dense.data_ptr(), M, N, K
        )
    torch.cuda.synchronize()
    dense_time = (time.time() - t0) / iters
    
    # Benchmark sparse
    t0 = time.time()
    for _ in range(iters):
        sparse_lib.launch_sparse_ref_v3(
            A_comp.data_ptr(), E.data_ptr(), B.data_ptr(), C_sparse.data_ptr(),
            M, N, K
        )
    torch.cuda.synchronize()
    sparse_time = (time.time() - t0) / iters
    
    # Calculate metrics
    flops = 2 * M * N * K
    dense_tflops = flops / (dense_time * 1e12)
    sparse_tflops = flops / (sparse_time * 1e12)
    speedup = dense_time / sparse_time
    
    # Verify correctness
    max_err = (C_dense - C_sparse).abs().max().item()
    
    print(f"\n{'Results':-^70}")
    print(f"Dense WMMA:     {dense_time*1000:7.3f} ms  |  {dense_tflops:6.2f} TFLOPS")
    print(f"Sparse Ref v3:  {sparse_time*1000:7.3f} ms  |  {sparse_tflops:6.2f} TFLOPS")
    print(f"{'='*70}")
    print(f"Speedup:        {speedup:.2f}×")
    print(f"Max error:      {max_err:.2e}")
    
    return sparse_tflops, speedup

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    
    # Progressive benchmarks
    for size in [128, 256, 512, 1024, 2048]:
        tflops, speedup = benchmark(size, size, size)
        
    print(f"\n{'='*70}")
    print("Benchmark complete!")
    print(f"{'='*70}")
