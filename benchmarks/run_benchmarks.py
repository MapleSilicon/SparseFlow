#!/usr/bin/env python3
"""
SparseFlow Performance Benchmark Suite
Measures dense vs sparse performance across LLaMA shapes
"""
import torch
import time
import csv
from pathlib import Path

torch.backends.cuda.matmul.allow_tf32 = False

def manual_24_prune(dense_tensor):
    M, K = dense_tensor.shape
    pruned = torch.zeros_like(dense_tensor)
    for i in range(M):
        for j in range(0, K, 4):
            block = dense_tensor[i, j:j+4]
            _, indices = torch.topk(torch.abs(block), k=2, sorted=False)
            for idx in indices:
                pruned[i, j + idx] = block[idx]
    return pruned

def benchmark_shape(M, N, K, warmup=10, iterations=100):
    """Benchmark dense vs sparse for given shape"""
    print(f"Benchmarking M={M}, N={N}, K={K}...")
    
    # Setup
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')
    A_pruned = manual_24_prune(A)
    A_sparse = torch.sparse.to_sparse_semi_structured(A_pruned)
    
    # Warmup
    for _ in range(warmup):
        _ = torch.matmul(A_pruned, B)
        _ = torch.matmul(A_sparse, B)
    torch.cuda.synchronize()
    
    # Benchmark dense
    start = time.perf_counter()
    for _ in range(iterations):
        C_dense = torch.matmul(A_pruned, B)
    torch.cuda.synchronize()
    dense_time = (time.perf_counter() - start) / iterations * 1000
    
    # Benchmark sparse
    start = time.perf_counter()
    for _ in range(iterations):
        C_sparse = torch.matmul(A_sparse, B)
    torch.cuda.synchronize()
    sparse_time = (time.perf_counter() - start) / iterations * 1000
    
    # Calculate metrics
    dense_flops = 2 * M * N * K
    sparse_flops = dense_flops
    
    dense_tflops = (dense_flops / (dense_time * 1e-3)) / 1e12
    sparse_tflops = (sparse_flops / (sparse_time * 1e-3)) / 1e12
    
    speedup = dense_time / sparse_time
    
    # A100 theoretical peak: 312 TFLOPS (FP16 Tensor Core)
    a100_peak_tflops = 312
    dense_util = (dense_tflops / a100_peak_tflops) * 100
    sparse_util = (sparse_tflops / a100_peak_tflops) * 100
    
    return {
        'M': M, 'N': N, 'K': K,
        'dense_time_ms': f'{dense_time:.4f}',
        'sparse_time_ms': f'{sparse_time:.4f}',
        'speedup': f'{speedup:.2f}',
        'dense_tflops': f'{dense_tflops:.2f}',
        'sparse_tflops': f'{sparse_tflops:.2f}',
        'dense_util_%': f'{dense_util:.1f}',
        'sparse_util_%': f'{sparse_util:.1f}',
    }

def main():
    print("="*70)
    print("SparseFlow Performance Benchmark Suite")
    print("="*70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}\n")
    
    shapes = [
        (512, 512, 512, "Small square"),
        (1024, 1024, 1024, "Medium square"),
        (2048, 2048, 2048, "Large square"),
        (4096, 4096, 4096, "XL square"),
        (128, 4096, 4096, "LLaMA-70B attn (seq=128)"),
        (512, 4096, 4096, "LLaMA-70B attn (seq=512)"),
        (2048, 4096, 4096, "LLaMA-70B attn (seq=2048)"),
        (512, 11008, 4096, "LLaMA-70B FFN gate (seq=512)"),
        (2048, 11008, 4096, "LLaMA-70B FFN gate (seq=2048)"),
        (512, 4096, 11008, "LLaMA-70B FFN down (seq=512)"),
        (2048, 4096, 11008, "LLaMA-70B FFN down (seq=2048)"),
    ]
    
    results = []
    for M, N, K, name in shapes:
        result = benchmark_shape(M, N, K)
        result['shape_name'] = name
        results.append(result)
        print(f"  {name}: {result['speedup']}x speedup "
              f"({result['sparse_tflops']} TFLOPS)\n")
    
    output_file = Path(__file__).parent / 'results_sparseflow.csv'
    fieldnames = ['shape_name', 'M', 'N', 'K', 'dense_time_ms', 'sparse_time_ms', 
                  'speedup', 'dense_tflops', 'sparse_tflops', 'dense_util_%', 'sparse_util_%']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")
    
    speedups = [float(r['speedup']) for r in results]
    avg_speedup = sum(speedups) / len(speedups)
    max_speedup = max(speedups)
    
    print(f"\nSummary:")
    print(f"  Average speedup: {avg_speedup:.2f}x")
    print(f"  Max speedup: {max_speedup:.2f}x")
    print(f"  Shapes tested: {len(results)}")

if __name__ == "__main__":
    main()
