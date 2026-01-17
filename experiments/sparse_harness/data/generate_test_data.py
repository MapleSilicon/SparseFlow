import numpy as np
import torch

def apply_24_sparsity_cpu(A):
    """Apply 2:4 sparsity pattern (keep top 2 magnitudes per group of 4)"""
    M, K = A.shape
    A_pruned = A.copy()
    
    for row in range(M):
        for g in range(K // 4):
            group = A_pruned[row, g*4:(g+1)*4]
            # Find indices of 2 smallest magnitude values
            abs_vals = np.abs(group)
            indices = np.argsort(abs_vals)[:2]  # 2 smallest
            # Zero them out
            for idx in indices:
                A_pruned[row, g*4 + idx] = 0
    
    return A_pruned

def compress_24_cpu(A_pruned):
    """Compress 2:4 sparse matrix to [M, K/2] and generate metadata [K/4, M]"""
    M, K = A_pruned.shape
    A_comp = np.zeros((M, K//2), dtype=np.float16)
    E = np.zeros((K//4, M), dtype=np.uint16)
    
    # Encoding: which 2 of 4 are kept
    pairs = {(0,1):0, (0,2):1, (0,3):2, (1,2):3, (1,3):4, (2,3):5}
    
    for row in range(M):
        for g in range(K // 4):
            group = A_pruned[row, g*4:(g+1)*4]
            nz_indices = np.where(group != 0)[0]
            
            if len(nz_indices) >= 2:
                i0, i1 = nz_indices[0], nz_indices[1]
                code = pairs.get((i0, i1), 0)
                E[g, row] = code  # Transposed: [K/4, M]
                A_comp[row, g*2] = group[i0]
                A_comp[row, g*2 + 1] = group[i1]
    
    return A_comp, E

def generate_test_case(M, N, K, seed=42):
    """Generate test matrices for given size"""
    np.random.seed(seed)
    
    print(f"Generating {M}×{K} @ {K}×{N}...")
    
    # Generate random matrices
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(K, N).astype(np.float16)
    
    # Apply 2:4 sparsity to A
    A_pruned = apply_24_sparsity_cpu(A)
    
    # Compress
    A_comp, E = compress_24_cpu(A_pruned)
    
    # Compute reference (CPU)
    D_ref = (A_pruned.astype(np.float32) @ B.astype(np.float32)).astype(np.float32)
    
    # Save to files
    prefix = f"test_{M}x{N}x{K}"
    A_comp.tofile(f"{prefix}_Acomp.bin")
    A_pruned.tofile(f"{prefix}_Apruned.bin")
    E.flatten().tofile(f"{prefix}_E.bin")
    B.tofile(f"{prefix}_B.bin")
    D_ref.tofile(f"{prefix}_Dref.bin")
    
    print(f"  Saved: {prefix}_*.bin")
    print(f"  A_comp: {A_comp.shape}, E: {E.shape}, B: {B.shape}")
    return prefix

# Generate test cases
sizes = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
]

print("="*70)
print("Generating Test Data for Scale Testing")
print("="*70)

prefixes = []
for M, N, K in sizes:
    prefix = generate_test_case(M, N, K)
    prefixes.append((M, N, K, prefix))
    print()

print("="*70)
print("Test data generation complete!")
print("="*70)
