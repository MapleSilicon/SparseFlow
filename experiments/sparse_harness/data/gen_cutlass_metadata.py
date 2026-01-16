import numpy as np

M = 128
K = 128
GROUP = 4

# Load pruned dense A (float16)
A = np.fromfile("A_pruned_dense.bin", np.float16).reshape(M, K)

# Output metadata
E = np.zeros((K // GROUP, M), dtype=np.uint16)

pair_to_code = {
    (0,1): 0,
    (0,2): 1,
    (0,3): 2,
    (1,2): 3,
    (1,3): 4,
    (2,3): 5,
}

for m in range(M):
    for g in range(K // GROUP):
        block = A[m, g*4:(g+1)*4]
        nz = tuple(np.nonzero(block)[0].tolist())
        if len(nz) != 2:
            raise RuntimeError(f"Invalid sparsity at row {m}, group {g}: {nz}")
        E[g, m] = pair_to_code[nz]

# ColumnMajorInterleaved<2> wants this exact shape/layout
E.tofile("dE.bin")
print("✓ Wrote CUTLASS-formatted dE.bin")
print("First 16 entries:", E.flatten()[:16])
