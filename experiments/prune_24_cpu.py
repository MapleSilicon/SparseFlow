import numpy as np

M = N = K = 128
GROUPS = K // 4

A = np.fromfile("A_dense.bin", np.float16).reshape(M, K)

A_pruned = np.zeros_like(A)
E = np.zeros((M, GROUPS), dtype=np.uint8)
A_comp = np.zeros((M, K//2), dtype=np.float16)

PAIR_TO_CODE = {
    (0,1): 9, (0,2): 4, (0,3): 8,
    (1,2): 12, (1,3): 14, (2,3): 13,
}

for m in range(M):
    c = 0
    for g in range(GROUPS):
        vals = A[m, 4*g:4*g+4]
        idx = np.argsort(np.abs(vals))[-2:]
        idx.sort()
        A_pruned[m, 4*g + idx[0]] = vals[idx[0]]
        A_pruned[m, 4*g + idx[1]] = vals[idx[1]]
        A_comp[m, c] = vals[idx[0]]
        A_comp[m, c+1] = vals[idx[1]]
        c += 2
        E[m, g] = PAIR_TO_CODE[tuple(idx)]

A_pruned.astype(np.float16).tofile("A_pruned_dense.bin")
A_comp.tofile("dA.bin")
E.tofile("dE.bin")

print("✓ CPU pruning complete")
