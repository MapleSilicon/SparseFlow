import os, numpy as np
np.random.seed(0)

M=N=K=128
GROUPS=K//4
K_HALF=K//2

OUT_DA="dA.bin"
OUT_DB="dB.bin"
OUT_DE0="dE_case_0.bin"
OUT_IDX="indices_sorted.npy"
OUT_AREF="A_pruned_dense.bin"

A_dense = np.random.randn(M, K).astype(np.float16)

idx = np.zeros((M, GROUPS, 2), dtype=np.int32)
A_comp = np.zeros((M, K_HALF), dtype=np.float16)
A_pruned = np.zeros((M, K), dtype=np.float16)

PAIR_TO_CODE = {
    (0,1): 9, (0,2): 4, (0,3): 8,
    (1,2): 12, (1,3): 14, (2,3): 13,
}

# E: 16 bytes/row (32 groups => 32 nibbles => 16 bytes)
E = np.zeros((M, 16), dtype=np.uint8)

for m in range(M):
    comp_col = 0
    for g in range(GROUPS):
        block = A_dense[m, 4*g:4*g+4].astype(np.float32)
        picks = np.argsort(np.abs(block))[-2:]
        picks = np.sort(picks)
        i0, i1 = int(picks[0]), int(picks[1])
        idx[m, g, 0] = i0
        idx[m, g, 1] = i1

        v0 = A_dense[m, 4*g+i0]
        v1 = A_dense[m, 4*g+i1]

        A_comp[m, comp_col+0] = v0
        A_comp[m, comp_col+1] = v1
        comp_col += 2

        A_pruned[m, 4*g+i0] = v0
        A_pruned[m, 4*g+i1] = v1

        code = PAIR_TO_CODE[(i0,i1)]
        byte_i = g // 2
        if (g % 2) == 0:
            E[m, byte_i] |= (code & 0xF)
        else:
            E[m, byte_i] |= ((code & 0xF) << 4)

# B written ColumnMajor (Fortran order)
B = np.random.randn(K, N).astype(np.float16)
B.ravel(order="F").tofile(OUT_DB)

A_comp.tofile(OUT_DA)
E.tofile(OUT_DE0)
np.save(OUT_IDX, idx)
A_pruned.astype(np.float16).tofile(OUT_AREF)

print("Wrote:", OUT_DA, OUT_DB, OUT_DE0, OUT_IDX, OUT_AREF)
for f in [OUT_DA, OUT_DB, OUT_DE0, OUT_IDX, OUT_AREF]:
    print(f, os.path.getsize(f))
