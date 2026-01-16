import numpy as np

K = 128
N = 128
TILE = 32

B = np.fromfile("dB.bin", np.float16).reshape(K, N, order="F")

B_col32 = np.zeros((K, N), dtype=np.float16)

for n in range(N):
    for k in range(K):
        row = (k % TILE) + n * TILE
        col = k // TILE
        B_col32[k, n] = B[row % K, col] if row < K else 0

B_col32.tofile("dB_col32.bin")
print("✓ Correct CUTLASS COL32 B written")
