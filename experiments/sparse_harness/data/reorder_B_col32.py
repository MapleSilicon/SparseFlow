import numpy as np

K = 128
N = 128
TILE = 32

B = np.fromfile("dB.bin", np.float16).reshape(K, N, order="F")

# COL32 layout: columns grouped in 32
B_col32 = np.zeros_like(B)

for n in range(N):
    tile = (n // TILE) * TILE
    offset = n % TILE
    B_col32[:, tile + offset] = B[:, n]

B_col32.tofile("dB_col32.bin")
print("✓ Wrote dB_col32.bin")
