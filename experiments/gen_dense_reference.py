import numpy as np

M = N = K = 128
np.random.seed(0)

A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)

D = A @ B

A.astype(np.float16).tofile("A_dense.bin")
B.astype(np.float16, order="F").tofile("B_dense.bin")
D.tofile("D_dense_ref.bin")

print("✓ Wrote A_dense.bin, B_dense.bin, D_dense_ref.bin")
