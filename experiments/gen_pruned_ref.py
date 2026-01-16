import numpy as np

M = N = K = 128

A = np.fromfile("A_pruned_dense.bin", np.float16).reshape(M,K).astype(np.float32)
B = np.fromfile("B_dense.bin", np.float16).reshape(K,N,order="F").astype(np.float32)

D = A @ B
D.tofile("D_pruned_ref.bin")

print("✓ Wrote D_pruned_ref.bin")
print("Sample D[0,:5] =", D[0,:5])
