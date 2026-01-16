import numpy as np

M = N = K = 128

A = np.fromfile("A_pruned_dense.bin", np.float16).reshape(M,K).astype(np.float32)
B = np.fromfile("B_dense.bin", np.float16).reshape(K,N,order="F").astype(np.float32)
D_ref = np.fromfile("D_dense_ref.bin", np.float32).reshape(M,N)

D = A @ B
diff = np.abs(D - D_ref)

print("CPU pruned vs dense:")
print("max error:", diff.max())
print("mean error:", diff.mean())
