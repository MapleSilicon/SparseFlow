import numpy as np

M=N=K=128
DATA="data"

A = np.fromfile(f"{DATA}/A_pruned_dense.bin",np.float16).reshape(M,K).astype(np.float32)
B = np.fromfile(f"{DATA}/dB.bin",np.float16).reshape(K,N,order="F").astype(np.float32)
Dg = np.fromfile("D_cutlass.bin",np.float32).reshape(M,N)

Dc = A @ B
diff = np.abs(Dg-Dc)

print("D_gpu[0,:5] =",Dg[0,:5])
print("D_cpu[0,:5] =",Dc[0,:5])
print("max error =",diff.max())
print("mean error =",diff.mean())

assert diff.max()<1e-2
print("✅ SPARSEFLOW VERIFIED")
