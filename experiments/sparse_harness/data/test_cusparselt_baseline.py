import torch
import time
import numpy as np

# Check if cuSPARSELt is available
try:
    import cusparselt
    print("✅ cuSPARSELt available")
except:
    print("❌ cuSPARSELt not available - install with: pip install nvidia-cusparselt-cu12")
    exit(1)

M = N = K = 128

# Load your validated data
Acomp = np.fromfile("dA.bin", np.float16).reshape(M, K//2)
E = np.fromfile("dE.bin", np.uint16)
B = np.fromfile("dB.bin", np.float16).reshape(K, N)
D_ref = np.fromfile("D_pruned_ref.bin", np.float32).reshape(M, N)

# TODO: Convert to cuSPARSELt format and benchmark
# This gives us the "theoretical best" with tensor cores
