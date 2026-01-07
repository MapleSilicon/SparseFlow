import ctypes
import torch

lib = ctypes.CDLL("./gemm_pipeline.so")
lib.launch_gemm_pipeline.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p
]

def sf_matmul(A, B):
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), device="cuda", dtype=torch.float32)

    lib.launch_gemm_pipeline(
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_void_p(B.data_ptr()),
        ctypes.c_void_p(C.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
    )
    torch.cuda.synchronize()
    return C

shapes = [
    (16, 16, 16),
    (128, 128, 128),
    (129, 257, 65),
    (512, 512, 512),
]

for (M, N, K) in shapes:
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)

    C1 = sf_matmul(A, B)
    C2 = (A.float() @ B.float())

    max_err = (C1 - C2).abs().max().item()
    print(f"{M}x{N}x{K} -> max_err = {max_err:.6f}")
