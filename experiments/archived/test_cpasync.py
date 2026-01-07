import torch, ctypes, time

lib = ctypes.CDLL('./gemm_ptx_cpasync.so')
lib.launch_gemm_ptx_scaled_cpasync = lib.gemm_ptx_scaled_cpasync
lib.launch_gemm_ptx_scaled_cpasync.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3

print("="*60)
print("CP.ASYNC KERNEL TEST (Triple-buffered)")
print("="*60)

# Correctness
M = 1024
A = torch.randn(M, M, device='cuda', dtype=torch.float16)
B = torch.randn(M, M, device='cuda', dtype=torch.float16)
C = torch.zeros(M, M, device='cuda', dtype=torch.float32)

lib.launch_gemm_ptx_scaled_cpasync(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, M, M)
torch.cuda.synchronize()

C_ref = torch.matmul(A.float(), B.float())
err = torch.max(torch.abs(C.cpu() - C_ref.cpu())).item()

print(f"Correctness: max_error = {err:.6f}")

if err < 0.01:
    print("✅ CORRECTNESS PASSED!\n")
    
    # Performance
    def bench(M):
        A = torch.randn(M, M, device='cuda', dtype=torch.float16)
        B = torch.randn(M, M, device='cuda', dtype=torch.float16)
        C = torch.zeros(M, M, device='cuda', dtype=torch.float32)
        
        for _ in range(10):
            lib.launch_gemm_ptx_scaled_cpasync(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, M, M)
        torch.cuda.synchronize()
        
        s = time.perf_counter()
        for _ in range(50):
            lib.launch_gemm_ptx_scaled_cpasync(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, M, M)
        torch.cuda.synchronize()
        
        return (2*M**3*50)/(time.perf_counter()-s)/1e12
    
    print("Performance (Triple-buffered cp.async):")
    for size in [2048, 4096, 8192]:
        tflops = bench(size)
        print(f"  {size}×{size}: {tflops:.2f} TFLOPS")
else:
    print(f"❌ FAILED - Error too high")
