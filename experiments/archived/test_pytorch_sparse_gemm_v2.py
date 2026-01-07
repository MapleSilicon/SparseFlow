import torch
import time

print("="*60)
print("PYTORCH 2:4 SPARSE TENSOR CORE TEST")
print("="*60)

# First, understand the correct API
print("\nExploring sparse API...")
print(torch._sparse_semi_structured_apply.__doc__)

# Try simpler approach - use the high-level API
try:
    # Create test matrix
    M, K, N = 128, 128, 128
    A_dense = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')
    
    # Apply 2:4 pattern
    A_sparse_dense = A_dense.clone()
    for i in range(0, K, 4):
        group = A_sparse_dense[:, i:i+4].abs()
        _, indices = torch.topk(group, k=2, dim=1, largest=False)
        for row in range(M):
            A_sparse_dense[row, i + indices[row, 0]] = 0
            A_sparse_dense[row, i + indices[row, 1]] = 0
    
    # Try different compression approach
    # The API might need metadata tensor
    print("\nTrying to compress matrix...")
    
    # Check if we can use cusparse directly
    if hasattr(torch.sparse, 'semi_structured'):
        print("Found torch.sparse.semi_structured!")
    
    # Try the _cslt functions which might be easier
    print("\nAvailable _cslt functions:")
    cslt_funcs = [attr for attr in dir(torch) if '_cslt' in attr]
    for func in cslt_funcs:
        print(f"  {func}")
        if hasattr(getattr(torch, func), '__doc__'):
            doc = getattr(torch, func).__doc__
            if doc:
                print(f"    {doc.split(chr(10))[0]}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Try the simplest possible test
print("\n" + "="*60)
print("Attempting basic sparse matmul...")

try:
    # Maybe there's a simpler function
    M = 16
    A = torch.randn(M, M, dtype=torch.float16, device='cuda')
    B = torch.randn(M, M, dtype=torch.float16, device='cuda')
    
    # Check what parameters _cslt_sparse_mm needs
    import inspect
    if hasattr(torch, '_cslt_sparse_mm'):
        sig = inspect.signature(torch._cslt_sparse_mm)
        print(f"_cslt_sparse_mm signature: {sig}")
    
except Exception as e:
    print(f"Error: {e}")
