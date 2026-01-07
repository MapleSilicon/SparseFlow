import torch
import torch.nn.utils.prune as prune

print("="*60)
print("EXPLORING 2:4 SPARSE FORMAT")
print("="*60)

# Create a small dense matrix
dense = torch.randn(16, 16, dtype=torch.float16, device='cuda')

# PyTorch has built-in 2:4 pruning
from torch.ao.pruning import WeightNormSparsifier
sparsifier = WeightNormSparsifier(
    sparsity_level=0.5,  # 50% sparse
    sparse_block_shape=(1, 4),  # 2:4 pattern
    zeros_per_block=2
)

# Apply 2:4 pattern
# Note: PyTorch doesn't directly expose 2:4, we'll need to implement

# For now, let's understand the pattern
def create_2_4_pattern(tensor):
    """Manually create 2:4 sparse pattern"""
    out = tensor.clone()
    for i in range(0, tensor.shape[1], 4):
        # In each group of 4, zero out 2 smallest values
        group = out[:, i:i+4]
        _, indices = torch.topk(torch.abs(group), k=2, dim=1, largest=False)
        for row in range(out.shape[0]):
            out[row, i + indices[row, 0]] = 0
            out[row, i + indices[row, 1]] = 0
    return out

sparse_24 = create_2_4_pattern(dense)

print(f"\nDense sample (first row):")
print(dense[0, :8])
print(f"\n2:4 Sparse sample (first row):")
print(sparse_24[0, :8])

# Count zeros
total = sparse_24.numel()
zeros = (sparse_24 == 0).sum().item()
print(f"\nSparsity: {zeros}/{total} = {100*zeros/total:.1f}%")
print(f"Expected: 50%")

# Verify 2:4 pattern
def verify_2_4(tensor):
    for i in range(0, tensor.shape[1], 4):
        group = tensor[:, i:i+4]
        zeros_per_group = (group == 0).sum(dim=1)
        if not torch.all(zeros_per_group == 2):
            return False
    return True

print(f"\nValid 2:4 pattern: {verify_2_4(sparse_24)}")
