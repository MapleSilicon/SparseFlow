# SparseFlow API Reference

## Python API

### Core Operations

#### `sparseflow.check_sparse_support()`
Check if current GPU supports 2:4 sparse operations.

**Returns:**
- `(bool, str)`: (supported, message)

**Example:**
```python
import sparseflow as sf

supported, msg = sf.check_sparse_support()
print(msg)
# "✅ A100 (SM80) supports 2:4 sparse"
```

---

#### `sparseflow.prune_2_4(tensor, method='magnitude')`
Prune dense tensor to 2:4 sparsity pattern.

**Args:**
- `tensor` (torch.Tensor): Dense input tensor
- `method` (str): Pruning method ('magnitude' or 'random')

**Returns:**
- `torch.Tensor`: Pruned tensor with 2:4 pattern

**Example:**
```python
weight = torch.randn(4096, 4096)
weight_sparse = sf.prune_2_4(weight, method='magnitude')
```

---

#### `sparseflow.validate_2_4(tensor)`
Validate tensor has valid 2:4 sparsity pattern.

**Args:**
- `tensor` (torch.Tensor): Tensor to validate

**Returns:**
- `bool`: True if valid 2:4 pattern

**Example:**
```python
is_valid = sf.validate_2_4(weight_sparse)
assert is_valid, "Invalid 2:4 pattern"
```

---

#### `sparseflow.compress_2_4(tensor)`
Compress tensor with 2:4 pattern to compressed format.

**Args:**
- `tensor` (torch.Tensor): Tensor with valid 2:4 pattern

**Returns:**
- `torch.Tensor`: Compressed tensor (50% size)

**Example:**
```python
compressed = sf.compress_2_4(weight_sparse)
print(f"Size: {compressed.numel()} (50% of original)")
```

---

### Neural Network Modules

#### `sparseflow.SparseLinear`
Drop-in replacement for `nn.Linear` with 2:4 sparsity.

**Constructor:**
```python
SparseLinear(weight_compressed, metadata=None, bias=None)
```

**Class Method: `from_dense()`**
```python
@staticmethod
def from_dense(
    dense_linear: nn.Linear,
    method: str = "magnitude",
    validate: bool = True,
    return_diff: bool = False
) -> Union[SparseLinear, Tuple[SparseLinear, Dict]]
```

**Args:**
- `dense_linear`: Original nn.Linear module
- `method`: Pruning method ('magnitude')
- `validate`: Check 2:4 pattern compliance
- `return_diff`: Return accuracy impact metrics

**Returns:**
- `SparseLinear` or `(SparseLinear, dict)` if return_diff=True

**Example:**
```python
import torch.nn as nn
import sparseflow as sf

# Create dense layer
dense = nn.Linear(4096, 4096).cuda().half()

# Convert to sparse
sparse, diff = sf.SparseLinear.from_dense(
    dense,
    method="magnitude",
    return_diff=True
)

print(f"Max error: {diff['max_error']:.6f}")
print(f"Mean error: {diff['mean_error']:.6f}")

# Use like normal Linear
x = torch.randn(128, 4096, device='cuda', dtype=torch.float16)
y = sparse(x)  # 2× faster
```

---

## C API

### Context Management

#### `sparseflow_create_context()`
```c
SparseFlowStatus sparseflow_create_context(
    SparseFlowContext* ctx,
    int device_id
);
```

Create context for specific GPU.

**Example:**
```c
SparseFlowContext ctx;
SparseFlowStatus status = sparseflow_create_context(&ctx, 0);
if (status != SPARSEFLOW_SUCCESS) {
    fprintf(stderr, "Error: %s\n", sparseflow_get_error_string(status));
}
```

---

#### `sparseflow_destroy_context()`
```c
SparseFlowStatus sparseflow_destroy_context(SparseFlowContext ctx);
```

Destroy context and free resources.

---

### Version & Compatibility

#### `sparseflow_get_version()`
```c
SparseFlowVersion sparseflow_get_version();
```

Get library version.

**Example:**
```c
SparseFlowVersion v = sparseflow_get_version();
printf("Version: %d.%d.%d\n", v.major, v.minor, v.patch);
```

---

#### `sparseflow_is_abi_compatible()`
```c
int sparseflow_is_abi_compatible(int major, int minor);
```

Check ABI compatibility. Returns 1 if compatible, 0 otherwise.

---

## CLI Tools

### sparseflow-audit

Analyze deployment costs and savings.

**Usage:**
```bash
sparseflow-audit --model MODEL --qps QPS [--gpu GPU]
```

**Options:**
- `--model`: Model architecture (llama-7b, gpt2-xl, etc.)
- `--qps`: Target queries per second
- `--gpu`: GPU type (default: a100-80gb)

**Example:**
```bash
sparseflow-audit --model llama-7b --qps 1000
```

---

### sparseflow-convert

Convert models to SparseFlow format.

**Usage:**
```bash
sparseflow-convert --input PATH --output PATH [OPTIONS]
```

**Options:**
- `--input`: Input model path
- `--output`: Output sparse model path
- `--method`: Pruning method (default: magnitude)
- `--validate`: Validate 2:4 patterns

**Example:**
```bash
sparseflow-convert \
  --input model.pt \
  --output model.sf \
  --validate
```

---

### sparseflow-benchmark

Benchmark performance on current hardware.

**Usage:**
```bash
sparseflow-benchmark [OPTIONS]
```

**Options:**
- `--size`: Layer size (default: 4096x4096)
- `--batch-size`: Batch size (default: 128)
- `--iterations`: Number of iterations (default: 100)
- `--quick`: Quick test mode

**Example:**
```bash
sparseflow-benchmark --size 8192x8192 --iterations 100
```

---

## Error Handling

### Python
```python
try:
    sparse = sf.SparseLinear.from_dense(dense)
except RuntimeError as e:
    print(f"Error: {e}")
```

### C
```c
SparseFlowStatus status = sparseflow_create_context(&ctx, 0);
if (status != SPARSEFLOW_SUCCESS) {
    const char* error = sparseflow_get_last_error();
    fprintf(stderr, "Error: %s\n", error);
}
```

---

## Best Practices

### 1. Always Check GPU Support
```python
supported, msg = sf.check_sparse_support()
if not supported:
    raise RuntimeError("2:4 sparse not supported")
```

### 2. Validate Accuracy Impact
```python
sparse, diff = sf.SparseLinear.from_dense(dense, return_diff=True)
if diff['max_error'] > 0.01:
    print(f"Warning: High accuracy impact: {diff['max_error']}")
```

### 3. Use FP16 for Best Performance
```python
model = model.half()  # Convert to FP16
```

### 4. Batch for Performance
```python
# Small batches don't show speedup
x = torch.randn(1, 4096)      # Slow
x = torch.randn(128, 4096)    # Fast (2× speedup)
```
