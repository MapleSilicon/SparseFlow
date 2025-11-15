# SparseFlow: sparseflow-sparse-compute Pass

This pass lowers `linalg.matmul` with `sparseflow.nm` and `sparseflow.mask`
attributes into explicit `scf.for` loop nests implementing a 2:4 style sparsity
pattern on the K dimension.

## Pipeline

Typical pipeline:

```bash
mlir-opt-19 -load-pass-plugin ./compiler/build/passes/libSparseFlowPasses.so \
  -pass-pipeline='builtin.module(func.func(
    sparseflow-annotate-nm{n=2 m=4},
    sparseflow-verify-pattern,
    sparseflow-generate-mask,
    sparseflow-sparse-compute
  ))' \
  test/simple-test.mlir
