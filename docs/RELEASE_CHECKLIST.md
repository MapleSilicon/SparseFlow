# Release Checklist (SparseFlow)

## v0.3-alpha
- [x] Tag created: `v0.3-alpha`
- [x] `mlir-opt-19 --load-pass-plugin=... --help-list | grep sparseflow` works
- [x] GPU rewrite pass runs on test file without verifier errors
- [x] Full alpha pipeline runs:
      `sparseflow-spa -> sparseflow-rewrite-matmul -> sparseflow-gpu-rewrite`
- [ ] Release notes published on GitHub

## Next: v0.3-beta
- [ ] GPU pass consumes func.call (no more %cst return)
- [ ] Kernel ABI defined with real arguments
- [ ] Bufferization pipeline stabilized
