# Contributing to SparseFlow

## 🎯 Mission: Efficiency-First GPU Computing

SparseFlow prioritizes **measured efficiency gains** and **correctness guarantees** over feature velocity. Every contribution must advance both performance and reliability.

## 🔬 Contribution Standards

### Performance Contributions

#### Benchmark Methodology Requirements
All performance claims must include:

1. **Energy Measurement**
   - NVML power readings over full benchmark duration
   - pJ/FLOP calculations with methodology
   - Idle power subtraction documented
   - Temperature and throttling state verification

2. **Correctness Validation** 
   - Mathematical verification against reference implementation
   - Maximum relative error < 1e-5 for FP32 accumulation
   - All matrix sizes from 512² to 8192² tested
   - Edge cases (non-power-of-2, rectangular) validated

3. **Resource Analysis**
   - ptxas register/shared memory utilization
   - Occupancy analysis with justification
   - Memory bandwidth utilization measurements
   - L2 cache hit rate profiling

#### No Performance Claims Without:
- [ ] CUDA event timing (minimum 10 iterations)
- [ ] Energy counter validation
- [ ] Correctness verification script
- [ ] Comparative analysis vs baseline

### Code Quality Standards

#### Kernel Implementations
```cuda
// Required: Explicit precision and accumulation
__global__ void kernel_name(
    const half* A,     // Input precision documented
    const half* B, 
    float* C,          // FP32 accumulation explicit
    int M, int N, int K
) {
    // Required: Resource usage comment
    // Registers: XX, Shared: XXXX bytes, Occupancy: X blocks/SM
}
```

## 🚫 What We Don't Accept

### Benchmark PRs Without Methodology
- No theoretical peak % claims without measurement
- No "faster than cuBLAS" without identical precision/fusion
- No isolated microbenchmarks without end-to-end validation
- No performance claims on synthetic/favorable data only

### Feature Creep
- No new APIs without proven performance necessity
- No framework integrations before kernel stability
- No multi-GPU before single-GPU optimization complete
- No new sparsity patterns before N:M mastery

## 🎯 Contribution Priorities

### High Priority
1. **N:M Sparse Pattern Optimization** - 2:4, 4:8, 8:16 patterns
2. **Energy Efficiency Improvements** - Lower pJ/FLOP at same performance
3. **Correctness Infrastructure** - Better verification tools
4. **Architecture Portability** - Volta, Turing, Ampere, Hopper support

### Low Priority (Blocked Until Core Stable)
1. **Framework Integration** - PyTorch/TensorFlow bindings
2. **Multi-GPU Scaling** - Distributed implementations  
3. **Custom Sparsity Patterns** - Beyond structured N:M
4. **CPU Fallback Implementations** - Non-GPU code paths

---

*Each release prioritizes measured efficiency gains and correctness guarantees over feature velocity.*
