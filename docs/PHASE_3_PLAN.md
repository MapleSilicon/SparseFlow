# Phase-3 Technical Plan: GPU Kernel Specialization
## 12-Week Execution Plan (Measured, Bounded, Fundable)

---

## Goal (Single Sentence)

**Deliver a specialized GPU path where Sparse 2:4 beats Dense consistently on transformer-relevant shapes, with reproducible benchmarks and zero Phase-2 regressions.**

---

## Win Conditions

**Primary:** Sparse >= Dense at 2048³ consistently (≥1.2× speedup)

**Secondary:** Fused path (GEMM+bias+activation) shows clear advantage vs dense unfused baseline

---

## Non-Negotiable Rules

- ✅ No changes to Phase-2 runtime (except new kernel IDs)
- ✅ All Phase-3 behind feature flag: `SPARSEFLOW_ENABLE_PHASE3=1`
- ✅ CI remains stable (CPU-only)
- ✅ Every week: reproducible script + table entry

---

## Week-by-Week Breakdown

### Week 1: Baseline & Targets
**Goal:** Stop guessing, start measuring

**Deliverables:**
- `bench/bench_phase3_shapes.cpp`
- Fixed suite: 1024³, 2048³, 3072³, 4096³
- Transformer shapes: M=4096 N=4096 K=11008
- Measures: cuBLAS dense, cuSPARSELt sparse
- Reports: TFLOPS + time + variance (min/median/p95)

**Exit Criteria:**
```bash
./scripts/bench_phase3.sh  # → stable baseline table
```

**Why:** Reviewers need to trust the numbers

---

### Week 2: Measurement Hygiene
**Goal:** Make numbers real

**Work:**
- CUDA event timing (precise)
- Warmup control
- Stream sync policy
- Repeat counts (≥20 runs)
- Variance reporting
- JSON output: `--json out/results.json`

**Exit Criteria:**
Baseline variance < 3% for 2048³ over 20 runs

---

### Week 3: Identify the Bottleneck
**Goal:** Profile, don't guess

**Work:**
- Nsight Compute / Nsight Systems profiling
- Answer:
  - Memory bandwidth bound?
  - Format conversion overhead?
  - Poor occupancy?

**Deliverable:**
`docs/PHASE3_PROFILING_NOTES.md` with:
- Top 3 kernels by time
- Roofline reasoning
- Decision: What's the lever?

**Exit Criteria:**
Pick Phase-3 strategy:
- **(A) Fusion-first** (recommended: quickest win)
- **(B) Custom kernel-first** (harder, headline-worthy)
- **(C) cuSPARSELt optimization** (lowest risk, smaller gains)

---

### Week 4: Data Pipeline Production-Ready
**Goal:** Pack weights correctly

**Work:**
- Real 2:4 packing pathway
- Pack weights once
- Reuse packed weights
- Avoid "compress every time" in hot path

**Deliverable:**
`runtime/weights/packed_nm24.{h,cu}`

**Exit Criteria:**
Packed path works for 2048³ with correctness check

---

### Week 5: Fused Epilogue Baseline
**Goal:** Set up advantage zone

**Work:**
- Fused epilogue for dense: GEMM + bias + ReLU
- Simple custom CUDA epilogue kernel (post-GEMM)

**Exit Criteria:**
Benchmark both:
- Dense GEMM + epilogue
- Sparse GEMM + epilogue

**Why:** Fusion is where sparse wins

---

### Week 6: Phase-3 Runtime Integration
**Goal:** Clean extension of Phase-2

**Work:**
- New KernelID: `SPARSE_NM24_FUSED`
- Plug into existing dispatcher
- Behind feature flag

**Exit Criteria:**
Runtime selection includes Phase-3 kernel only when enabled

---

### Week 7: Custom Sparse Kernel (Correctness First)
**Goal:** Minimal working prototype

**Work:**
- Minimal CUDA kernel for 2:4 sparse GEMM
- One datatype (FP16)
- One layout
- One tile size
- Correct, not fast yet

**Exit Criteria:**
- Correctness passes (1024³, 2048³)
- Bench runs without crashes

---

### Week 8: Kernel Optimization Pass 1
**Goal:** Close the gap

**Work:**
- Improved tiling
- Shared memory optimization
- Vectorized loads
- Warp-level MMA (if feasible)

**Exit Criteria:**
Sparse TFLOPS at 2048³ improves +10-20%

---

### Week 9: Fusion into Sparse Path
**Goal:** Real win territory

**Work:**
- Fuse bias + activation into sparse kernel
- Reduce memory traffic
- Avoid writing intermediate C

**Exit Criteria:**
Fused sparse > unfused sparse (clearly)

---

### Week 10: Expand Shape Coverage
**Goal:** Stop overfitting

**Work:**
Validate:
- 2048³
- 4096³
- 2 transformer-like GEMMs

**Exit Criteria:**
Sparse wins on ≥1 real larger workload

---

### Week 11: Reliability Hardening
**Goal:** No production footguns

**Work:**
- Guardrails for unsupported shapes
- Fallback to dense
- Failure quarantine integration (Phase-2 behavior)
- "Known bad shapes" list

**Exit Criteria:**
- No crashes on invalid sizes
- All Phase-2 tests still pass

---

### Week 12: Release + Evidence Package
**Goal:** Reproducible proof

**Deliverables:**
- `PHASE_3_PREVIEW.md`
- Updated `CANONICAL_RESULTS.md` (Phase-3 section)
- `scripts/bench_phase3.sh` → markdown table + JSON
- Tag: `v0.3.0-preview`

**Exit Criteria:**
Reviewer can:
1. Clone repo (tag v0.3.0-preview)
2. Run one command
3. Get table
4. Reproduce results

---

## Milestone Gates (Weekly Tracking)

1. **Correctness:** Random input check vs dense (within tolerance)
2. **Stability:** 50 consecutive runs without errors
3. **Performance:** Median time improvement on target shapes
4. **Reproducibility:** Scripts run from clean clone

---

## Strategy Decision Required

**Choose ONE to start:**

### Option A: Fusion-First ⭐ (RECOMMENDED)
**Why:** Quickest path to "sparse wins"
**Risk:** Lower
**Timeline:** 8-10 weeks to measurable win
**Funding narrative:** "Production deployment ready"

### Option B: Custom Kernel-First
**Why:** Headline-worthy performance
**Risk:** Higher
**Timeline:** 10-12 weeks to measurable win
**Funding narrative:** "Technical leadership demonstration"

### Option C: cuSPARSELt Optimization
**Why:** Lowest risk, leverages NVIDIA work
**Risk:** Lowest
**Timeline:** 6-8 weeks to modest improvement
**Funding narrative:** "Incremental production gains"

---

## Budget Alignment

**Phase-3 (12 weeks):**
- Senior GPU Engineer: $60K (3 months)
- Cloud GPU credits (A100): $5K
- Development hardware: $5K
- **Total: $70K**

**Scales to:**
- 6 months (full Phase-3): $140K
- IRAP/NGen budget: $180K (includes overhead)

---

## Risk Mitigation

**Technical Risks:**
1. Sparse doesn't win → Mitigation: Fusion guarantees advantage zone
2. Hardware portability → Mitigation: Test on Ampere + Ada
3. Integration breaks Phase-2 → Mitigation: Feature flag + regression tests

**Schedule Risks:**
1. Week slippage → Mitigation: Weekly gates enforce progress
2. Scope creep → Mitigation: "One goal" rule strictly enforced

---

## Success Metrics

**Phase-3 Complete When:**
- [ ] Sparse ≥ Dense at 2048³ (3+ consecutive runs)
- [ ] Fused path shows >1.2× vs unfused dense
- [ ] Zero Phase-2 regressions
- [ ] Reproducible from tag v0.3.0-preview
- [ ] Published benchmark table (CANONICAL_RESULTS.md updated)

---

## Next Action Required

**Say: A, B, or C**

I will then provide:
- Exact folder structure
- Exact files to create
- `bench_phase3.sh` script (copy-paste ready)
- CMake hooks
- Initial benchmark harness code

**No fluff. Just execution.**

---

**Version:** 1.0  
**Status:** Ready for execution  
**Decision pending:** Strategy selection (A/B/C)
