# Benchmarks

SparseFlow is presented publicly as a benchmark-first inference product.

## Public Benchmark Summary

- `1.4x` average speedup on validated production benchmark shapes
- `1.6x-1.7x` peak gains on FFN-heavy inference paths
- `30-40%` potential inference cost reduction on a good workload fit
- Zero model changes required to start evaluating

## Validation Status

- Validated platforms: `A100`, `RTX 3090`
- In active validation: `RTX 4090`
- Not yet publicly validated in this repository: `H100`, additional `RTX 30/40`
  variants

## How To Read These Numbers

These are public benchmark summaries intended for product evaluation, not a full
public dump of benchmark internals.

Publicly visible material should stay at the level of:

- Supported workload categories
- Validated hardware status
- Reported speedup ranges
- Evaluation scope and current limits

Keep the following private:

- Kernel-level traces
- Internal tuning thresholds
- Deployment-specific operating points
- Proprietary validation fixtures
- Customer-specific workload details
