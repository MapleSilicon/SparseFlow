# SparseFlow

SparseFlow is a commercial GPU inference product for reducing LLM inference cost
on NVIDIA GPUs with structured sparsity.

## Public Repo Scope

This public repository is intentionally limited to product-facing material:

- Public benchmark framing
- Evaluation guidance
- Commercial access details
- Repository metadata for buyers and evaluators

Core implementation, kernel code, runtime internals, compiler passes, and
deployment logic are kept private.

## Public Performance Summary

Current public benchmark framing for SparseFlow:

- `1.4x` average speedup on validated production benchmark shapes
- `1.6x-1.7x` peak gains on FFN-heavy inference paths
- `30-40%` potential inference cost reduction on a good workload fit
- Zero model changes required to start evaluating

These are buyer-facing benchmark summaries, not a full public source release of
the implementation.

## Hardware Validation Status

- Validated: `A100` (primary benchmark platform), `RTX 3090`
- In active validation: `RTX 4090`
- Architecturally supported directionally, but not yet publicly validated here:
  `H100`, additional `RTX 30/40` variants

## Evaluation Path

- Free benchmark review:
  Share representative model shapes, configs, or workload details and get a
  lightweight screening read on expected upside.
- Paid pilot:
  Validate SparseFlow against your workload and deployment path in a deeper
  engineering engagement.

## Public Docs

- [docs/PUBLIC_OVERVIEW.md](docs/PUBLIC_OVERVIEW.md)
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- [docs/COMMERCIAL_ACCESS.md](docs/COMMERCIAL_ACCESS.md)

## Commercial Access

For evaluation access, pilot discussions, or commercial conversations:

- Founder contact: `gourav.kumar@maplesilicon.co`
- General inquiries: `info@maplesilicon.co`
- Public website: [maplesilicon.co](https://maplesilicon.co)

## License

This public repository remains under the existing [MIT](LICENSE) license for the
materials published here. Commercial product access is handled separately.
