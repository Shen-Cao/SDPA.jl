# SDPA.jl Documentation

**SDPA.jl** is a pure-Julia arbitrary-precision semidefinite programming (SDP)
solver. It implements the primal-dual Mehrotra predictor-corrector
interior-point method with Monteiro–Zhang A5 (Jordan-product)
symmetrization, matching the algorithm of SDPA-GMP but using Julia
`BigFloat` (MPFR) instead of GMP extras.

This gives true thread-safe arbitrary precision — something the C++
SDPA-GMP cannot achieve because GMP's global allocator breaks under
threads.

## Table of Contents

1. [Getting Started](getting-started.md) — Installation, first solve
2. [Examples](examples.md) — JuMP, MOI, Convex.jl, raw API
3. [Architecture](architecture.md) — Solver internals, algorithm overview
4. [API Reference](api.md) — Public types and functions
5. [Sparse Path](sparse-path.md) — Sparse Schur complement for large problems
6. [Threading](threading.md) — Multithreading model and bit-identity
7. [Performance](performance.md) — Cost factors, rejected hypotheses, remaining gaps
8. [Roadmap](roadmap.md) — Deferred findings and evaluated directions

## Quick Facts

| Feature | Status |
|---------|--------|
| Precision | Arbitrary (BigFloat, default 200-bit, matching C++ SDPA-GMP) |
| Problem types | SDP + LP (no SOCP) |
| Sparse path | Profile-form sparse Cholesky, AMD/METIS ordering |
| Threading | OhMyThreads `@tasks`, bit-identical across thread counts |
| Interfaces | MOI/JuMP, Convex.jl, raw `solve!` |
| Reference | C++ SDPA-GMP (`REVERSE_PRIMAL_DUAL=1`) |
| License | GPL-2.0-or-later |

## Key Design Decisions

- **BigFloat over GMP**: Julia's `BigFloat` wraps MPFR, which is
  thread-safe. C++ SDPA-GMP uses GMP's `mpf_class`, whose global
  allocator is not thread-safe — this limits C++ SDPA-GMP to
  single-threaded operation.
- **Profile-form sparse Cholesky**: No Julia package supports BigFloat
  sparse Cholesky (CHOLMOD/SuiteSparse are Float64-only). SDPA.jl
  implements its own left-looking profile-form Cholesky, matching the
  C++ SDPA-GMP data layout.
- **REVERSE_PRIMAL_DUAL convention**: The C++ reference is compiled with
  `REVERSE_PRIMAL_DUAL=1`, which swaps primal/dual labels and negates
  objectives. SDPA.jl replicates this (the `solve!` default) to match
  reference outputs byte-for-byte; the MOI path instead uses
  `reverse_primal_dual=false` (standard-form semantics). See
  [architecture.md](architecture.md) for the limitation.
