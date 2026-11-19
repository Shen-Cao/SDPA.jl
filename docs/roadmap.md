# Roadmap

Deferred findings and evaluated-but-not-taken improvement directions.
None of this is scheduled work; each entry records the current state, the
known future fix, and the acceptance bar for landing it.

## Dense-assembly workspace churn (deferred finding)

`threaded_compute_bmat_dense!` currently acquires a per-block per-task
workspace pool on every `make_bmat!` call, churning 2·bs² BigFloats per
call. The churn is correctness-neutral and allocation gates pass with
3.7-8.6× headroom.

Two candidate fixes:

1. **TLV pools keyed by (thread, block-size)** living across regions —
   removes the per-call acquisition while keeping per-task isolation.
2. **Pair-ownership restructuring** — each (i,j) constraint pair gets a
   single owning task across all blocks. This also removes the serial
   block-order requirement, but it is a real design change.

**Acceptance criteria**: extend the multi-block test fixture to cover the
restructured ownership and prove bit-identity against the current code
before landing.

## GMP direct bindings (evaluated, deliberately not taken)

A Julia type backed by GMP's `mpf_t` (bypassing MPFR's correct rounding)
was assessed in detail (see the GMP-vs-MPFR table in
[performance.md](performance.md)):

- **Feasibility is good**: `libgmp` is already loadable from Julia (MPFR
  links it), the mpf_t C API ccalls cleanly, and the repo's containers
  are parametric over `T<:Real` so they would host an `Mpf` type
  unchanged.
- **The work**: a new mutable `Mpf` struct wrapping mpf_t with a
  finalizer (`mpf_clear`), ~20 ccall wrappers (add/sub/mul/div/sqrt/set/
  cmp/...), conversions and promotion rules, plus overloads for every
  `mp_*!` kernel and every `where {T<:BigFloat}` signature in the hot
  path (11 files), plus an `Mpf` variant of the test suite and goldens.
- **The cost that kills it**: GMP has no fused ops — `mp_fma!` and
  `mp_rec_sqrt!` have no mpf analog, so kernels revert to mul+add
  (2 roundings) and the codebase loses the single-rounding property.
- **Measured benefit**: ~1.5× on the dominant FMA op, i.e. **1.3-1.5×
  end-to-end** on arithmetic-bound iterations — not the ~3× once
  documented.

Standing decision: **let the mpf_t gap go.** The same factor is
available from threading (already working, bit-identical) and ordering
(`SDPA_ORDERING`); revisit only if a use case makes the 1.3-1.5×
decisive.

## Long-term structural work

- **Port SPOOLES-style ordering to Julia**: would help problems with
  genuine block-level sparsity (not NSPS, whose block graph is complete).
- **Supernodal/blocked sparse Cholesky**: potential cache-efficiency win
  for large profiles — **with a caveat**: for BigFloat the arithmetic is
  compute-bound (see "Cache blocking" in [performance.md](performance.md)),
  so cache-oriented tricks (blocking, supernodes) are unlikely to help
  the numeric factorization the way they do for Float64; ordering still
  matters for the structural side.

## Small parallelization candidate

`multi_plus_to_a!` (`src/types/SDPAProblem.jl`) is deliberately serial:
threading it needs per-task `retMat` buffers plus a reduction, and the
block count is small. If profiling ever shows it hot (dual-residual
computation and the corrector's DzMat assembly), it is a candidate for
per-task buffers + sequential reduction — the same pattern the sparse
assembly already uses for bit-identical accumulation.

## Performance gates in the test suite

The suite's performance assertions are machine-relative by nature and
live behind an opt-in switch (`SDPA_PERF_GATES=1`); the rationale and
calibration context are in [performance.md](performance.md) ("Performance
gates are machine-relative").
