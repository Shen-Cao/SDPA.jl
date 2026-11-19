# SDPA.jl

Pure-Julia arbitrary-precision semidefinite programming (SDP) solver.

SDPA.jl implements the primal-dual Mehrotra predictor-corrector interior-point
method with Monteiro-Zhang A5 (Jordan-product) symmetrization, matching the
algorithm of SDPA-GMP but with Julia `BigFloat` (MPFR) instead of GMP extras.
This gives true thread-safe arbitrary precision — something the C++ SDPA-GMP
cannot achieve because GMP's global allocator breaks under threads.

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, first solve |
| [Examples](docs/examples.md) | JuMP, MOI, Convex.jl, raw API |
| [Architecture](docs/architecture.md) | Solver internals, algorithm overview |
| [API Reference](docs/api.md) | Public types and functions |
| [Sparse Path](docs/sparse-path.md) | Sparse Schur complement for large problems |
| [Threading](docs/threading.md) | Multithreading model and bit-identity |
| [Performance](docs/performance.md) | Cost factors, rejected hypotheses, remaining gaps |
| [Roadmap](docs/roadmap.md) | Deferred findings and evaluated directions |

## Quick Start

```julia
using SDPA

problem = SDPA.read_dat_s("example.dat-s", BigFloat)
params = SDPA.Parameters{BigFloat}(precision=200)
log = SDPA.solve!(problem, params; verbose=true)

println("Phase: $(log.terminal_phase), Obj: $(log.final_obj_primal)")
```

Or via JuMP:

```julia
using JuMP, SDPA
model = Model(() -> SDPA.Optimizer{Float64}(silent=true))
@variable(model, X[1:2, 1:2], PSD)
@constraint(model, X[1, 1] + X[2, 2] == 1)
@objective(model, Min, X[1, 1] + X[2, 2])
optimize!(model)
```

JuMP emits `Float64` coefficients, so JuMP models use
`SDPA.Optimizer{Float64}`. For arbitrary precision via JuMP, use a model
whose coefficient type matches the optimizer (e.g.
`JuMP.GenericModel{BigFloat}(() -> SDPA.Optimizer{BigFloat}(precision=200))`),
or drive `SDPA.Optimizer{BigFloat}` directly through MOI.

## Requirements

- **Julia ≥ 1.12** (required for thread-safe `setprecision(BigFloat, n) do ... end`).
- Recommended `JULIA_NUM_THREADS=auto`. BLAS thread policy is set
  conditionally inside `solve!` based on numeric type: BigFloat → 1 BLAS
  thread (avoids oversubscription with Julia threads), Float64 →
  `Threads.nthreads()` (LAPACK parallelism). The policy is restored to the
  previous count on exit. Because the process-global policy is installed
  for the whole solve, the lock serializes the entire solve: concurrent
  `solve!` calls run one at a time (independent problems are not solved
  concurrently within one process). `solve!` does not mutate the
  caller's problem: `reverse_primal_dual=true` solves an internal copy of
  the problem with a negated `C`.

## Features

- **Arbitrary precision**: Solve at any BigFloat precision (default 200-bit).
- **In-place MPFR**: Zero-allocation `BigFloat` arithmetic via direct
  `mpfr_fma`/`mpfr_mul` ccalls (wrappers at `src/mpfr_inplace.jl`).
- **MPLAPACK port**: `Rgemm!` (matrix multiply), `Rpotf2!`/`Rpotrf!`
  (Cholesky), `Rtrsm!` (triangular solve), `Rblas` (Level-1 BLAS) —
  ported from MPLAPACK with OhMyThreads `@tasks` parallelism.
- **Sparse path**: Profile-form sparse Cholesky for large problems (AMD/METIS
  ordering). Skyline density check auto-selects dense vs sparse path.
  Threaded block-profile Cholesky + threaded assembly (bit-identical to
  serial at any thread count); compact LP storage keeps memory ~C++-level.
  **Benchmark status (NSPS, 256-bit)**: both the dense and forced-sparse
  paths converge with C++-parity iteration counts and optima, and the
  forced-sparse path runs ~1.6-1.9× faster per iteration than dense and
  single-threaded C++ — see [docs/performance.md](docs/performance.md)
  and benchmark/nsps/README.md.
- **Thread-safe**: Bit-identical results at any thread count.
- **MOI/JuMP integration**: Full MathOptInterface surface + Convex.jl support.
- **C++ reference match**: verified at `rtol=1e-30` for `BigFloat(200)` on
  the small fixture suite (example1/arch0/control5/minimal_sdp_lp/block_sparse);
  Float64 and NSPS-class problems are compared at looser tolerances
  (1e-7..1e-5 per the tests).

## Environment Variables

- `SDPA_ORDERING`: ordering routing for the block-level graph — `auto`
  (default; routes on position-level closed-profile density) or `block`.
  Invalid values error.
- `SDPA_BLOCK_ORDERING=1`: use the block-level ordering (C++-comparable,
  SPOOLES-like) instead of position-level METIS. `SDPA_ORDERING` takes
  precedence if both are set. Invalid values error.
- `SDPA_FORCE_SPARSE_PROFILE=1`: bench-only knob that bypasses the density
  gate and forces the sparse profile-form path (use only with the
  block-level profile; a truncated profile diverges). Invalid values warn.

See [docs/sparse-path.md](docs/sparse-path.md) for the ordering rules and
the measured trade-offs.

## Time limit

The MOI path supports `MOI.TimeLimitSec` (seconds; default `nothing` = no
limit). The raw `solve!` API takes the `time_limit_sec` keyword.

## Testing

Run the suite with:

```bash
JULIA_NUM_THREADS=4 julia --project=. -e 'using Pkg; Pkg.test()'
```

The BigFloat(256) acceptance tests (control5, use_formula_per_block; ~2 h each) are
skipped by default; set `SDPA_RUN_SLOW_TESTS=1` to enable them.

## License

GPL-2.0-or-later. See `LICENSE`.
