# API Reference

The supported public surface is declared via Julia's `public` statement in
`src/SDPA.jl` (`solve!`, the readers/writer, `Parameters` + presets, the
core types, `PhaseValue`/`ExitReason`, and `Optimizer`); nothing is
auto-exported.

## Core Types

### `SDPAProblem{T<:Real}`

The SDP problem definition. Constructed by parsers (`read_dat_s`,
`read_dat`) or built incrementally.

**Key fields**:
- `m::Int` — number of constraints
- `nBlock::Int` — total number of blocks
- `blockStruct::Vector{Int}` — signed block sizes (negative = LP block, ≥2 = SDP block)
- `b::SDPAVector{T}` — RHS vector (length m)
- `C::SDPASparseLinearSpace{T}` — objective matrix
- `A::Vector{SDPASparseLinearSpace{T}}` — constraint matrices
- `SDP_nConstraint`, `SDP_constraint`, `SDP_blockIndex` — aggregate sparsity tables

### `Parameters{T<:Real}`

Solver parameters. Constructed via keyword arguments:

```julia
params = SDPA.Parameters{BigFloat}(
    precision=200,
    maxIteration=200,
    betaStar=BigFloat("0.1"),
    betaBar=BigFloat("0.3"),
    gammaStar=BigFloat("0.9"),
    setting=SDPA.DEFAULT,
)
```

**Presets** (mirror C++ `Parameter::setDefaultParameter`):
- `DEFAULT`: maxIter=200, β*=0.1, β̄=0.3, γ*=0.9, precision=200
- `UNSTABLE_BUT_FAST`: maxIter=100, β*=0.01, β̄=0.02, γ*=0.98, precision=100
- `STABLE_BUT_SLOW`: maxIter=1000, β*=0.2, β̄=0.4, γ*=0.5, precision=300

**Validators**: `epsilonStar > 0`, `lambdaStar > 0`, `omegaStar > 1`,
`0 < betaStar < 1`, `betaStar ≤ betaBar < 1`, `0 < gammaStar < 1`,
`epsilonDash > 0`, `precision > 0`.

### `SolutionLog{T<:Real}`

Returned by `solve!`. Contains the full solve trajectory.

**Key fields**:
- `iterations::Int` — final iteration count
- `terminal_phase::PhaseValue` — phase at termination (e.g., `pdOPT`)
- `trajectory::Vector{IterationRow{T}}` — per-iteration scalars
- `final_obj_primal::T` — final `⟨C, X⟩`
- `final_obj_dual::T` — final `⟨b, y⟩`
- `final_mu::T` — final complementarity μ
- `final_primal_res::T` — `‖b - ⟨A, X⟩‖∞`
- `final_dual_res::T` — `‖C - Σ y·A - Z‖∞`
- `xMat::SDPADenseLinearSpace{T}` — final primal iterate
- `yVec::SDPAVector{T}` — final dual iterate
- `zMat::SDPADenseLinearSpace{T}` — final dual slack
- `timings::NamedTuple` — `(mehrotra=..., eigen=..., residual=..., total=...)`
- `timed_out::Bool` — `true` when the wall-clock limit fired (legacy flag; equivalent to `exit_reason == TIME_LIMIT`)
- `exit_reason::ExitReason` — why the IPM loop terminated

### `PhaseValue`

Enum for solver phase:
`noINFO`, `pFEAS`, `dFEAS`, `pdFEAS`, `pdINF`, `pFEAS_dINF`,
`pINF_dFEAS`, `pdOPT`, `pUNBD`, `dUNBD`, `not_called`.

### `ExitReason`

Enum for why the IPM loop terminated (reported in
`SolutionLog.exit_reason`): `OPTIMAL_TERMINAL`, `MAX_ITERATION`,
`CHOLESKY_FAILURE`, `CANNOT_MOVE`, `TIME_LIMIT`, `EMPTY_PROBLEM`, `NONFINITE_STEP`.
`terminal_phase` alone cannot distinguish a healthy terminal phase from
one frozen mid-solve by a non-terminal exit (iteration budget, Cholesky
failure, cannot move, time limit); this enum can. The MOI layer consumes
it for `TerminationStatus` (`MAX_ITERATION` → `ITERATION_LIMIT`,
`CHOLESKY_FAILURE`/`CANNOT_MOVE` → `OTHER_LIMIT`, `NONFINITE_STEP` →
`NUMERICAL_ERROR`, `TIME_LIMIT` →
`TIME_LIMIT`). A failed candidate keeps the last accepted point. A strictly
interior tiny step is counted, and its diagnostics are recomputed before
reporting stagnation.

### `NewtonState{T<:Real}`

Internal state for one Newton iteration. Not typically accessed by
users, but available for advanced use:

- `bmat::SDPADenseMatrix{T}` — dense Schur complement matrix
- `sparse_bmat::Union{Nothing,SDPASparseMatrix{T}}` — profile-form sparse Schur
- `diagonalIndex::Vector{Int}` — column pointers for profile form
- `ordering::Vector{Int}` — AMD/METIS permutation
- `use_sparse::Bool` — dispatch flag
- Location-table arrays: `SDP_constraint1`, `SDP_constraint2`, `blockIndex1`, `blockIndex2`, `SDP_location_sparse_bMat`

## Core Functions

### `solve!`

```julia
solve!(problem::SDPAProblem{T}, params::Parameters{T};
       verbose::Bool=true, io::IO=stdout,
       reverse_primal_dual::Bool=true,
       force_sparse::Bool=false,
       init::Union{Nothing,SDPASolution{T}}=nothing) -> SolutionLog{T}
```

Solve the SDP problem. Returns a `SolutionLog`.

**Keyword arguments**:
- `verbose`: Print iteration log to `io`.
- `reverse_primal_dual`: Apply REVERSE_PRIMAL_DUAL convention (default
  `true` — matches C++ SDPA-GMP output; the solver negates C internally
  and displays values per the C++ convention). `false` gives standard-form
  `min` semantics with direct unbounded/infeasible detection. The MOI path
  always passes `false`.
- `force_sparse`: Force sparse Schur complement path even if
  `analyze_sparsity` selects dense.
- `init`: optional warm-start point (e.g. from `read_ini_s`); see
  "Warm start" below. Defaults to the cold start (X = Z = λI, y = 0).

### Warm start (`init`)

`solve!` accepts an optional `init::SDPASolution{T}` (e.g. from
`read_ini_s`) and warm-starts from that point: X/y/Z are used as-is and
the initial μ is set to ⟨X, Z⟩/nDim (mirroring C++
`AverageComplementarity::initialize`). The primal x⁰ need NOT be
feasible — the IPM is an infeasible-start method; only X⁰ and Z⁰ must
be strictly interior.

**Exporter contract** (the point must be strictly interior for the
Cholesky-based machinery; a shipped set of NSPS MOSEK export points was
found indefinite, which collapses the first step for any solver):

- LP diagonals: ≥ `max(1e-6, 1e-8·max|Z|)` (absolute floor 1e-6;
  a scale-aware variant is allowed).
- SDP blocks: eigmin ≥ 1e-8 (diagnosis number, absolute; a scale-aware
  `eigmin ≥ 1e-8·λmax(Z)` variant is also acceptable).
- A trivially safe point is the isotropic X = Z = λI for a moderate λ
  (e.g. 1e4, the cold-start default).

**ini-s layout** (SDPA-GMP convention, as `read_ini_s` parses it): line 1
carries the m-vector y (stored NEGATED — the file holds the
SCS/MOSEK-sign y), then 4-tuples `l i j value`: l = 1..nBlock labels
**Z** entries, nBlock+1..2·nBlock labels **X** entries (Z-first; this
differs from the old manual's X/Y 1/2 labeling). Missing entries stay
zero. The reader's fixed 1e-6 LP clip (and the εI SDP shifts) is a
backstop for near-optimal points, NOT a substitute for a valid interior
point.

### `read_dat_s`

```julia
read_dat_s(filename::AbstractString, T::Type{<:Real}=Float64) -> SDPAProblem{T}
```

Parse an SDPA sparse format (`.dat-s`) file.

### `read_dat`

```julia
read_dat(filename::AbstractString, T::Type{<:Real}=Float64) -> SDPAProblem{T}
```

Parse an SDPA dense format (`.dat`) file.

### `write_result`

```julia
write_result(io::IO, problem, params, trajectory, phase, iteration,
             mu, relgap, gap, digits, obj_p, obj_d, p_feas, d_feas,
             compute_time; kwargs...) -> Nothing
```

Write a `.result`-format file. T-aware formatting (BigFloat gets
40 significant digits, Float64 gets 16).

**SolveInfo-based signature** (for callers that already hold a
`SolveInfo`; see `src/io/write_result.jl`):
```julia
write_result(io, problem, solution, params, phase, solve_info,
             compute_time, iteration; kwargs...)
```

### `print_one_iteration`

```julia
print_one_iteration(io::IO, row::IterationRow{T}) -> Nothing
```

Print one iteration row of the progress table.

**SolveInfo-based signature** (see `src/io/write_result.jl`):
```julia
print_one_iteration(io, iter, mu, thetaP, thetaD, solve_info, alpha, beta)
```

### `analyze_sparsity`

```julia
analyze_sparsity(problem; method=:metis) -> (use_dense::Bool, ordering::Vector{Int})
```

Decide dense vs sparse path. `method` can be `:metis` (default) or `:amd`.

## MOI/JuMP Interface

### `Optimizer{T<:Real}`

```julia
opt = SDPA.Optimizer{BigFloat}(silent=true, precision=200)
```

**Keyword arguments**:
- `silent`: Suppress iteration output.
- `precision`: BigFloat precision in bits.
- `params`: Custom `Parameters{T}` (overrides `precision`).

**Custom accessors**:
```julia
MOI.get(opt, SDPA.PrimalSolutionMatrix())  # Vector{Matrix{T}}
MOI.get(opt, SDPA.DualSolutionVector())   # Vector{T}
MOI.get(opt, SDPA.DualSlackMatrix())       # Vector{Matrix{T}}
```

**Standard MOI attributes**: `TerminationStatus`, `ObjectiveValue`,
`DualObjectiveValue`, `VariablePrimal`, `ConstraintPrimal`,
`ConstraintDual`, `RawStatusString`, `RawOptimizerAttribute`,
`SolveTimeSec`.

### `RawOptimizerAttribute`

Set any parameter field by name:

```julia
MOI.set(opt, MOI.RawOptimizerAttribute("maxIteration"), 500)
MOI.set(opt, MOI.RawOptimizerAttribute("betaStar"), BigFloat("0.05"))
```

Validates through `Parameters{T}` constructors + validators.

## Utility Functions

### In-Place MPFR Wrappers (`src/mpfr_inplace.jl`)

Zero-allocation BigFloat arithmetic via direct MPFR ccalls. Available
as `SDPA.mp_*` functions:

| Function | Operation | Allocation |
|----------|-----------|------------|
| `mp_fma!(r,x,y,w)` | `r = x*y + w` (single rounding) | 0 |
| `mp_fms!(r,x,y,z)` | `r = x*y - z` (single rounding) | 0 |
| `mp_add!(r,a,b)` | `r = a + b` | 0 |
| `mp_sub!(r,a,b)` | `r = a - b` | 0 |
| `mp_mul!(r,a,b)` | `r = a * b` | 0 |
| `mp_div!(r,a,b)` | `r = a / b` | 0 |
| `mp_sqrt!(r,x)` | `r = sqrt(x)` | 0 |
| `mp_rec_sqrt!(r,x)` | `r = 1/sqrt(x)` | 0 |
| `mp_neg!(r,x)` | `r = -x` | 0 |
| `mp_set!(r,x)` | `r = x` (deep copy of value) | 0 |
| `mp_add_mul!(r,y,z)` | `r += y*z` | 0 |
| `mp_sub_mul!(r,y,z)` | `r -= y*z` | 0 |

All support full aliasing (e.g., `mp_fma!(s, v, G[r,c], s)` is legal).
MPFR must be built thread-safe (`mpfr_buildopt_tls_p() == 1`, verified
at module load).

### MPLAPACK Operations (`src/mplapack/`)

Arbitrary-precision BLAS/LAPACK ported from MPLAPACK:

#### Matrix Multiply (`Rgemm!`)

```julia
SDPA.Rgemm!("N", "N", m, n, k, alpha, A, B, beta, C)
# Computes C = alpha * A * B + beta * C
# transa/transb ∈ {"N", "T", "C"} (C = conjugate transpose = T for real)
```

11× faster than `LinearAlgebra.mul!` at n=100 BigFloat(200). Uses
`OhMyThreads.@tasks` for outer-loop parallelism.

#### Type-Based Dispatcher (`sdpa_mul!`)

```julia
SDPA.sdpa_mul!(C, A, B)        # BigFloat → Rgemm!, Float64 → mul!
SDPA.sdpa_mul!(C, A, B, α, β)  # with alpha/beta scaling
```

#### Cholesky Factorization (`Rpotf2!` / `Rpotrf!`)

```julia
SDPA.Rpotf2!("U", n, A)   # Unblocked Cholesky, returns info (0=success)
SDPA.Rpotrf!("U", n, A; nb=64)  # Blocked Cholesky, uses Rgemm!/Rtrsm!
```

#### Triangular Solve (`Rtrsm!`)

```julia
SDPA.Rtrsm!("L", "U", "T", "N", m, n, alpha, A, B)
# Solves op(A) * X = alpha * B, overwriting B
```

#### Level-1 BLAS (`Rblas`)

```julia
SDPA.Rdot(n, dx, incx, dy, incy)      # Dot product
SDPA.Raxpy!(n, da, dx, incx, dy, incy)  # dy += da * dx
SDPA.Rscal!(n, da, dx, incx)          # dx *= da
SDPA.Rcopy!(n, dx, incx, dy, incy)   # dy = dx
SDPA.Rgemv!(transa, m, n, alpha, A, lda, x, incx, beta, y, incy)
```

### `setprecision` for BigFloat

The solver stores `precision` in `Parameters` and applies it inside
`solve!` via `setprecision(BigFloat, n) do ... end`. Do NOT call
`setprecision(BigFloat, n)` globally — the 2-argument form is not
thread-safe in Julia < 1.12.
