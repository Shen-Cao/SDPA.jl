# Getting Started

## Requirements

- **Julia ≥ 1.12** (required for thread-safe `setprecision(BigFloat, n) do ... end`).
- Recommended `JULIA_NUM_THREADS=auto`. The BLAS thread policy is set
  inside `solve!`'s `_solve_inner!` based on numeric type: BigFloat → 1
  BLAS thread (BigFloat arithmetic doesn't use BLAS; Julia threads do the
  parallel work), Float64 → `Threads.nthreads()` (LAPACK parallelism),
  restored to the previous count on exit.

## Installation

```julia
using Pkg
Pkg.add("SDPA")
```

For development:

```bash
git clone <the repository URL>
cd SDPA.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick Start

### Raw API (simplest)

```julia
using SDPA

# Read a problem in .dat-s format.
problem = SDPA.read_dat_s("example.dat-s", BigFloat)

# Configure solver parameters.
params = SDPA.Parameters{BigFloat}(precision=200, maxIteration=200)

# Solve.
log = SDPA.solve!(problem, params; verbose=true)

println("Phase:        $(log.terminal_phase)")
println("Obj primal:   $(log.final_obj_primal)")
println("Obj dual:     $(log.final_obj_dual)")
println("Primal resid: $(log.final_primal_res)")
println("Dual resid:   $(log.final_dual_res)")
println("Iterations:   $(log.iterations)")
println("Timings:      $(log.timings)")
```

### Via JuMP

```julia
using JuMP, SDPA

model = Model(() -> SDPA.Optimizer{BigFloat}(silent=true, precision=200))
@variable(model, X[1:2, 1:2], PSD)
@constraint(model, X[1,1] + X[2,2] == 1)
@objective(model, Min, X[1,1] + X[2,2])
optimize!(model)

println("Status:   $(termination_status(model))")
println("Objective: $(objective_value(model))")
```

### Via MathOptInterface (for BigFloat precision control)

JuMP macros generate `Float64` coefficients even for `BigFloat` optimizers.
For full arbitrary-precision problems, use MOI directly:

```julia
using MathOptInterface
const MOI = MathOptInterface
using SDPA

T = BigFloat
src = MOI.Utilities.Model{T}()
# Build the problem with T coefficients...
opt = SDPA.Optimizer{T}(silent=true, precision=200)
MOI.optimize!(opt, src)
```

## File Formats

SDPA.jl reads the standard SDPA sparse format (`.dat-s`) and dense
format (`.dat`). See the [SDPA format
specification](http://plato.asu.edu/ftp/sdpa_format.txt).

Result output matches the C++ `.result` schema via `SDPA.write_result`.

## Choosing Dense vs Sparse Path

The solver automatically selects dense or sparse Schur complement
assembly via `SDPA.analyze_sparsity`. Thresholds:

- `m < 100` constraints → dense
- `nBlock < 5` → dense
- Aggregate density > 25% → dense
- Otherwise → sparse (AMD ordering)

To force the sparse path (useful for benchmarking or when the
auto-selector picks dense):

```julia
log = SDPA.solve!(problem, params; force_sparse=true)
```
