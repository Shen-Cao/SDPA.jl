# Examples

## Example 1: Solve from a `.dat-s` file (raw API)

```julia
using SDPA

problem = SDPA.read_dat_s("example1.dat-s", BigFloat)
params = SDPA.Parameters{BigFloat}(precision=200, maxIteration=200)

log = SDPA.solve!(problem, params; verbose=true)

println("Phase:         $(log.terminal_phase)")
println("Primal obj:    $(log.final_obj_primal)")
println("Dual obj:      $(log.final_obj_dual)")
println("Primal resid:  $(log.final_primal_res)")
println("Dual resid:    $(log.final_dual_res)")
println("Iterations:    $(log.iterations)")
println("Wall time:     $(log.timings.total) sec")
```

## Example 2: JuMP with SDP constraints

```julia
using JuMP, SDPA

model = Model(() -> SDPA.Optimizer{Float64}(silent=true))
@variable(model, X[1:3, 1:3], PSD)
@constraint(model, sum(X[i,i] for i in 1:3) == 1)
@constraint(model, X[1,2] == 0.5)
@objective(model, Min, X[1,1] + 2*X[2,2] + 3*X[3,3])
optimize!(model)

@assert termination_status(model) == MOI.OPTIMAL
println("Objective: $(objective_value(model))")
```

## Example 3: BigFloat precision via MOI

JuMP macros produce `Float64` coefficients. For full arbitrary
precision, use MOI directly:

```julia
using MathOptInterface
const MOI = MathOptInterface
using SDPA

T = BigFloat
src = MOI.Utilities.Model{T}()
x, _ = MOI.add_constrained_variables(src, MOI.PositiveSemidefiniteConeTriangle(2))
MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([one(T), one(T)], [x[1], x[3]]), zero(T)))
eq_ci = MOI.add_constraint(src,
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([one(T), one(T)], [x[1], x[3]]), zero(T)),
        MOI.EqualTo(one(T)))

opt = SDPA.Optimizer{T}(silent=true, precision=200)
MOI.set(opt, MOI.RawOptimizerAttribute("maxIteration"), 100)
MOI.optimize!(opt, src)

# Results at full 200-bit precision.
y = MOI.get(opt, MOI.ConstraintDual(), eq_ci)
primal_mats = MOI.get(opt, SDPA.PrimalSolutionMatrix())
```

## Example 4: Convex.jl with BigFloat

```julia
using Convex, SDPA

setprecision(256)
x = Variable()
y = Variable()
p = minimize(x + y, [x + y == 1, x >= 0, y >= 0])
solve!(p, SDPA.Optimizer{BigFloat}(silent=true, precision=200);
       numeric_type=BigFloat)

println("Objective: $(p.optval)")
```

The `numeric_type=BigFloat` kwarg is critical — it tells Convex to
build `BigFloat`-typed MOI constraints that match the optimizer's
`BigFloat` precision.

## Example 5: Sparse path with theta-sized problems

```julia
using SDPA

# theta5: m=3028, one 250×250 SDP block, density≈0.0003.
problem = SDPA.read_dat_s("theta5.dat-s", Float64)
params = SDPA.Parameters{Float64}(maxIteration=50)

# Auto-select: analyze_sparsity picks sparse for theta5.
log = SDPA.solve!(problem, params; verbose=true)
println("Auto-selected path. Phase: $(log.terminal_phase), Time: $(log.timings.total)")

# Force dense for comparison:
problem2 = SDPA.read_dat_s("theta5.dat-s", Float64)
log2 = SDPA.solve!(problem2, params; verbose=false, force_sparse=false)
println("Dense path. Phase: $(log2.terminal_phase), Time: $(log2.timings.total)")

speedup = log2.timings.total / log.timings.total
println("Sparse speedup: $(speedup)x")
```

## Example 6: Querying custom solution data

```julia
using SDPA, MathOptInterface
const MOI = MathOptInterface

# ... after optimize! on an SDPA.Optimizer ...

# Get the full primal matrix (one Matrix{T} per SDP block):
primal_mats = MOI.get(opt, SDPA.PrimalSolutionMatrix())
# primal_mats[1] is the first SDP block's X matrix.

# Get the dual vector (one entry per equality constraint):
dual_vec = MOI.get(opt, SDPA.DualSolutionVector())

# Get the dual slack matrices (one Matrix{T} per SDP block):
dual_slack = MOI.get(opt, SDPA.DualSlackMatrix())

# Per-constraint primal/dual for equality, PSD, and LP:
# (returns scalar for equality, Vector{T} for PSD triangle-packed, Vector{T} for LP)
y_eq = MOI.get(opt, MOI.ConstraintDual(), eq_constraint_index)
z_psd = MOI.get(opt, MOI.ConstraintDual(), psd_constraint_index)
s_lp = MOI.get(opt, MOI.ConstraintPrimal(), lp_constraint_index)
```

## Example 7: Writing results to file

```julia
using SDPA

problem = SDPA.read_dat_s("example1.dat-s", BigFloat)
params = SDPA.Parameters{BigFloat}(precision=200)
log = SDPA.solve!(problem, params; verbose=false, io=devnull)

open("example1.result", "w") do io
    SDPA.write_result(io, problem, params, log.trajectory,
                      Symbol(string(log.terminal_phase)),
                      log.iterations, log.final_mu,
                      BigFloat(0), BigFloat(0), BigFloat(0),
                      log.final_obj_primal, log.final_obj_dual,
                      log.final_primal_res, log.final_dual_res, 0.0)
end
```

## SDPLIB Reference Values

For benchmarking against known optima, the SDPLIB reference table
(SDPLIB 1.2 archive; values transcribed here). Selected values:

| Problem | m | nBlock | Optimal (GMP) |
|---------|---|--------|---------------|
| arch0 | 174 | [161, -174] | 5.6651727321592959e-01 |
| control5 | 351 | [50, 25] | 1.6883599050955793e+01 |
| theta5 | 3028 | [250] | 5.7232307282180003e+01 |
| theta6 | 4375 | [300] | 6.3477087177964743e+01 |
