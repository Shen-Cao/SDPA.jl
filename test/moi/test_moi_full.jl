using SDPA
using Test
using MathOptInterface
const MOI = MathOptInterface

# Full MOI surface at BigFloat(256).
#
# Uses MOI.Utilities.Model{BigFloat} directly (rather than JuMP) because
# JuMP's macros generate Float64 coefficients even for BigFloat optimizers,
# and the type mismatch breaks the bridge.
#
# Exercises: RawOptimizerAttribute, RawStatusString, DualObjectiveValue,
# ConstraintDual, ConstraintPrimal, PrimalSolutionMatrix, DualSolutionVector,
# DualSlackMatrix. Uses rtol=BigFloat("1e-20") for the duality gap.

@testset "full MOI surface at BigFloat(256)" begin
    T = BigFloat
    src = MOI.Utilities.Model{T}()
    psd_vars, psd_ci = MOI.add_constrained_variables(
        src, MOI.PositiveSemidefiniteConeTriangle(2))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
            MOI.ScalarAffineFunction(
                MOI.ScalarAffineTerm.([one(T), one(T)],
                                       [psd_vars[1], psd_vars[3]]), zero(T)))
    eq_ci = MOI.add_constraint(
        src,
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([one(T), one(T)],
                                   [psd_vars[1], psd_vars[3]]), zero(T)),
        MOI.EqualTo(one(T)))

    opt = SDPA.Optimizer{T}(silent=true, precision=256)
    MOI.set(opt, MOI.RawOptimizerAttribute("maxIteration"), 100)
    MOI.optimize!(opt, src)

    # Termination + raw status.
    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(opt, MOI.RawStatusString()) == "PD_OPTIMAL"

    # Strong duality at rtol=1e-20 (both sides at BigFloat(256)).
    obj_primal = MOI.get(opt, MOI.ObjectiveValue())
    obj_dual = MOI.get(opt, MOI.DualObjectiveValue())
    @test isapprox(obj_dual, obj_primal; rtol=BigFloat("1e-20"))

    # ConstraintDual for equality constraint — must be nonzero.
    y = MOI.get(opt, MOI.ConstraintDual(), eq_ci)
    @test abs(y) > BigFloat("1e-6")

    # ConstraintPrimal for equality — b value (== 1.0 here).
    cp = MOI.get(opt, MOI.ConstraintPrimal(), eq_ci)
    @test cp isa BigFloat
    @test abs(cp - 1) < BigFloat("1e-6")

    # PSD ConstraintPrimal/Dual.
    psd_primal = MOI.get(opt, MOI.ConstraintPrimal(), psd_ci)
    @test psd_primal isa Vector{BigFloat}
    @test length(psd_primal) == 3
    psd_dual = MOI.get(opt, MOI.ConstraintDual(), psd_ci)
    @test psd_dual isa Vector{BigFloat}
    @test length(psd_dual) == 3

    # All 3 custom accessors.
    primal_mats = MOI.get(opt, SDPA.PrimalSolutionMatrix())
    @test primal_mats isa Vector{Matrix{BigFloat}}
    @test size(primal_mats[1]) == (2, 2)

    dual_vec = MOI.get(opt, SDPA.DualSolutionVector())
    @test dual_vec isa Vector{BigFloat}
    @test length(dual_vec) >= 1

    dual_slack = MOI.get(opt, SDPA.DualSlackMatrix())
    @test dual_slack isa Vector{Matrix{BigFloat}}
    @test length(dual_slack) >= 1
end
