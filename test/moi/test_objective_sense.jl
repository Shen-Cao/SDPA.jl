using SDPA
using MathOptInterface
const MOI = MathOptInterface
using Test

# Objective-sense correctness for the MOI wrapper.
#
# The reported objective, primal point, and termination status must
# agree with the true optimum of the problem as stated.
#
# All problems here are chosen so min != max (unlike the constant-objective
# problems in test_moi_basic.jl), so a sense mishandling fails every one.

@testset "MOI objective sense: LP min x s.t. x+y=1, x,y>=0 (1-arg path)" begin
    # True optimum: min x = 0 at (x,y) = (0,1).
    src = MOI.Utilities.Model{Float64}()
    x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 0.0], x), 0.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0), MOI.EqualTo(1.0))

    opt = SDPA.Optimizer{Float64}(silent=true)
    MOI.copy_to(opt, src)
    MOI.optimize!(opt)

    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 0.0 atol=1e-6
    @test MOI.get(opt, MOI.VariablePrimal(), x[1]) ≈ 0.0 atol=1e-6
    @test MOI.get(opt, MOI.VariablePrimal(), x[2]) ≈ 1.0 atol=1e-6
    @test MOI.get(opt, MOI.DualObjectiveValue()) ≈ MOI.get(opt, MOI.ObjectiveValue()) atol=1e-6
end

@testset "MOI objective sense: LP max x s.t. x+y=1, x,y>=0" begin
    # True optimum: max x = 1 at (x,y) = (1,0).
    src = MOI.Utilities.Model{Float64}()
    x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 0.0], x), 0.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0), MOI.EqualTo(1.0))

    opt = SDPA.Optimizer{Float64}(silent=true)
    MOI.optimize!(opt, src)

    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 1.0 atol=1e-6
    @test MOI.get(opt, MOI.VariablePrimal(), x[1]) ≈ 1.0 atol=1e-6
    @test MOI.get(opt, MOI.VariablePrimal(), x[2]) ≈ 0.0 atol=1e-6
    @test MOI.get(opt, MOI.DualObjectiveValue()) ≈ MOI.get(opt, MOI.ObjectiveValue()) atol=1e-6
end

@testset "MOI objective sense: SDP min X[1,1] s.t. X[1,1]+2X[2,2]=1, X⪰0" begin
    # True optimum: min X[1,1] = 0 at X = diag(0, 0.5).
    src = MOI.Utilities.Model{Float64}()
    psd_vars, psd_ci = MOI.add_constrained_variables(
        src, MOI.PositiveSemidefiniteConeTriangle(2))
    # psd_vars[1] = X[1,1], psd_vars[2] = X[1,2], psd_vars[3] = X[2,2]
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], [psd_vars[1]]), 0.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 2.0], [psd_vars[1], psd_vars[3]]), 0.0),
        MOI.EqualTo(1.0))

    opt = SDPA.Optimizer{Float64}(silent=true)
    MOI.optimize!(opt, src)

    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 0.0 atol=1e-6
    @test MOI.get(opt, MOI.VariablePrimal(), psd_vars[1]) ≈ 0.0 atol=1e-6
    @test MOI.get(opt, MOI.VariablePrimal(), psd_vars[3]) ≈ 0.5 atol=1e-6
    psd_primal = MOI.get(opt, MOI.ConstraintPrimal(), psd_ci)
    @test psd_primal isa Vector{Float64}
    @test length(psd_primal) == 3  # 2×2 triangle
    @test MOI.get(opt, MOI.DualObjectiveValue()) ≈ MOI.get(opt, MOI.ObjectiveValue()) atol=1e-6
end

@testset "MOI objective sense: unbounded LP min -x s.t. x-y=0, x,y>=0" begin
    # True optimum: -Infinity (x = y -> +Inf).
    src = MOI.Utilities.Model{Float64}()
    x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-1.0, 0.0], x), 0.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, -1.0], x), 0.0), MOI.EqualTo(0.0))

    opt = SDPA.Optimizer{Float64}(silent=true, maxIteration=200)
    MOI.optimize!(opt, src)

    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
    @test MOI.get(opt, MOI.PrimalStatus()) == MOI.INFEASIBILITY_CERTIFICATE

    # A5: new result-count / iteration attributes + the independent
    # DualStatus mapping (pUNBD certifies dual infeasibility; the dual
    # side has no solution).
    @test MOI.get(opt, MOI.ResultCount()) == 1
    @test MOI.get(opt, MOI.BarrierIterations()) isa Int
    @test MOI.get(opt, MOI.BarrierIterations()) > 0
    @test MOI.get(opt, MOI.DualStatus()) == MOI.NO_SOLUTION
end

@testset "MOI A5: infeasible LP reports INFEASIBLE (independent rows detect)" begin
    # min x1 s.t. x1 + x2 = 1, x1 + 2x2 = 0.5, x >= 0. Solving the two
    # equalities gives x2 = -0.5, which violates x2 >= 0 -> primal infeasible.
    # The constraint rows are linearly INDEPENDENT so the IPM iterates
    # progress and rho grows, letting the phase machine reach pdINF
    # (Float64) / dUNBD (BigFloat) instead of failing Cholesky at iter 0.
    # Both phases map to MOI.INFEASIBLE.
    src = MOI.Utilities.Model{Float64}()
    x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], [x[1]]), 0.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], [x[1], x[2]]), 0.0), MOI.EqualTo(1.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 2.0], [x[1], x[2]]), 0.0), MOI.EqualTo(0.5))

    opt = SDPA.Optimizer{Float64}(silent=true, maxIteration=5000)
    MOI.optimize!(opt, src)

    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.INFEASIBLE
    @test MOI.get(opt, MOI.DualStatus()) == MOI.INFEASIBILITY_CERTIFICATE
    @test MOI.get(opt, MOI.PrimalStatus()) == MOI.NO_SOLUTION
    @test MOI.get(opt, MOI.ResultCount()) == 1
    @test MOI.get(opt, MOI.BarrierIterations()) isa Int
    @test MOI.get(opt, MOI.BarrierIterations()) > 0
end

@testset "MOI A5: pre-optimize result queries throw GetAttributeNotAllowed" begin
    # Fresh optimizer (no copy_to / optimize!): result attributes must not
    # leak the error("optimize! not called") ErrorException but raise the
    # MOI-conformant GetAttributeNotAllowed instead.
    opt = SDPA.Optimizer{Float64}(silent=true)
    @test_throws MOI.GetAttributeNotAllowed MOI.get(opt, MOI.ObjectiveValue())
    @test_throws MOI.GetAttributeNotAllowed MOI.get(opt, MOI.DualObjectiveValue())
    @test_throws MOI.GetAttributeNotAllowed MOI.get(opt, SDPA.PrimalSolutionMatrix())
    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
    @test MOI.get(opt, MOI.ResultCount()) == 0
end

@testset "MOI A5: phase -> status mapping table (unit)" begin
    # Direct check of the internal mapping so ALMOST_OPTIMAL (hard to
    # trigger deterministically end-to-end) is covered deterministically.
    # pFEAS/dFEAS/pdFEAS = feasible but not converged -> ALMOST_OPTIMAL.
    @test SDPA._moi_termination(SDPA.pdOPT) == MOI.OPTIMAL
    @test SDPA._moi_termination(SDPA.pFEAS) == MOI.ALMOST_OPTIMAL
    @test SDPA._moi_termination(SDPA.dFEAS) == MOI.ALMOST_OPTIMAL
    @test SDPA._moi_termination(SDPA.pdFEAS) == MOI.ALMOST_OPTIMAL
    # pFEAS_dINF (DUAL_INFEASIBLE) keeps the feasible primal
    # iterate as FEASIBLE_POINT so ResultCount/ObjectiveValue stay
    # consistent (no ray is extracted — documented interim).
    @test SDPA._moi_primal_status(SDPA.pFEAS_dINF) == MOI.FEASIBLE_POINT
    @test SDPA._moi_dual_status(SDPA.pFEAS_dINF) == MOI.NO_SOLUTION
    @test SDPA._moi_termination(SDPA.pUNBD) == MOI.DUAL_INFEASIBLE
    @test SDPA._moi_termination(SDPA.dUNBD) == MOI.INFEASIBLE
    @test SDPA._moi_termination(SDPA.pINF_dFEAS) == MOI.INFEASIBLE
    @test SDPA._moi_termination(SDPA.pFEAS_dINF) == MOI.DUAL_INFEASIBLE
    @test SDPA._moi_termination(SDPA.pdINF) == MOI.INFEASIBLE
    @test SDPA._moi_termination(SDPA.noINFO) == MOI.OTHER_LIMIT
    @test SDPA._moi_termination(SDPA.not_called) == MOI.OPTIMIZE_NOT_CALLED

    # PrimalStatus: feasible sides -> FEASIBLE_POINT; pUNBD is the primal ray
    # certifying dual infeasibility; everything else has no primal solution.
    @test SDPA._moi_primal_status(SDPA.pdOPT) == MOI.FEASIBLE_POINT
    @test SDPA._moi_primal_status(SDPA.pFEAS) == MOI.FEASIBLE_POINT
    @test SDPA._moi_primal_status(SDPA.pdFEAS) == MOI.FEASIBLE_POINT
    @test SDPA._moi_primal_status(SDPA.pUNBD) == MOI.INFEASIBILITY_CERTIFICATE
    @test SDPA._moi_primal_status(SDPA.dUNBD) == MOI.NO_SOLUTION
    @test SDPA._moi_primal_status(SDPA.pINF_dFEAS) == MOI.NO_SOLUTION
    @test SDPA._moi_primal_status(SDPA.pdINF) == MOI.NO_SOLUTION
    # pFEAS_dINF keeps the feasible primal iterate (see the
    # _moi_termination block above); the dual side has no solution.
    @test SDPA._moi_primal_status(SDPA.pFEAS_dINF) == MOI.FEASIBLE_POINT
    @test SDPA._moi_primal_status(SDPA.noINFO) == MOI.NO_SOLUTION

    # DualStatus (independent mapping, no longer delegating to PrimalStatus):
    # dUNBD/pINF_dFEAS/pdINF carry the dual ray certifying primal
    # infeasibility; pUNBD has no dual solution.
    @test SDPA._moi_dual_status(SDPA.pdOPT) == MOI.FEASIBLE_POINT
    @test SDPA._moi_dual_status(SDPA.dFEAS) == MOI.FEASIBLE_POINT
    @test SDPA._moi_dual_status(SDPA.pdFEAS) == MOI.FEASIBLE_POINT
    @test SDPA._moi_dual_status(SDPA.dUNBD) == MOI.INFEASIBILITY_CERTIFICATE
    @test SDPA._moi_dual_status(SDPA.pINF_dFEAS) == MOI.INFEASIBILITY_CERTIFICATE
    @test SDPA._moi_dual_status(SDPA.pdINF) == MOI.INFEASIBILITY_CERTIFICATE
    @test SDPA._moi_dual_status(SDPA.pUNBD) == MOI.NO_SOLUTION
    @test SDPA._moi_dual_status(SDPA.pFEAS_dINF) == MOI.NO_SOLUTION
    @test SDPA._moi_dual_status(SDPA.pFEAS) == MOI.NO_SOLUTION
    @test SDPA._moi_dual_status(SDPA.noINFO) == MOI.NO_SOLUTION
end

@testset "free scalar variables rejected loudly" begin
    # Plain free scalar variables (MOI.add_variable) referenced only by an
    # equality + objective. Silently mapping them to LP-nonnegative
    # scalars solves the wrong problem (min x s.t. x + y = 1 would
    # become x,y >= 0 with optimum 0), so they are rejected with the
    # MOI-contract error. (Variables DECLARED via the Reals cone are
    # supported through the x⁺−x⁻ split — see the Reals testsets.)
    src = MOI.Utilities.Model{Float64}()
    x = MOI.add_variable(src)
    y = MOI.add_variable(src)
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 0.0], [x, y]), 0.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], [x, y]), 0.0), MOI.EqualTo(1.0))

    opt = SDPA.Optimizer{Float64}(silent=true)
    @test_throws MOI.UnsupportedConstraint{MOI.VariableIndex, MOI.Reals} MOI.optimize!(opt, src)
end

@testset "unsupported constraints throw, not silently dropped" begin
    # ScalarAffineFunction-in-LessThan: unsupported F/S pair.
    src = MOI.Utilities.Model{Float64}()
    x = MOI.add_variable(src)
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], [x]), 0.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0], [x]), 0.0), MOI.LessThan(1.0))

    opt = SDPA.Optimizer{Float64}(silent=true)
    @test_throws MOI.UnsupportedConstraint MOI.optimize!(opt, src)

    # VectorOfVariables-in-SecondOrderCone: also unsupported (different code path).
    src2 = MOI.Utilities.Model{Float64}()
    v = MOI.add_variables(src2, 2)
    MOI.set(src2, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src2, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 0.0], v), 0.0))
    MOI.add_constraint(src2, MOI.VectorOfVariables(v), MOI.SecondOrderCone(2))

    opt2 = SDPA.Optimizer{Float64}(silent=true)
    @test_throws MOI.UnsupportedConstraint MOI.optimize!(opt2, src2)
end

@testset "non-ScalarAffineFunction objective throws UnsupportedAttribute" begin
    # The solver represents objectives as ScalarAffineFunction coefficients.
    # A VectorAffineFunction objective has no representation: ingest must
    # reject it with the MOI-contract error.
    src = MOI.Utilities.Model{Float64}()
    x = MOI.add_variable(src)
    vaf = MOI.VectorAffineFunction(
        [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x))], [0.0])
    MOI.set(src, MOI.ObjectiveFunction{typeof(vaf)}(), vaf)
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    opt = SDPA.Optimizer{Float64}(silent=true)
    @test_throws MOI.UnsupportedAttribute MOI.optimize!(opt, src)
end

@testset "1x1 PSD cone ingested as nonnegativity (min x s.t. x=2)" begin
    # A 1x1 PSD cone is mathematically R_+ and is ingested into the LP
    # block (not as a size-1 SDP block).
    src = MOI.Utilities.Model{Float64}()
    x, psd_ci = MOI.add_constrained_variables(
        src, MOI.PositiveSemidefiniteConeTriangle(1))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], x), 0.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0], x), 0.0), MOI.EqualTo(2.0))

    opt = SDPA.Optimizer{Float64}(silent=true)
    MOI.optimize!(opt, src)

    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 2.0 atol=1e-6
    @test MOI.get(opt, MOI.VariablePrimal(), x[1]) ≈ 2.0 atol=1e-6
    @test MOI.get(opt, MOI.DualObjectiveValue()) ≈ MOI.get(opt, MOI.ObjectiveValue()) atol=1e-6
    # The PSD(1) cone is now an LP cone: its primal is a length-1 vector.
    @test MOI.get(opt, MOI.ConstraintPrimal(), psd_ci) ≈ [2.0] atol=1e-6
end

@testset "mixed 2x2 PSD + 1x1 PSD var (coefficient lands in LP part)" begin
    # min t s.t. t + X[1,1] = 1, X[2,2] = 1, X (2x2) PSD, t in PSD(1).
    # True optimum: t=0, X[1,1]=1, X[2,2]=1 -> obj 0; the 1x1 PSD var's
    # coefficients land in the LP part.
    src = MOI.Utilities.Model{Float64}()
    X, X_ci = MOI.add_constrained_variables(
        src, MOI.PositiveSemidefiniteConeTriangle(2))
    t, _ = MOI.add_constrained_variables(
        src, MOI.PositiveSemidefiniteConeTriangle(1))
    # X[1] = X[1,1], X[2] = X[1,2], X[3] = X[2,2]
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], [t[1]]), 0.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], [t[1], X[1]]), 0.0), MOI.EqualTo(1.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0], [X[3]]), 0.0), MOI.EqualTo(1.0))

    opt = SDPA.Optimizer{Float64}(silent=true)
    MOI.optimize!(opt, src)

    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 0.0 atol=1e-6
    @test MOI.get(opt, MOI.VariablePrimal(), t[1]) ≈ 0.0 atol=1e-6
    @test MOI.get(opt, MOI.VariablePrimal(), X[1]) ≈ 1.0 atol=1e-6
    @test MOI.get(opt, MOI.VariablePrimal(), X[3]) ≈ 1.0 atol=1e-6
    @test MOI.get(opt, MOI.DualObjectiveValue()) ≈ MOI.get(opt, MOI.ObjectiveValue()) atol=1e-6
    # The 2x2 PSD cone still reports its own 3-vector primal.
    Xp = MOI.get(opt, MOI.ConstraintPrimal(), X_ci)
    @test Xp isa Vector{Float64}
    @test length(Xp) == 3
end

@testset "VectorAffineFunction-in-PSD(1) throws UnsupportedConstraint" begin
    # Affine 1x1 PSD is out of scope for the direct VAF conversion; it must
    # throw honestly rather than silently mis-build the problem.
    src = MOI.Utilities.Model{Float64}()
    x = MOI.add_variable(src)
    f = MOI.VectorAffineFunction(
        [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x))], [0.0])
    MOI.add_constraint(src, f, MOI.PositiveSemidefiniteConeTriangle(1))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], [x]), 0.0))

    opt = SDPA.Optimizer{Float64}(silent=true)
    @test_throws MOI.UnsupportedConstraint MOI.optimize!(opt, src)
end

@testset "MOI objective sense: regression, constant-objective problems" begin
    # min x1+x2 s.t. x1+x2=1, x1,x2>=0 (min == max == 1). The fix must not
    # disturb this degenerate family from test_moi_basic.jl.
    src = MOI.Utilities.Model{Float64}()
    x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0), MOI.EqualTo(1.0))

    opt = SDPA.Optimizer{Float64}(silent=true)
    MOI.optimize!(opt, src)

    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 1.0 atol=1e-6
    @test MOI.get(opt, MOI.VariablePrimal(), x[1]) + MOI.get(opt, MOI.VariablePrimal(), x[2]) ≈ 1.0 atol=1e-6
end

@testset "TimeLimitSec set/get round-trip" begin
    # A real time limit attribute.
    opt = SDPA.Optimizer{Float64}(silent=true)
    @test MOI.supports(opt, MOI.TimeLimitSec())
    # Default: no time limit.
    @test MOI.get(opt, MOI.TimeLimitSec()) === nothing
    # Set a number.
    MOI.set(opt, MOI.TimeLimitSec(), 30.0)
    @test MOI.get(opt, MOI.TimeLimitSec()) == 30.0
    # Integer inputs are normalized to Float64.
    MOI.set(opt, MOI.TimeLimitSec(), 5)
    @test MOI.get(opt, MOI.TimeLimitSec()) == 5.0
    # Clearing restores nothing.
    MOI.set(opt, MOI.TimeLimitSec(), nothing)
    @test MOI.get(opt, MOI.TimeLimitSec()) === nothing

    # empty! is an optimizer-level reset: it must PRESERVE the time limit
    # (like `silent`), because TimeLimitSec is an optimizer attribute, not
    # per-model state.
    MOI.set(opt, MOI.TimeLimitSec(), 30.0)
    src = MathOptInterface.Utilities.Model{Float64}()
    # Use a cone-constrained variable (cone-less free variables are
    # rejected).
    x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(1))
    f = MOI.ScalarAffineFunction{Float64}([MOI.ScalarAffineTerm(1.0, x[1])], 0.0)
    MOI.add_constraint(src, f, MOI.EqualTo(1.0))
    MOI.set(src, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.copy_to(opt, src)
    MOI.empty!(opt)
    @test MOI.get(opt, MOI.TimeLimitSec()) == 30.0  # preserved across empty!
end

@testset "TimeLimitSec enforcement returns TIME_LIMIT" begin
    # A BigFloat(512) PSD(4) solve needs ~40 iterations / seconds of wall
    # time; a 1 ms limit expires on the first wall-clock check in the
    # iteration loop.
    T = BigFloat
    src = MOI.Utilities.Model{T}()
    psd_vars, _ = MOI.add_constrained_variables(
        src, MOI.PositiveSemidefiniteConeTriangle(4))
    diag_idx = [div(j*(j + 1), 2) for j in 1:4]  # (j,j) → triangle index
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(one(T), psd_vars[k]) for k in diag_idx],
                zero(T)))
    MOI.add_constraint(
        src,
        MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(one(T), psd_vars[k]) for k in diag_idx],
            zero(T)),
        MOI.EqualTo(one(T)))

    opt = SDPA.Optimizer{T}(silent=true, precision=512, maxIteration=200)
    MOI.set(opt, MOI.TimeLimitSec(), 0.001)
    MOI.optimize!(opt, src)
    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.TIME_LIMIT
end

@testset "supports_constraint advertises the ingestable constraint types" begin
    # The optimizer ingests four constraint types; the three below are
    # advertised (MOI contract: copy_to does not throw UnsupportedConstraint
    # for them). VectorAffineFunction-in-PSD is NOT advertised: the direct
    # VAF ingest path stalls at dFEAS on boundary-touching solutions, so
    # matrix-form constraints are routed through the VectorFunctionizeBridge
    # to VectorOfVariables-in-PSD + ScalarAffineFunction-in-EqualTo, which
    # converge reliably (documented deviation).
    opt = SDPA.Optimizer{Float64}(silent=true)
    @test MOI.supports_constraint(
        opt, MOI.VectorOfVariables, MOI.PositiveSemidefiniteConeTriangle)
    @test MOI.supports_constraint(
        opt, MOI.VectorOfVariables, MOI.Nonnegatives)
    @test MOI.supports_constraint(
        opt, MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64})
    @test !MOI.supports_constraint(
        opt, MOI.VectorAffineFunction{Float64},
        MOI.PositiveSemidefiniteConeTriangle)
end

@testset "unmapped constraint-index queries throw InvalidIndex" begin
    # Querying an index outside the model must throw
    # MOI.InvalidIndex per the contract.
    src = MOI.Utilities.Model{Float64}()
    x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0), MOI.EqualTo(1.0))

    opt = SDPA.Optimizer{Float64}(silent=true)
    MOI.copy_to(opt, src)
    MOI.optimize!(opt)
    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL

    fake_ci = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}(999)
    @test_throws MOI.InvalidIndex MOI.get(opt, MOI.ConstraintPrimal(), fake_ci)
    @test_throws MOI.InvalidIndex MOI.get(opt, MOI.ConstraintDual(), fake_ci)
end

@testset "mismatched-T equality constraint throws UnsupportedConstraint" begin
    # A Float64 ScalarAffineFunction-in-EqualTo into an Optimizer{BigFloat}
    # must throw a clean MOI.UnsupportedConstraint.
    src = MOI.Utilities.Model{Float64}()
    x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0), MOI.EqualTo(1.0))

    opt = SDPA.Optimizer{BigFloat}(silent=true)
    @test_throws MOI.UnsupportedConstraint MOI.optimize!(opt, src)
end

@testset "objective constant offset" begin
    # Ingest stores the ScalarAffineFunction constant separately; the
    # accessors must add it back (sign-correct under MIN/MAX).
    for T in (Float64, BigFloat)
        for sense in (MOI.MIN_SENSE, MOI.MAX_SENSE)
            src = MOI.Utilities.Model{T}()
            x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
            MOI.set(src, MOI.ObjectiveSense(), sense)
            MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(T[1, 1], x), T(5)))
            MOI.add_constraint(src, MOI.ScalarAffineFunction(
                MOI.ScalarAffineTerm.(T[1, 1], x), zero(T)), MOI.EqualTo(one(T)))
            opt = SDPA.Optimizer{T}(silent=true)
            MOI.optimize!(opt, src)
            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
            # x1 + x2 = 1, xi >= 0, objective x1 + x2 + 5 → optimum 6 for
            # both senses (min and max of x1+x2 over the simplex agree).
            @test MOI.get(opt, MOI.ObjectiveValue()) ≈ T(6) atol=T(1e-6)
            @test MOI.get(opt, MOI.DualObjectiveValue()) ≈ T(6) atol=T(1e-6)
        end
    end
end

@testset "MOI conic duals for min f and max -f" begin
    for T in (Float64, BigFloat), sense in (MOI.MIN_SENSE, MOI.MAX_SENSE)
        sign = sense == MOI.MIN_SENSE ? one(T) : -one(T)
        # Float64 uses 1e-7 solver tolerances; BigFloat uses 1e-30.
        tol = T === Float64 ? T(1e-6) : T(1e-24)

        @testset "same objective, different senses ($T, $sense)" begin
            # Keep the objective +2x (do not negate it between senses).
            # For x=5>0 the cone slack is zero; the equality dual is
            # +2 at minimization and -2 at maximization.
            src = MOI.Utilities.Model{T}()
            x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(1))
            eq = MOI.add_constraint(src,
                MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(one(T), x[1])], zero(T)),
                MOI.EqualTo(T(5)))
            MOI.set(src, MOI.ObjectiveSense(), sense)
            MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(T(2), x[1])], zero(T)))
            opt = SDPA.Optimizer{T}(silent=true)
            MOI.optimize!(opt, src)
            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
            @test MOI.get(opt, MOI.ConstraintDual(), eq) ≈ sign * 2 atol=tol
        end

        @testset "LP and equality ($T, $sense)" begin
            # min 2x + y (or max its negative), x + y = 5, x,y >= 0.
            # Both forms have the MOI equality dual 1 and cone dual [1,0].
            src = MOI.Utilities.Model{T}()
            v, lp_ci = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
            c = T[2, 1]
            eq = MOI.add_constraint(src,
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(T, 2), v), zero(T)),
                MOI.EqualTo(T(5)))
            MOI.set(src, MOI.ObjectiveSense(), sense)
            MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(sign .* c, v), zero(T)))
            opt = SDPA.Optimizer{T}(silent=true)
            MOI.optimize!(opt, src)
            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
            x = MOI.get(opt, MOI.ConstraintPrimal(), lp_ci)
            z = MOI.get(opt, MOI.ConstraintDual(), lp_ci)
            y = MOI.get(opt, MOI.ConstraintDual(), eq)
            @test sum(x) ≈ 5 atol=tol
            @test x ≈ T[0, 5] atol=tol
            @test y ≈ 1 atol=tol
            @test minimum(z) >= -tol
            @test c ≈ y .+ z atol=tol       # stationarity in minimization form
            @test abs(sum(x .* z)) <= tol  # complementarity
            @test MOI.get(opt, MOI.ObjectiveValue()) ≈ sign * 5 atol=tol
            @test MOI.get(opt, MOI.DualObjectiveValue()) ≈ sign * 5 atol=tol
        end

        @testset "PSD triangle and equality ($T, $sense)" begin
            # C = [2 1; 1 2], tr(X) = 1. The optimal duals are y=1,
            # Z=C-I=[1 1; 1 1]. Nonzero off-diagonals exercise the
            # triangle scalar product as well as the sign of the dual.
            src = MOI.Utilities.Model{T}()
            v, psd_ci = MOI.add_constrained_variables(
                src, MOI.PositiveSemidefiniteConeTriangle(2))
            eq = MOI.add_constraint(src,
                MOI.ScalarAffineFunction(
                    MOI.ScalarAffineTerm.(ones(T, 2), v[[1, 3]]), zero(T)),
                MOI.EqualTo(one(T)))
            MOI.set(src, MOI.ObjectiveSense(), sense)
            MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(fill(2 * sign, 3), v),
                                         4 * sign))
            opt = SDPA.Optimizer{T}(silent=true)
            MOI.optimize!(opt, src)
            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
            x11, x12, x22 = MOI.get(opt, MOI.ConstraintPrimal(), psd_ci)
            z11, z12, z22 = MOI.get(opt, MOI.ConstraintDual(), psd_ci)
            y = MOI.get(opt, MOI.ConstraintDual(), eq)
            @test x11 + x22 ≈ 1 atol=tol
            @test y ≈ 1 atol=tol
            @test T[y + z11, z12, y + z22] ≈ T[2, 1, 2] atol=tol
            @test min(z11, z22) >= -tol
            @test z11 * z22 - z12^2 >= -tol
            @test abs(x11*z11 + 2*x12*z12 + x22*z22) <= tol
            @test MOI.get(opt, MOI.ObjectiveValue()) ≈ sign * 5 atol=tol
            @test MOI.get(opt, MOI.DualObjectiveValue()) ≈ sign * 5 atol=tol
        end
    end
end
