using SDPA
using MathOptInterface
const MOI = MathOptInterface
using Test

# Tests that the MOI wrapper can solve a basic SDP problem via
# direct in-memory conversion (no file round-trip).

@testset "MOI basic LP via SDPA" begin
    # Problem: min x1 + x2  s.t.  x1 + x2 = 1,  x1 ≥ 0,  x2 ≥ 0
    # Optimal: objP = 1.0,  x1 + x2 = 1
    src = MOI.Utilities.Model{Float64}()
    x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
    # Objective: min x[1] + x[2]
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], x),
            0.0,
        ),
    )
    # Constraint: x[1] + x[2] = 1
    MOI.add_constraint(src,
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], x),
            0.0,
        ),
        MOI.EqualTo(1.0),
    )

    opt = SDPA.Optimizer{Float64}(silent=true)
    MOI.optimize!(opt, src)

    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 1.0 atol=1e-6
    # Check primal values
    x1_val = MOI.get(opt, MOI.VariablePrimal(), x[1])
    x2_val = MOI.get(opt, MOI.VariablePrimal(), x[2])
    @test x1_val ≥ -1e-6       # non-negative
    @test x2_val ≥ -1e-6       # non-negative
    @test x1_val + x2_val ≈ 1.0 atol=1e-6
end

@testset "MOI basic SDP via SDPA" begin
    # Problem: min X[1,1] + X[2,2]
    #   s.t.  X[1,1] + X[2,2] = 1
    #         X = [[X11, X12], [X12, X22]] ⪰ 0
    # Optimal: objP = 1.0
    src = MOI.Utilities.Model{Float64}()
    # PSD variable: 2x2 triangle → 3 variables
    psd_vars, psd_ci = MOI.add_constrained_variables(
        src, MOI.PositiveSemidefiniteConeTriangle(2),
    )
    # psd_vars[1] = X[1,1], psd_vars[2] = X[1,2], psd_vars[3] = X[2,2]
    # Objective: min X[1,1] + X[2,2]
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], [psd_vars[1], psd_vars[3]]),
            0.0,
        ),
    )
    # Constraint: X[1,1] + X[2,2] = 1
    MOI.add_constraint(src,
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], [psd_vars[1], psd_vars[3]]),
            0.0,
        ),
        MOI.EqualTo(1.0),
    )

    opt = SDPA.Optimizer{Float64}(silent=true)
    MOI.optimize!(opt, src)

    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 1.0 atol=1e-6

    # Verify PSD cone ConstraintDual returns triangle-packed Vector
    psd_dual = MOI.get(opt, MOI.ConstraintDual(), psd_ci)
    @test psd_dual isa Vector{Float64}
    @test length(psd_dual) == 3  # 2×2 triangle = 3 entries
    @test any(!iszero, psd_dual)  # not all zeros (real dual values)

    # Verify PSD cone ConstraintPrimal returns triangle-packed Vector
    psd_primal = MOI.get(opt, MOI.ConstraintPrimal(), psd_ci)
    @test psd_primal isa Vector{Float64}
    @test length(psd_primal) == 3
end

@testset "Optimizer precision= kwarg works for all T" begin
    for T in (Float64, BigFloat)
        opt = SDPA.Optimizer{T}(silent=true, precision=53)
        @test opt.params.precision == 53
        # End-to-end: no MethodError
        src = MOI.Utilities.Model{T}()
        x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(T[1, 1], x), zero(T)))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.(T[1, 1], x), zero(T)), MOI.EqualTo(one(T)))
        MOI.optimize!(opt, src)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    end

    @testset "RawOptimizerAttribute routes through validators" begin
        T = BigFloat
        opt = SDPA.Optimizer{T}()

        # 1. Happy: valid value sets the field.
        MOI.set(opt, MOI.RawOptimizerAttribute("maxIteration"), 500)
        @test opt.params.maxIteration == 500

        # 2. Failure: bad value — validator must reject (betaStar must be > 0).
        #    Validator throws ArgumentError via _validate.
        @test_throws ArgumentError MOI.set(opt,
            MOI.RawOptimizerAttribute("betaStar"), BigFloat("-0.1"))

        # 3. Failure: type mismatch (maxIteration is Int; convert(Int, 1.5)
        #    throws InexactError). Use Exception to cover any rejection.
        @test_throws Exception MOI.set(opt,
            MOI.RawOptimizerAttribute("maxIteration"), 1.5)

        # 4. Failure: unknown attribute name -> UnsupportedAttribute.
        @test_throws MOI.UnsupportedAttribute MOI.set(opt,
            MOI.RawOptimizerAttribute("nonexistent"), 1)
    end

    @testset "Custom result accessors" begin
        # Build a 2x2 PSD problem so the accessors have non-trivial data to return.
        src = MOI.Utilities.Model{Float64}()
        psd_vars, _ = MOI.add_constrained_variables(
            src, MOI.PositiveSemidefiniteConeTriangle(2))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(
                    MOI.ScalarAffineTerm.([1.0, 1.0], [psd_vars[1], psd_vars[3]]), 0.0))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], [psd_vars[1], psd_vars[3]]), 0.0),
            MOI.EqualTo(1.0))

        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.optimize!(opt, src)

        # PrimalSolutionMatrix: returns Vector{Matrix{T}} with one 2x2 block.
        primal_mats = MOI.get(opt, SDPA.PrimalSolutionMatrix())
        @test primal_mats isa Vector{Matrix{Float64}}
        @test length(primal_mats) >= 1
        @test size(primal_mats[1]) == (2, 2)

        # DualSolutionVector: returns Vector{T} with one entry per equality.
        dual_vec = MOI.get(opt, SDPA.DualSolutionVector())
        @test dual_vec isa Vector{Float64}
        @test length(dual_vec) >= 1

        # DualSlackMatrix: returns Vector{Matrix{T}} with one 2x2 block.
        dual_slack = MOI.get(opt, SDPA.DualSlackMatrix())
        @test dual_slack isa Vector{Matrix{Float64}}
        @test length(dual_slack) >= 1
    end

    @testset "ConstraintDual equality case" begin
        # Set up: min X[1,1] + X[2,2] s.t. X[1,1] + X[2,2] == 1, X PSD.
        # At optimality the equality dual y is nonzero (sensitive to obj).
        src = MOI.Utilities.Model{Float64}()
        psd_vars, psd_ci = MOI.add_constrained_variables(
            src, MOI.PositiveSemidefiniteConeTriangle(2))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(
                    MOI.ScalarAffineTerm.([1.0, 1.0], [psd_vars[1], psd_vars[3]]), 0.0))
        eq_ci = MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], [psd_vars[1], psd_vars[3]]), 0.0),
            MOI.EqualTo(1.0))

        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.optimize!(opt, src)

        # Query equality dual.
        y = MOI.get(opt, MOI.ConstraintDual(), eq_ci)
        @test y isa Float64
        @test abs(y) > 1e-6  # NOT the zero stub
    end

    @testset "ConstraintDual LP cone case" begin
        # min x1 + x2 s.t. x >= 0, x1 + x2 == 1.
        # The LP cone has 2 variables; querying the LP constraint dual
        # returns the full LP dual slack vector.
        src = MOI.Utilities.Model{Float64}()
        x, lp_ci = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(
                    MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0),
            MOI.EqualTo(1.0))

        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.optimize!(opt, src)

        # Query LP cone dual (returns full LP dual vector).
        lp_dual = MOI.get(opt, MOI.ConstraintDual(), lp_ci)
        @test lp_dual isa Vector{Float64}
        @test length(lp_dual) == 2  # 2 LP variables
        # The dual of Nonnegatives is Nonnegatives for either sense.
        @test minimum(lp_dual) >= -1e-6
    end
end

@testset "LP-cone ConstraintPrimal/ConstraintDual are per-CI slices" begin
    # Two Nonnegatives cones (2 and 3 vars) with distinguishable optima.
    #   min a1 + b1  s.t.  a1 + a2 == 1,  b1 + b2 + b3 == 1,  a, b >= 0
    # Optimum: a = [0, 1] (a1 pinned to 0 by min a1), b = [0, ½, ½]
    # (analytic center of the b-optimal face); obj 0. Duals: y = [0, 0],
    # cone dual z = c - A'y → cone A slice [1, 0], cone B slice [1, 0, 0].
    function build_two_cone_model(::Type{T}, sense) where {T}
        src = MOI.Utilities.Model{T}()
        a, ci_a = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
        b, ci_b = MOI.add_constrained_variables(src, MOI.Nonnegatives(3))
        MOI.set(src, MOI.ObjectiveSense(), sense)
        sign = sense === MOI.MAX_SENSE ? -1.0 : 1.0
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                MOI.ScalarAffineFunction(
                    MOI.ScalarAffineTerm.(T[sign, 0, sign, 0, 0], [a; b]),
                    zero(T)))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.(T[1, 1], a), zero(T)), MOI.EqualTo(one(T)))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.(T[1, 1, 1], b), zero(T)), MOI.EqualTo(one(T)))
        return src, ci_a, ci_b
    end

    @testset "MIN_SENSE: each CI gets exactly its own slice" begin
        src, ci_a, ci_b = build_two_cone_model(Float64, MOI.MIN_SENSE)
        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.optimize!(opt, src)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL

        # Primal: EXACTLY each CI's k values, in user variable order.
        cp_a = MOI.get(opt, MOI.ConstraintPrimal(), ci_a)
        cp_b = MOI.get(opt, MOI.ConstraintPrimal(), ci_b)
        @test cp_a isa Vector{Float64} && length(cp_a) == 2
        @test cp_b isa Vector{Float64} && length(cp_b) == 3
        @test cp_a ≈ [0.0, 1.0] atol=1e-5
        @test cp_b ≈ [0.0, 0.5, 0.5] atol=1e-5
        @test length(cp_a) != length(cp_b)  # the two CIs' results differ

        # Dual: same slicing — cone A's dual lives on positions 1:2,
        # cone B's on 3:5 of zMat.LP_block.
        cd_a = MOI.get(opt, MOI.ConstraintDual(), ci_a)
        cd_b = MOI.get(opt, MOI.ConstraintDual(), ci_b)
        @test cd_a isa Vector{Float64} && length(cd_a) == 2
        @test cd_b isa Vector{Float64} && length(cd_b) == 3
        @test cd_a ≈ [1.0, 0.0] atol=1e-5
        @test cd_b ≈ [1.0, 0.0, 0.0] atol=1e-5
    end

    @testset "MAX_SENSE: sliced cone duals stay nonnegative" begin
        # max -(a1+b1) has the same conic duals as min a1+b1.
        src, ci_a, ci_b = build_two_cone_model(Float64, MOI.MAX_SENSE)
        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.optimize!(opt, src)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 0.0 atol=1e-6
        cd_a = MOI.get(opt, MOI.ConstraintDual(), ci_a)
        cd_b = MOI.get(opt, MOI.ConstraintDual(), ci_b)
        @test length(cd_a) == 2 && length(cd_b) == 3
        @test cd_a ≈ [1.0, 0.0] atol=1e-5
        @test cd_b ≈ [1.0, 0.0, 0.0] atol=1e-5
    end

    @testset "PSD(1)-routed CI returns its single value" begin
        # A PositiveSemidefiniteConeTriangle(1) is ingested into the LP
        # block: its CI must return exactly its 1 variable — not the full
        # LP vector (which also contains the Nonnegatives(2) cone's vars).
        src = MOI.Utilities.Model{Float64}()
        y, ci_psd1 = MOI.add_constrained_variables(
            src, MOI.PositiveSemidefiniteConeTriangle(1))
        b, ci_b = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(
                    MOI.ScalarAffineTerm.([1.0], [b[1]]), 0.0))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0], [y[1]]), 0.0), MOI.EqualTo(3.0))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], b), 0.0), MOI.EqualTo(2.0))
        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.optimize!(opt, src)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL

        cp = MOI.get(opt, MOI.ConstraintPrimal(), ci_psd1)
        @test cp isa Vector{Float64} && length(cp) == 1
        @test cp ≈ [3.0] atol=1e-6
        cd = MOI.get(opt, MOI.ConstraintDual(), ci_psd1)
        @test cd isa Vector{Float64} && length(cd) == 1

        # The coexisting Nonnegatives cone still gets its own slice.
        cp_b = MOI.get(opt, MOI.ConstraintPrimal(), ci_b)
        @test length(cp_b) == 2
        @test cp_b ≈ [0.0, 2.0] atol=1e-5
    end
end

@testset "caller-exposed LP result copies are mutation-isolated" begin
    # Mutating a vector returned by an MOI result getter must NOT change
    # solver state. Base copy() on Vector{BigFloat} copies REFERENCES,
    # so BigFloat is the case the isolation assertions actually exercise;
    # Float64 is a correctness variant (plain copy is already
    # value-semantic there).
    for T in (BigFloat, Float64)
        src = MOI.Utilities.Model{T}()
        x, lp_ci = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(T[1, 1], x), zero(T)))
        eq_ci = MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.(T[1, 1], x), zero(T)), MOI.EqualTo(one(T)))

        opt = SDPA.Optimizer{T}(silent=true)
        MOI.optimize!(opt, src)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL

        # In-place element mutation (BigFloat: mp_add! mutates shared MPFR
        # data; Float64: plain assignment).
        mutate1! = T === BigFloat ?
            (v -> SDPA.mp_add!(v[1], v[1], one(T))) :
            (v -> (v[1] = v[1] + one(T)))

        # ConstraintDual LP vector — reads zMat.LP_block (dual slack Z);
        # value copy
        dual = MOI.get(opt, MOI.ConstraintDual(), lp_ci)
        @test dual isa Vector{T} && length(dual) == 2
        orig_dual1 = deepcopy(dual[1])
        mutate1!(dual)
        dual2 = MOI.get(opt, MOI.ConstraintDual(), lp_ci)
        @test dual2[1] == orig_dual1  # solver state unchanged
        @test dual2 != dual           # returned vectors are independent

        # ConstraintPrimal LP vector — reads xMat.LP_block (primal X);
        # value copy
        primal = MOI.get(opt, MOI.ConstraintPrimal(), lp_ci)
        @test primal isa Vector{T} && length(primal) == 2
        orig_p1 = deepcopy(primal[1])
        mutate1!(primal)
        primal2 = MOI.get(opt, MOI.ConstraintPrimal(), lp_ci)
        @test primal2[1] == orig_p1

        # DualSolutionVector — wraps yVec.ele; value copy
        y = MOI.get(opt, SDPA.DualSolutionVector())
        @test y isa Vector{T} && length(y) == 1
        orig_y1 = deepcopy(y[1])
        mutate1!(y)
        y2 = MOI.get(opt, SDPA.DualSolutionVector())
        @test y2[1] == orig_y1

        # PrimalSolutionMatrix — wraps xMat (primal X); LP-only problem
        # → single 1×2 row; value copy
        pm = MOI.get(opt, SDPA.PrimalSolutionMatrix())
        @test length(pm) == 1 && size(pm[1]) == (1, 2)
        orig_pm11 = deepcopy(pm[1][1, 1])
        mutate1!(pm[1])
        pm2 = MOI.get(opt, SDPA.PrimalSolutionMatrix())
        @test pm2[1][1, 1] == orig_pm11

        # DualSlackMatrix — wraps zMat (dual slack Z); same _wrap_dls
        # copy path
        dsm = MOI.get(opt, SDPA.DualSlackMatrix())
        @test length(dsm) == 1 && size(dsm[1]) == (1, 2)
        orig_dsm11 = deepcopy(dsm[1][1, 1])
        mutate1!(dsm[1])
        dsm2 = MOI.get(opt, SDPA.DualSlackMatrix())
        @test dsm2[1][1, 1] == orig_dsm11

        # Scalar result reads are value-isolated too:
        # mutating a returned BigFloat must NOT change solver state.
        # (Float64 scalars are immutable values — nothing to test there.)
        if T === BigFloat
            xp = MOI.get(opt, MOI.VariablePrimal(), x[1])
            orig_xp = deepcopy(xp)
            SDPA.mp_add!(xp, xp, one(T))
            @test MOI.get(opt, MOI.VariablePrimal(), x[1]) == orig_xp

            yd = MOI.get(opt, MOI.ConstraintDual(), eq_ci)
            orig_yd = deepcopy(yd)
            SDPA.mp_add!(yd, yd, one(T))
            @test MOI.get(opt, MOI.ConstraintDual(), eq_ci) == orig_yd
        end
    end
end

@testset "PSD cone result vectors are mutation-isolated" begin
    # Regression: _dense_to_triangle_vector used to store REFERENCES into
    # the solution matrices (BigFloat elements are mutable MPFR objects),
    # so mutating a returned PSD cone vector corrupted solver state and
    # every subsequent query. The value-copy fix covers both
    # ConstraintPrimal (xMat) and ConstraintDual (zMat) packing paths.
    for T in (BigFloat, Float64)
        src = MOI.Utilities.Model{T}()
        x, psd_ci = MOI.add_constrained_variables(
            src, MOI.PositiveSemidefiniteConeTriangle(2))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(T(1), x[1])], zero(T)))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(T(1), x[1])], zero(T)), MOI.EqualTo(one(T)))

        opt = SDPA.Optimizer{T}(silent=true)
        MOI.optimize!(opt, src)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL

        # In-place element mutation (BigFloat: mp_add! mutates shared MPFR
        # data; Float64: plain assignment).
        mutate1! = T === BigFloat ?
            (v -> SDPA.mp_add!(v[1], v[1], one(T))) :
            (v -> (v[1] = v[1] + one(T)))

        cp = MOI.get(opt, MOI.ConstraintPrimal(), psd_ci)
        @test cp isa Vector{T} && length(cp) == 3
        orig_cp1 = deepcopy(cp[1])
        mutate1!(cp)
        cp2 = MOI.get(opt, MOI.ConstraintPrimal(), psd_ci)
        @test cp2[1] == orig_cp1  # solver state unchanged

        cd = MOI.get(opt, MOI.ConstraintDual(), psd_ci)
        @test cd isa Vector{T} && length(cd) == 3
        orig_cd1 = deepcopy(cd[1])
        mutate1!(cd)
        cd2 = MOI.get(opt, MOI.ConstraintDual(), psd_ci)
        @test cd2[1] == orig_cd1
    end
end

@testset "Float64 defaults converge on degenerate-but-solvable PSD" begin
    # min X[1,1] s.t. X[1,1] = 1, X PSD(2). With the per-T double
    # presets (1e-7) it must reach OPTIMAL.
    src = MOI.Utilities.Model{Float64}()
    x, psd_ci = MOI.add_constrained_variables(
        src, MOI.PositiveSemidefiniteConeTriangle(2))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], [x[1]]), 0.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0], [x[1]]), 0.0), MOI.EqualTo(1.0))

    opt = SDPA.Optimizer{Float64}(silent=true)
    MOI.optimize!(opt, src)
    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 1.0 atol=1e-6
end

@testset "ObjectiveSense / SolverVersion getters" begin
        opt = SDPA.Optimizer{Float64}(silent=true)
        @test occursin(r"^\d+\.\d+\.\d+", MOI.get(opt, MOI.SolverVersion()))
        # SolverVersion is get-only — `supports` must NOT advertise
        # settability.
        @test !MOI.supports(opt, MOI.SolverVersion())
        # Before copy_to the sense is FEASIBILITY_SENSE (MOI default).
        @test MOI.get(opt, MOI.ObjectiveSense()) == MOI.FEASIBILITY_SENSE
        # After copy_to it reflects the model's sense.
        src = MOI.Utilities.Model{Float64}()
        x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0), MOI.EqualTo(1.0))
        MOI.optimize!(opt, src)
        @test MOI.get(opt, MOI.ObjectiveSense()) == MOI.MAX_SENSE
    end

@testset "ListOfOptimizerAttributesSet lists only settable attributes" begin
        # SolverName/SolverVersion are get-only; advertising them broke the
        # MOI copy machinery (MOIU.reset_optimizer: get on src, set on dest
        # died with SetAttributeNotAllowed mid-copy).
        opt = SDPA.Optimizer{Float64}(silent=true)
        list = MOI.get(opt, MOI.ListOfOptimizerAttributesSet())
        @test list == [MOI.Silent(), MOI.TimeLimitSec()]
        # The copy loop MOI machinery performs between optimizers must run
        # without throwing (get on src, set on dest for every attr).
        opt2 = SDPA.Optimizer{Float64}()
        for attr in list
            MOI.set(opt2, attr, MOI.get(opt, attr))
        end
        @test MOI.get(opt2, MOI.Silent()) == true
    end

@testset "free variables rejected loudly" begin
    # A cone-less scalar variable must fail with the MOI-contract error
    # (not be silently modeled as x >= 0).
    src = MOI.Utilities.Model{Float64}()
    x, y = MOI.add_variables(src, 2)  # no cone -> genuinely free
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], [x]), 0.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], [x, y]), 0.0), MOI.EqualTo(1.0))
    opt = SDPA.Optimizer{Float64}(silent=true)
    @test_throws MOI.UnsupportedConstraint{MOI.VariableIndex, MOI.Reals} MOI.optimize!(opt, src)
end

@testset "InvalidIndex + unused cone-less variables" begin
    # (a) VariablePrimal on an unmapped index throws MOI.InvalidIndex per
    # the MOI contract.
    src = MOI.Utilities.Model{Float64}()
    x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(1))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], x), 0.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0], x), 0.0), MOI.EqualTo(1.0))
    opt = SDPA.Optimizer{Float64}(silent=true)
    MOI.optimize!(opt, src)
    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test_throws MOI.InvalidIndex MOI.get(
        opt, MOI.VariablePrimal(), MOI.VariableIndex(999))

    # (b) a bare MOI.add_variable scalar used by no cone/equality/objective
    # must be rejected DURING copy_to (contract error) — not left
    # out of the IndexMap to die later as a post-solve KeyError on
    # value(x).
    src2 = MOI.Utilities.Model{Float64}()
    y, _ = MOI.add_constrained_variables(src2, MOI.Nonnegatives(1))
    MOI.add_variable(src2)  # cone-less stray, never referenced again
    MOI.set(src2, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src2, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], y), 0.0))
    MOI.add_constraint(src2, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0], y), 0.0), MOI.EqualTo(1.0))
    opt2 = SDPA.Optimizer{Float64}(silent=true)
    @test_throws MOI.UnsupportedConstraint{MOI.VariableIndex, MOI.Reals} MOI.copy_to(
        opt2, src2)
end

@testset "Reals via x+ - x- split" begin
        for T in (Float64, BigFloat)
            src = MOI.Utilities.Model{T}()
            x, reals_ci = MOI.add_constrained_variables(src, MOI.Reals(2))
            # min x1 - x2  s.t.  x1 - x2 = 1, x free. Bounded (the objective
            # depends only on the constrained difference); optimum 1.
            MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(T[1, -1], x), zero(T)))
            MOI.add_constraint(src, MOI.ScalarAffineFunction(
                MOI.ScalarAffineTerm.(T[1, -1], x), zero(T)), MOI.EqualTo(one(T)))

            opt = SDPA.Optimizer{T}(silent=true)
            MOI.optimize!(opt, src)
            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
            @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 1.0 atol=T(1e-6)
            x1 = MOI.get(opt, MOI.VariablePrimal(), x[1])
            x2 = MOI.get(opt, MOI.VariablePrimal(), x[2])
            @test x1 - x2 ≈ 1.0 atol=T(1e-6)
            # The Reals cone CI exposes the reconstructed free vector and a
            # zero dual.
            cp = MOI.get(opt, MOI.ConstraintPrimal(), reals_ci)
            @test cp isa Vector{T} && length(cp) == 2
            @test cp[1] == x1 && cp[2] == x2
            cd = MOI.get(opt, MOI.ConstraintDual(), reals_ci)
            @test cd isa Vector{T} && all(iszero, cd)
        end
    end

@testset "RawOptimizerAttribute supports reflects known fields" begin
    opt = SDPA.Optimizer{Float64}(silent=true)
    @test MOI.supports(opt, MOI.RawOptimizerAttribute("maxIteration"))
    @test MOI.supports(opt, MOI.RawOptimizerAttribute("betaStar"))
    @test !MOI.supports(opt, MOI.RawOptimizerAttribute("garbage"))
    # The set path still rejects unknown names with UnsupportedAttribute.
    @test_throws MOI.UnsupportedAttribute MOI.set(
        opt, MOI.RawOptimizerAttribute("garbage"), 1)
end

@testset "variable linking requires identity equalities" begin
    # Only ±1·v1 ∓ 1·v2 == 0 is a
    # genuine identity; anything else is a real constraint row.

    @testset "nonzero rhs must not link (X[1,1] - f == 3)" begin
        # min f  s.t.  X[1,1] - f == 3, X ⪰ 0  →  f = X[1,1] - 3 ≥ -3.
        src = MOI.Utilities.Model{Float64}()
        X, _ = MOI.add_constrained_variables(src, MOI.PositiveSemidefiniteConeTriangle(2))
        f, _ = MOI.add_constrained_variables(src, MOI.Reals(1))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, -1.0], [X[1], f[1]]), 0.0),
            MOI.EqualTo(3.0))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], [f[1]]), 0.0))
        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.optimize!(opt, src)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ -3.0 atol=1e-6
        @test MOI.get(opt, MOI.VariablePrimal(), f[1]) ≈ -3.0 atol=1e-6
    end

    @testset "scaled coefficients must not link (2*X[1,1] - f == 0)" begin
        # max f  s.t.  2*X[1,1] - f == 0, X[1,1] + s == 5, s ≥ 0
        # → f = 2·X[1,1] ≤ 10.
        src = MOI.Utilities.Model{Float64}()
        X, _ = MOI.add_constrained_variables(src, MOI.PositiveSemidefiniteConeTriangle(2))
        f, _ = MOI.add_constrained_variables(src, MOI.Reals(1))
        s, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(1))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([2.0, -1.0], [X[1], f[1]]), 0.0),
            MOI.EqualTo(0.0))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], [X[1], s[1]]), 0.0),
            MOI.EqualTo(5.0))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], [f[1]]), 0.0))
        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.optimize!(opt, src)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 10.0 atol=1e-6
        @test MOI.get(opt, MOI.VariablePrimal(), f[1]) ≈ 10.0 atol=1e-6
    end

    @testset "identity equality still links (X[1,1] - f == 0)" begin
        # The legitimate Convex/JuMP auxiliary pattern must keep working:
        # max f  s.t.  X[1,1] - f == 0, X[1,1] + s == 5, s ≥ 0  →  f = 5.
        src = MOI.Utilities.Model{Float64}()
        X, _ = MOI.add_constrained_variables(src, MOI.PositiveSemidefiniteConeTriangle(2))
        f, _ = MOI.add_constrained_variables(src, MOI.Reals(1))
        s, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(1))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, -1.0], [X[1], f[1]]), 0.0),
            MOI.EqualTo(0.0))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], [X[1], s[1]]), 0.0),
            MOI.EqualTo(5.0))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], [f[1]]), 0.0))
        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.optimize!(opt, src)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 5.0 atol=1e-6
    end
end

@testset "duplicate objective terms accumulate" begin
    # Non-canonical objectives may repeat a variable; repeated terms
    # must ACCUMULATE. max (2+3)*X[1,1] s.t. X11+X22 == 1, X ⪰ 0
    # → optimum 5.
    src = MOI.Utilities.Model{Float64}()
    X, _ = MOI.add_constrained_variables(src, MOI.PositiveSemidefiniteConeTriangle(2))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], [X[1], X[3]]), 0.0), MOI.EqualTo(1.0))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([2.0, 3.0], [X[1], X[1]]), 0.0))
    opt = SDPA.Optimizer{Float64}(silent=true)
    MOI.optimize!(opt, src)
    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 5.0 atol=1e-6
end

@testset "0.5 symmetric-fill factor on variable coefficients" begin
    # Locks the /2 storage convention on the two VARIABLE-coefficient
    # paths (equality terms and objective terms on an off-diagonal PSD
    # triangle variable). All other in-repo SDP tests use diagonal
    # entries only; a dropped /2 would double the effective coefficient
    # and silently move the optimum.

    @testset "equality on X[1,2]: min t s.t. X11==1, X12+X22==1" begin
        # X = [1 t; t b] ⪰ 0 with b = 1-t needs 1-t >= t^2, i.e.
        # t >= -(1+√5)/2. Wrong factor → effective 2t+b=1 → min -(1+√2).
        src = MOI.Utilities.Model{Float64}()
        X, _ = MOI.add_constrained_variables(src, MOI.PositiveSemidefiniteConeTriangle(2))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0], [X[1]]), 0.0), MOI.EqualTo(1.0))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], [X[2], X[3]]), 0.0), MOI.EqualTo(1.0))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], [X[2]]), 0.0))
        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.optimize!(opt, src)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ -(1 + √5) / 2 atol=1e-6
    end

    @testset "objective on X[1,2]: max t s.t. X11==1, X12+X22==1" begin
        # Symmetric twin for the objective C path: max t with 1-t >= t^2
        # is t* = (√5-1)/2. A dropped /2 in C reports 2·t* instead.
        src = MOI.Utilities.Model{Float64}()
        X, _ = MOI.add_constrained_variables(src, MOI.PositiveSemidefiniteConeTriangle(2))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0], [X[1]]), 0.0), MOI.EqualTo(1.0))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], [X[2], X[3]]), 0.0), MOI.EqualTo(1.0))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], [X[2]]), 0.0))
        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.optimize!(opt, src)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ (√5 - 1) / 2 atol=1e-6
    end
end

@testset "PSD(1) var linked to SDP off-diagonal pins the /2 symmetric fill" begin
    # Covers the /2 symmetric-fill interaction: a PSD(1)-routed-to-LP variable
    # interacting with a 2×2 PSD OFF-DIAGONAL through an equality. The /2
    # symmetric-fill factor on the SDP side of the equality must match the
    # un-scaled LP side. max t s.t. X11==1, X22==1, X12−t==0: t is an LP
    # cone (PSD(1) → t ≥ 0) and X = [1 u; u 1] ⪰ 0 ⟺ |u| ≤ 1, so
    # t* = u* = 1. A dropped /2 would store ±1 at BOTH (1,2) and (2,1),
    # linking t = 2·X12 and moving the optimum to 2. (The MIN twin is not
    # discriminating: t ≥ 0 forces t = X12 ≥ 0 either way, so both factors
    # give 0 — only the MAX form separates 1 from 2.)
    src = MOI.Utilities.Model{Float64}()
    X, _ = MOI.add_constrained_variables(src, MOI.PositiveSemidefiniteConeTriangle(2))
    t, ci_psd1 = MOI.add_constrained_variables(src, MOI.PositiveSemidefiniteConeTriangle(1))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0], [X[1]]), 0.0), MOI.EqualTo(1.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0], [X[3]]), 0.0), MOI.EqualTo(1.0))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, -1.0], [X[2], t[1]]), 0.0), MOI.EqualTo(0.0))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], [t[1]]), 0.0))
    opt = SDPA.Optimizer{Float64}(silent=true)
    MOI.optimize!(opt, src)
    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 1.0 atol=1e-6

    # The PSD(1) CI reports its own single LP value (reroute visibility).
    cp = MOI.get(opt, MOI.ConstraintPrimal(), ci_psd1)
    @test cp isa Vector{Float64} && length(cp) == 1
    @test cp ≈ [1.0] atol=1e-6

    # The linked SDP off-diagonal carries the other half of the equality.
    @test MOI.get(opt, MOI.VariablePrimal(), X[2]) ≈ 1.0 atol=1e-6
end

@testset "one variable, two PSD affine constraints rejected" begin
    # v sits at the (1,1) triangle position of TWO affine-PSD
    # constraints: ([v 0;0 1] ⪰ 0) and ([v 0;0 2] ⪰ 0). The conflicting
    # var_map writes must become a loud MOI-contract rejection (not a
    # silent overwrite whose winner depends on iteration order); the
    # bridged (VectorFunctionizeBridge) path still solves such models
    # correctly via fresh aux variables.
    src = MOI.Utilities.Model{Float64}()
    v = MOI.add_variable(src)
    # Triangle outputs for PSDTriangle(2): 1→(1,1), 2→(1,2), 3→(2,2).
    f1 = MOI.VectorAffineFunction(
        [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, v))],
        [0.0, 0.0, 1.0])
    MOI.add_constraint(src, f1, MOI.PositiveSemidefiniteConeTriangle(2))
    f2 = MOI.VectorAffineFunction(
        [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, v))],
        [0.0, 0.0, 2.0])
    MOI.add_constraint(src, f2, MOI.PositiveSemidefiniteConeTriangle(2))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], [v]), 0.0))

    opt = SDPA.Optimizer{Float64}(silent=true)
    @test_throws MOI.UnsupportedConstraint MOI.optimize!(opt, src)
end

@testset "single PSD affine constraint still solves (guard does not over-reject)" begin
    # Positive control: max v s.t. [v 0; 0 1] ⪰ 0, v == 2 → v* = 2 with
    # X = [2 0; 0 1] ≻ 0 kept strictly definite on purpose (the direct
    # VAF ingest path stalls at dFEAS on boundary-touching optima).
    src = MOI.Utilities.Model{Float64}()
    v = MOI.add_variable(src)
    f = MOI.VectorAffineFunction(
        [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, v))],
        [0.0, 0.0, 1.0])
    MOI.add_constraint(src, f, MOI.PositiveSemidefiniteConeTriangle(2))
    MOI.add_constraint(src, MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0], [v]), 0.0), MOI.EqualTo(2.0))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], [v]), 0.0))

    opt = SDPA.Optimizer{Float64}(silent=true)
    MOI.optimize!(opt, src)
    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 2.0 atol=1e-6
    @test MOI.get(opt, MOI.VariablePrimal(), v) ≈ 2.0 atol=1e-6
end

@testset "cone variable also in PSD affine constraint rejected" begin
    # v in a Nonnegatives cone AND at a triangle position of an
    # affine-PSD constraint: the cone pass and the VAF pass both claim
    # v with different targets (:LP vs :SDP) → collision → rejection
    # regardless of ListOfConstraintTypesPresent iteration order (the
    # cone pass assigns after the whole discovery loop).
    src = MOI.Utilities.Model{Float64}()
    v, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(1))
    f = MOI.VectorAffineFunction(
        [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, v[1]))],
        [0.0, 0.0, 1.0])
    MOI.add_constraint(src, f, MOI.PositiveSemidefiniteConeTriangle(2))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], [v[1]]), 0.0))

    opt = SDPA.Optimizer{Float64}(silent=true)
    @test_throws MOI.UnsupportedConstraint MOI.optimize!(opt, src)
end

@testset "idempotent same-target re-assignment allowed" begin
    # The guard must not over-reject LEGITIMATE duplicate writes of the
    # SAME target (existing != target is the throw condition).
    var_map = Dict{MOI.VariableIndex,Tuple{Symbol,Int,Int,Int}}()
    vi = MOI.VariableIndex(1)
    SDPA._assign_var_map!(var_map, vi, (:SDP, 1, 1, 1), Float64)
    # Same target twice: idempotent, no throw.
    SDPA._assign_var_map!(var_map, vi, (:SDP, 1, 1, 1), Float64)
    @test var_map[vi] == (:SDP, 1, 1, 1)
    # Different target: throws the MOI-contract error.
    @test_throws MOI.UnsupportedConstraint SDPA._assign_var_map!(
        var_map, vi, (:LP, 1, 1, 0), Float64)
    @test_throws MOI.UnsupportedConstraint SDPA._assign_var_map!(
        var_map, vi, (:SDP, 2, 1, 1), Float64)
    # A fresh variable is unaffected.
    SDPA._assign_var_map!(var_map, MOI.VariableIndex(2), (:LP, 1, 1, 0), Float64)
    @test var_map[MOI.VariableIndex(2)] == (:LP, 1, 1, 0)
end

@testset "equality ConstraintPrimal reports the activity, not b" begin
    # The equality branch returns the constraint function value at the
    # solution point: Σ coeff·x + const (const = the SAF function
    # constant), not the RHS b — b would claim the constraint is
    # satisfied EXACTLY regardless of where the solve stopped (fiction
    # for infeasible/limit terminations).

    # Shared helper: min x1 + x2  s.t.  Σ == rhs,  x ≥ 0.
    function build_eq_lp(rhs::Float64, func_const::Float64=0.0)
        src = MOI.Utilities.Model{Float64}()
        x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(
                    MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0))
        ci = MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], x), func_const),
            MOI.EqualTo(rhs))
        return src, x, ci
    end

    @testset "solved to optimality: activity ≈ b" begin
        src, x, ci = build_eq_lp(1.0)
        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.optimize!(opt, src)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        activity = MOI.get(opt, MOI.ConstraintPrimal(), ci)
        # A well-solved model still looks feasible: activity ≈ b.
        @test activity ≈ 1.0 atol=1e-6
        # The activity IS the function value at the solution: x1 + x2.
        @test activity ≈ MOI.get(opt, MOI.VariablePrimal(), x[1]) +
                         MOI.get(opt, MOI.VariablePrimal(), x[2]) atol=1e-9
    end

    @testset "INFEASIBLE termination: activity is the iterate's value, NOT b" begin
        # THE distinguishing test. x1+x2 == 1 AND x1+x2 == 2 cannot both
        # hold; the solve stops on an infeasibility certificate (which
        # keeps ResultCount == 1 via the dual side), so ConstraintPrimal
        # is queryable — and must report the CURRENT iterate's activity,
        # not a fabricated "exactly 1.0" / "exactly 2.0".
        src = MOI.Utilities.Model{Float64}()
        x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(
                    MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0))
        ci1 = MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0), MOI.EqualTo(1.0))
        ci2 = MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0), MOI.EqualTo(2.0))
        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.optimize!(opt, src)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.INFEASIBLE
        @test MOI.get(opt, MOI.ResultCount()) == 1  # via dual certificate
        x1 = MOI.get(opt, MOI.VariablePrimal(), x[1])
        x2 = MOI.get(opt, MOI.VariablePrimal(), x[2])
        a1 = MOI.get(opt, MOI.ConstraintPrimal(), ci1)
        a2 = MOI.get(opt, MOI.ConstraintPrimal(), ci2)
        # Both report the iterate's ACTUAL x1 + x2 (bitwise: the getter
        # computes exactly 1.0·x1 + 1.0·x2 + 0.0)...
        @test a1 == x1 + x2
        @test a2 == x1 + x2
        # ...which satisfies NEITHER rhs (returning exactly b would be
        # feasibility fiction on an infeasible stop).
        @test abs(a1 - 1.0) > 1e-6
        @test abs(a2 - 2.0) > 1e-6
    end

    @testset "0-iteration TIME_LIMIT: no result point to report" begin
        # TimeLimitSec = 0.0 fires on the first wall-clock check inside
        # the iteration loop: 0 iterations, phase noINFO, xMat still the
        # lambdaStar start point (whitebox-pinned below). With no
        # solution, ResultCount == 0 and the MOI contract makes
        # ConstraintPrimal throw ResultIndexBoundsError — there is no
        # point at which to evaluate the activity. (Checked against the
        # Optimizer's TimeLimitSec plumbing: PrimalStatus/DualStatus are
        # both NO_SOLUTION at phase noINFO.) The activity getter serves
        # every termination that HAS a result — see the INFEASIBLE test
        # above, where a truncated solve's activity is reported.
        src, x, ci = build_eq_lp(1.0)
        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.set(opt, MOI.TimeLimitSec(), 0.0)
        MOI.optimize!(opt, src)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.TIME_LIMIT
        @test MOI.get(opt, MOI.BarrierIterations()) == 0
        @test MOI.get(opt, MOI.ResultCount()) == 0
        # The LP start point is lambdaStar for every variable (so the
        # activity there WOULD be 2·lambdaStar = 200.0 ≠ b = 1 — but it
        # is not a reportable result).
        @test opt.solution_log.xMat.LP_block ==
              [opt.params.lambdaStar, opt.params.lambdaStar]
        @test_throws MOI.ResultIndexBoundsError MOI.get(
            opt, MOI.ConstraintPrimal(), ci)
    end

    @testset "nonzero SAF constant is folded into the model AND the activity" begin
        # Σc·x + 5.0 == 6.0 with Σc·x = x1 + x2. The constant is folded
        # into the solver model (b = 6 - 5 = 1), so the solver enforces
        # x1 + x2 == 1 — the user's actual constraint. VariablePrimal
        # satisfies x1 + x2 ≈ 1, and the activity x1 + x2 + 5 ≈ 6
        # recovers the set rhs.
        src, x, ci = build_eq_lp(6.0, 5.0)
        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.optimize!(opt, src)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.VariablePrimal(), x[1]) +
              MOI.get(opt, MOI.VariablePrimal(), x[2]) ≈ 1.0 atol=1e-6
        activity = MOI.get(opt, MOI.ConstraintPrimal(), ci)
        @test activity ≈ 6.0 atol=1e-6                      # == set rhs
        @test activity ≈ MOI.get(opt, MOI.VariablePrimal(), x[1]) +
                         MOI.get(opt, MOI.VariablePrimal(), x[2]) + 5.0 atol=1e-9
    end

    @testset "nonzero constant moves the optimum" begin
        # min x  s.t.  x + 5 == 7, x ≥ 0  →  x* = 2, obj 2.
        src = MOI.Utilities.Model{Float64}()
        x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(1))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0], x), 5.0), MOI.EqualTo(7.0))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], x), 0.0))
        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.optimize!(opt, src)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 2.0 atol=1e-6
        @test MOI.get(opt, MOI.VariablePrimal(), x[1]) ≈ 2.0 atol=1e-6
    end

    @testset "cancelled infeasible equality is not filtered as redundant" begin
        # 0·x + 5 == 0 (terms cancel, nonzero constant, zero rhs) is
        # INFEASIBLE. The effective rhs is -5 and the row is KEPT as a
        # zero-A row with b = -5. A zero-A row makes the Schur matrix
        # singular, so the solver terminates loudly via the
        # Cholesky-failure ladder (pre-existing zero-A behavior, C++
        # parity) instead of reporting OPTIMAL — that is the guarantee:
        # never silently dropped.
        src = MOI.Utilities.Model{Float64}()
        x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(1))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([0.0], x), 5.0), MOI.EqualTo(0.0))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], x), 0.0))
        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.optimize!(opt, src)
        # The row survived ingest as a zero-A row with b = -5.
        @test length(opt.problem.b.ele) == 1
        @test opt.problem.b.ele[1] ≈ -5.0 atol=1e-12
        # And the solve does NOT claim success on an infeasible model.
        @test MOI.get(opt, MOI.TerminationStatus()) != MOI.OPTIMAL
        @test occursin("CHOLESKY_FAILURE",
                       MOI.get(opt, MOI.RawStatusString()))
    end
end
