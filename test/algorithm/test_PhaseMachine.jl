@testset "PhaseMachine" begin
    @testset "all 11 enum values exist" begin
        # Sanity: every enum name resolves.
        @test SDPA.noINFO isa SDPA.PhaseValue
        @test SDPA.pFEAS isa SDPA.PhaseValue
        @test SDPA.dFEAS isa SDPA.PhaseValue
        @test SDPA.pdFEAS isa SDPA.PhaseValue
        @test SDPA.pdINF isa SDPA.PhaseValue
        @test SDPA.pFEAS_dINF isa SDPA.PhaseValue
        @test SDPA.pINF_dFEAS isa SDPA.PhaseValue
        @test SDPA.pdOPT isa SDPA.PhaseValue
        @test SDPA.pUNBD isa SDPA.PhaseValue
        @test SDPA.dUNBD isa SDPA.PhaseValue
        @test SDPA.not_called isa SDPA.PhaseValue
        # 11 distinct values total.
        @test length(instances(SDPA.PhaseValue)) == 11
    end

    @testset "transitions (Float64 + BigFloat)" begin
        for T in (Float64, BigFloat)
            # Default params (per-T presets): BigFloat uses
            # epsilonStar=epsilonDash=1e-30; Float64 uses 1e-7. The probe
            # values below (1e-31 "tiny", 1 "large") sit on the same side
            # of both thresholds, so the transitions hold for either T.
            # lowerBound=-1e5, upperBound=1e5 for both.
            params = SDPA.Parameters{T}()

            phase_ref = Ref{SDPA.PhaseValue}(SDPA.not_called)

            # not_called -> noINFO on first call with bad residuals.
            SDPA.update_check!(phase_ref, T(1), T(1), T(0), T(0), T(0), params)
            @test phase_ref[] == SDPA.noINFO

            # When both residuals are tiny AND PDgap is below epsilonStar
            # (here PDgap = 0), the machine transitions through pdFEAS
            # directly to pdOPT (terminal). Verify that.
            continue_iter = SDPA.update_check!(
                phase_ref, T(1e-31), T(1e-31), T(0), T(0), T(0), params
            )
            @test phase_ref[] == SDPA.pdOPT
            @test continue_iter == false

            # noINFO -> pFEAS (primal only) when dual residual is large.
            # Use objP, objD values that DON'T trigger pUNBD/pINF branches:
            # objP=0 (well above lowerBound=-1e5), rho=0 (well below 1).
            SDPA.update_check!(phase_ref, T(1e-31), T(1), T(0), T(0), T(0), params)
            @test phase_ref[] == SDPA.pFEAS

            # noINFO -> dFEAS (dual only) when primal residual is large.
            SDPA.update_check!(phase_ref, T(1), T(1e-31), T(0), T(0), T(0), params)
            @test phase_ref[] == SDPA.dFEAS

            # pdFEAS state without pdOPT transition: PDgap > epsilonStar
            # but residuals are tiny. Use params with small epsilonStar.
            params2 = SDPA.Parameters{T}(epsilonStar=T(1e-100))
            SDPA.update_check!(
                phase_ref, T(1e-31), T(1e-31), T(1), T(2), T(0), params2
            )
            @test phase_ref[] == SDPA.pdFEAS  # not optimal (PDgap=1 > 1e-100)

            # pdFEAS -> pdINF when both residuals bad AND rho > 1 + 1e-6.
            continue_iter = SDPA.update_check!(
                phase_ref, T(1), T(1), T(0), T(0), T(2), params
            )
            @test phase_ref[] == SDPA.pdINF
            @test continue_iter == false

            # pFEAS -> pFEAS_dINF when primal is feasible but rho > 1.
            continue_iter = SDPA.update_check!(
                phase_ref, T(1e-31), T(1), T(0), T(0), T(2), params
            )
            @test phase_ref[] == SDPA.pFEAS_dINF
            @test continue_iter == false

            # pFEAS -> pUNBD when objValPrimal <= lowerBound.
            continue_iter = SDPA.update_check!(
                phase_ref, T(1e-31), T(1), T(-1e10), T(0), T(0), params
            )
            @test phase_ref[] == SDPA.pUNBD
            @test continue_iter == false

            # dFEAS -> pINF_dFEAS when dual is feasible but rho > 1.
            continue_iter = SDPA.update_check!(
                phase_ref, T(1), T(1e-31), T(0), T(0), T(2), params
            )
            @test phase_ref[] == SDPA.pINF_dFEAS
            @test continue_iter == false

            # dFEAS -> dUNBD when objValDual >= upperBound.
            continue_iter = SDPA.update_check!(
                phase_ref, T(1), T(1e-31), T(0), T(1e10), T(0), params
            )
            @test phase_ref[] == SDPA.dUNBD
            @test continue_iter == false
        end
    end

    @testset "pdOPT boundary is inclusive (<=)" begin
        # When PDgap = 0 exactly, dominator = 1, ratio = 0 ≤ epsilonStar
        # → pdOPT (boundary inclusive on the trivial case).
        for T in (Float64, BigFloat)
            params = SDPA.Parameters{T}()
            phase_ref = Ref{SDPA.PhaseValue}(SDPA.not_called)
            continue_iter = SDPA.update_check!(
                phase_ref, T(1e-31), T(1e-31), T(7), T(7), T(0), params
            )
            @test phase_ref[] == SDPA.pdOPT
            @test continue_iter == false
        end
    end

    @testset "update_check! is type-stable" begin
        params = SDPA.Parameters{Float64}()
        phase_ref = Ref{SDPA.PhaseValue}(SDPA.not_called)
        result = SDPA.update_check!(phase_ref, 1.0, 1.0, 0.0, 0.0, 0.0, params)
        @test result isa Bool
    end
end
