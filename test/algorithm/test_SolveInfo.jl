@testset "SolveInfo/AverageComplementarity/Ratio/Switch" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            # Build a small problem + iterate so we can compute things.
            m = 2
            blockStruct = [2]
            b = SDPA.SDPAVector{T}([T(1), T(4)])
            C = SDPA.SDPASparseLinearSpace{T}(1, 0)
            C.SDP_sp_index = [1]
            C.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 2)
            SDPA.setelement_sdp!(C, 1, 1, 1, T(5))
            SDPA.setelement_sdp!(C, 1, 2, 2, T(7))
            A1 = SDPA.SDPASparseLinearSpace{T}(1, 0)
            A1.SDP_sp_index = [1]
            A1.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
            SDPA.setelement_sdp!(A1, 1, 1, 1, T(1))
            A2 = SDPA.SDPASparseLinearSpace{T}(1, 0)
            A2.SDP_sp_index = [1]
            A2.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
            SDPA.setelement_sdp!(A2, 1, 2, 2, T(2))
            problem = SDPA.SDPAProblem{T}(m, 1, blockStruct, b, C, [A1, A2])
            SDPA.initialize_index!(problem)

            lambda = T(10000)
            sol = SDPA.SDPASolution{T}(m, blockStruct, lambda)

            # AverageComplementarity
            mu = SDPA.AverageComplementarity{T}(lambda)
            @test mu.initial == lambda * lambda
            @test mu.current == lambda * lambda
            SDPA.update!(mu, sol)
            # mu = <X,Z>/nDim = <10000*I, 10000*I>/2 = (2 * 1e8)/2 = 1e8
            @test mu.current == lambda * lambda

            # Switch
            sw = SDPA.Switch(SDPA.REDUCTION_ON)
            @test sw.switchType == SDPA.REDUCTION_ON
            SDPA.mehrotra_predictor!(sw, SDPA.noINFO)
            @test sw.switchType == SDPA.REDUCTION_ON
            SDPA.mehrotra_predictor!(sw, SDPA.pFEAS)
            @test sw.switchType == SDPA.REDUCTION_ON
            SDPA.mehrotra_predictor!(sw, SDPA.dFEAS)
            @test sw.switchType == SDPA.REDUCTION_ON
            SDPA.mehrotra_predictor!(sw, SDPA.pdFEAS)
            @test sw.switchType == SDPA.REDUCTION_OFF
            SDPA.mehrotra_predictor!(sw, SDPA.pdOPT)
            @test sw.switchType == SDPA.REDUCTION_OFF

            # RatioInitResCurrentRes
            res = SDPA.SDPAResiduals{T}(problem, sol)
            SDPA.compute!(res, problem, sol)
            params = SDPA.Parameters{T}(epsilonDash=T(1e-30))
            theta = SDPA.RatioInitResCurrentRes{T}(params, res)
            # Both residuals are large (1e4 - tiny), so both ratios = 1.
            @test theta.primal == T(1)
            @test theta.dual   == T(1)

            # Update with switch ON: theta *= |1 - alpha|
            SDPA.update!(theta, SDPA.Switch(SDPA.REDUCTION_ON), T(0.5), T(0.5))
            @test theta.primal == T(0.5)
            @test theta.dual   == T(0.5)

            # Update_exact!: with bad init_res (~1e4) and same current_res, ratio = 1
            res2 = SDPA.SDPAResiduals{T}(problem, sol)
            SDPA.compute!(res2, problem, sol)
            SDPA.update_exact!(theta, res, res2)
            # theta = current_res / init_res = 1
            @test theta.primal == T(1)
            @test theta.dual   == T(1)

            # SolveInfo
            omega_star = params.omegaStar
            solve_info = SDPA.SolveInfo{T}(problem, sol, mu.initial, omega_star)
            # objValPrimal = <C, xMat> = 5*1e4 + 7*1e4 = 12e4
            @test solve_info.objValPrimal == T(12) * T(10000)
            # objValDual = <b, yVec> = 0 (yVec = 0)
            @test solve_info.objValDual == zero(T)
            @test solve_info.rho == one(T)

            # Update with current residuals (both bad — rho stays 0 per
            # the "both infeasible" branch when lambda*Z is the dominant term).
            SDPA.update!(solve_info, lambda, problem, sol, res, mu, theta, params)
            # After update, objValPrimal/Dual should be unchanged.
            @test solve_info.objValPrimal == T(12) * T(10000)
            @test solve_info.objValDual   == zero(T)
            # rho formula has many variables; just check it's a nonneg real.
            @test solve_info.rho >= zero(T)

            # --- Degenerate convergence: rho denominator exactly zero ---
            # Branch 1 (primal infeasible, dual feasible) with a zero rho
            # denominator: mu.current = 0 → xzMat = 0, etaDual = 0, and
            # theta.primal = 0 → numerator 0 (0/0 = NaN pre-guard).
            res.normPrimalVec = one(T)
            res.normDualMat = zero(T)
            mu.current = zero(T)
            solve_info.etaDual = zero(T)
            theta.primal = zero(T)
            SDPA.update!(solve_info, lambda, problem, sol, res, mu, theta, params)
            # Guard fires: rho = typemax(T) (Inf sentinel), NOT NaN.
            @test solve_info.rho == typemax(T)
            @test !isnan(solve_info.rho)
        end
    end
end
