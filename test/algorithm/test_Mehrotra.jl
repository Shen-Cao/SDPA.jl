@testset "Mehrotra orchestrator" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            # Build example1 problem and run one full Mehrotra iteration.
            problem = SDPA.read_dat_s(
                joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T
            )
            params = SDPA.Parameters{T}()
            current_pt = SDPA.SDPASolution{T}(problem.m, problem.blockStruct, params.lambdaStar)
            work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)
            newton = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
            res = SDPA.SDPAResiduals{T}(problem, current_pt)
            SDPA.compute!(res, problem, current_pt)

            mu = SDPA.AverageComplementarity{T}(params.lambdaStar)
            SDPA.update!(mu, current_pt)
            beta = SDPA.DirectionParameter{T}(params.betaStar)
            reduction = SDPA.Switch(SDPA.REDUCTION_ON)

            # Predictor step.
            SDPA.mehrotra_predictor!(reduction, SDPA.noINFO)
            SDPA.mehrotra_predictor!(beta, SDPA.noINFO, reduction, params)
            ok = SDPA.mehrotra!(
                newton, :predictor, problem, current_pt, res,
                mu, beta, SDPA.noINFO, work,
            )
            @test ok == true

            # Verify predictor produced finite (non-NaN) directions.
            @test all(isfinite, newton.DyVec.ele)
            for k in 1:newton.DxMat.SDP_nBlock
                @test all(isfinite, newton.DxMat.SDP_block[k].de_ele)
                @test all(isfinite, newton.DzMat.SDP_block[k].de_ele)
            end

            # Cholesky factorization should be cached after predictor.
            @test newton.cholesky_factorization !== nothing

            # Corrector step (reuses factorization).
            # Use a hand-crafted alpha for the corrector beta computation.
            alpha_primal = T(0.5)
            alpha_dual = T(0.5)
            SDPA.mehrotra_corrector!(
                beta, SDPA.noINFO, alpha_primal, alpha_dual,
                current_pt, newton.DxMat, newton.DzMat, mu, params,
            )
            ok2 = SDPA.mehrotra!(
                newton, :corrector, problem, current_pt, res,
                mu, beta, SDPA.noINFO, work,
            )
            @test ok2 == true

            # Verify corrector also produced finite directions.
            @test all(isfinite, newton.DyVec.ele)
            for k in 1:newton.DxMat.SDP_nBlock
                @test all(isfinite, newton.DxMat.SDP_block[k].de_ele)
                @test all(isfinite, newton.DzMat.SDP_block[k].de_ele)
            end
        end
    end
end
