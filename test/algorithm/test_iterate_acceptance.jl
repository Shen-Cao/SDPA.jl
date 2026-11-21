@testset "tiny accepted step reports one consistent point" begin
    for T in (Float64, BigFloat)
        # The fraction-to-boundary cap is 100; this γ puts both step
        # lengths below the 1e-4 stagnation safeguard on the first step.
        problem = SDPA.read_dat_s(joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T)
        params = SDPA.Parameters{T}(gammaStar=T(1e-9), precision=precision(BigFloat))
        log = SDPA.solve!(problem, params; verbose=false, reverse_primal_dual=false)
        @test log.exit_reason == SDPA.CANNOT_MOVE
        @test log.iterations == length(log.trajectory) == 1
        @test log.final_obj_primal == SDPA._inner_product_C_xMat(problem, log.xMat)
        @test log.final_obj_dual == SDPA._inner_product_b_yVec(problem, log.yVec)
        @test log.final_mu == SDPA._frobenius_inner_product_dls(log.xMat, log.zMat) /
                              sum(abs, problem.blockStruct)
        point = SDPA.SDPASolution{T}(problem.m, problem.blockStruct, params.lambdaStar)
        point.xMat, point.yVec, point.zMat = log.xMat, log.yVec, log.zMat
        res = SDPA.SDPAResiduals{T}(problem, point)
        SDPA.compute!(res, problem, point)
        @test log.final_primal_res == res.normPrimalVec
        @test log.final_dual_res == res.normDualMat
        @test log.trajectory[1].alphaP < T(1e-4)
        @test log.trajectory[1].alphaD < T(1e-4)
    end
end

@testset "rejected candidate leaves last point and inverse caches usable" begin
    for T in (Float64, BigFloat)
        blocks = [2, -1]
        point = SDPA.SDPASolution{T}(1, blocks, T(1))
        work = SDPA.SDPAWorkVariables{T}(blocks)
        newton = SDPA.NewtonState{T}(1, blocks)
        newton.DxMat.SDP_block[1].de_ele[1, 1] = T(2)
        newton.DyVec.ele[1] = T(9)
        # X factors successfully, then Z fails on its LP component. Both
        # inverse caches must nevertheless agree with the old X and Z.
        newton.DzMat.SDP_block[1].de_ele[1, 1] = T(1)
        newton.DzMat.LP_block[1] = -T(2)^40
        alpha = SDPA.StepLength{T}(T(1), T(1))
        @test !SDPA.update!(point, alpha, newton, work)
        @test iszero(point.yVec.ele[1])
        for M in (point.xMat, point.zMat, point.invCholeskyX, point.invCholeskyZ, point.invzMat)
            @test M.SDP_block[1].de_ele == Matrix{T}(I, 2, 2)
            @test M.LP_block == T[1]
        end
        newton.DzMat.LP_block[1] = T(NaN)
        @test SDPA._update_iterate!(point, alpha, newton, work) === SDPA._NONFINITE_STEP
        @test iszero(point.yVec.ele[1]) && point.zMat.LP_block == T[1]
    end
end

@testset "factorization failure is not stagnation or infeasibility" begin
    # Two identical rows produce the exactly singular Schur matrix [1 1; 1 1].
    T = Float64
    C = SDPA.SDPASparseLinearSpace{T}(0, 1)
    A = [SDPA.SDPASparseLinearSpace{T}(0, 1) for _ in 1:2]
    foreach(a -> SDPA.setelement_lp!(a, 1, one(T)), A)
    problem = SDPA.SDPAProblem{T}(2, 1, [-1], SDPA.SDPAVector{T}(T[0, 0]), C, A)
    log = SDPA.solve!(problem, SDPA.Parameters{T}(lambdaStar=1);
                      verbose=false, reverse_primal_dual=false)
    @test log.exit_reason === SDPA.CHOLESKY_FAILURE
    @test log.iterations == 0 && isempty(log.trajectory)
    @test log.terminal_phase === SDPA.noINFO
    @test log.xMat.LP_block == log.zMat.LP_block == T[1]
    @test log.final_mu == log.final_primal_res == log.final_dual_res == one(T)
    @test log.final_obj_primal == log.final_obj_dual == zero(T)
end
