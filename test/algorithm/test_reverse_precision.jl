@testset "raw REVERSE solve uses its requested precision throughout" begin
    # Supply 512-bit problem data from outside the caller's later, much
    # narrower scope. X = 1 is primal feasible and the strictly positive
    # dual slack is already within the requested optimality tolerance.
    problem, params, init, expected_p, expected_d = setprecision(BigFloat, 512) do
        T = BigFloat
        c = one(T) + T(2)^(-400)
        z = T(2)^(-450)
        C = SDPA.SDPASparseLinearSpace{T}(0, 1)
        A = SDPA.SDPASparseLinearSpace{T}(0, 1)
        SDPA.setelement_lp!(C, 1, -c)
        SDPA.setelement_lp!(A, 1, one(T))
        problem = SDPA.SDPAProblem{T}(1, 1, [-1], SDPA.SDPAVector{T}(T[1]), C, [A])
        params = SDPA.Parameters{T}(precision=512)
        init = SDPA.SDPASolution{T}(1, [-1], one(T))
        init.yVec.ele[1] = c - z
        init.zMat.LP_block[1] = z
        @test SDPA.compute_inverse!(init, SDPA.SDPAWorkVariables{T}([-1]))
        problem, params, init, -(c - z), -c
    end

    setprecision(BigFloat, 128) do
        log = SDPA.solve!(problem, params; init, verbose=false)
        @test log.terminal_phase == SDPA.pdOPT && log.iterations == 0
        @test log.final_obj_primal == expected_p
        @test log.final_obj_dual == expected_d
        @test precision(log.final_obj_primal) == precision(log.final_obj_dual) == 512
        @test precision(BigFloat) == 128

        # Check the nonempty trajectory's sign/swap transformation too.
        sample = SDPA.solve!(problem, SDPA.Parameters{BigFloat}(precision=512, maxIteration=1);
                             verbose=false)
        @test length(sample.trajectory) == 1
        @test precision(only(sample.trajectory).objP) == 512
        @test precision(only(sample.trajectory).objD) == 512
    end
end
