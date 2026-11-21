@testset "a negative mode orthogonal to the Lanczos start cannot leave the cone" begin
    # D = I - 2(e1-e2)(e1-e2)'. Its all-ones mode is +1 even though
    # its smallest eigenvalue is -3. Starting at X=I, a step of 1 would
    # cross the true boundary at 1/3. Replaying this overestimate tests
    # recovery independently of future changes to the proposal heuristic.
    for T in (Float64, BigFloat)
        n = 64
        point = SDPA.SDPASolution{T}(1, [n], T(1))
        newton = SDPA.NewtonState{T}(1, [n])
        D = newton.DxMat.SDP_block[1].de_ele
        for i in 1:n
            D[i, i] = i <= 2 ? -one(T) : one(T)
        end
        D[1, 2], D[2, 1] = T(2), T(2)
        newton.DyVec.ele[1] = T(3)
        alpha = SDPA.StepLength{T}(one(T), one(T))
        @test SDPA.update!(point, alpha, newton, SDPA.SDPAWorkVariables{T}([n]))
        @test alpha.primal == alpha.dual == T(1)/4
        @test point.xMat.SDP_block[1].de_ele == Matrix{T}(I, n, n) + alpha.primal * D
        @test point.yVec.ele == T[3alpha.dual]
        @test isposdef(Symmetric(point.xMat.SDP_block[1].de_ele))
        @test isposdef(Symmetric(point.zMat.SDP_block[1].de_ele))
    end
end

@testset "trace SDP logs the step that reached the interior" begin
    # A bounded problem with the same hidden curvature: min <D,X> with
    # tr(X)=64, X positive semidefinite. Its first unshortened corrector
    # step loses dual positive definiteness.
    T = Float64
    n = 64
    C = SDPA.SDPASparseLinearSpace{T}(1, 0)
    A = SDPA.SDPASparseLinearSpace{T}(1, 0)
    C.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(n, n, :sparse, n+1)
    A.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(n, n, :sparse, n)
    for i in 1:n
        SDPA.setelement_sdp!(A, 1, i, i, 1.0)
        SDPA.setelement_sdp!(C, 1, i, i, i <= 2 ? -1.0 : 1.0)
    end
    SDPA.setelement_sdp!(C, 1, 1, 2, 2.0)
    problem = SDPA.SDPAProblem{T}(1, 1, [n], SDPA.SDPAVector{T}(T[n]), C, [A])
    log = SDPA.solve!(problem, SDPA.Parameters{T}(lambdaStar=1, maxIteration=1);
                      verbose=false, reverse_primal_dual=false)
    @test log.exit_reason === SDPA.MAX_ITERATION && log.iterations == 1
    row = only(log.trajectory)
    @test 0 < row.alphaD < 1
    # Dual residual is corrected on this step. The (1,2) entry of Z0
    # and A is zero, hence Z_new[1,2] must equal the *applied* αD C[1,2].
    @test log.zMat.SDP_block[1].de_ele[1,2] == 2row.alphaD
    @test isposdef(Symmetric(log.xMat.SDP_block[1].de_ele))
    @test isposdef(Symmetric(log.zMat.SDP_block[1].de_ele))
end
