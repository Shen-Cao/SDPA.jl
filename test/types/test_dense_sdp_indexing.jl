function _manual_mixed_storage_problem(::Type{T}) where {T<:Real}
    # x₁₁ = 1, tr(X) = 3, and each of fourteen independent LP entries is 1.
    # The objective is 2x₁₁ + x₂₂ + sum(lp) = 18. The single coupled pair
    # stays under the sparse profile's density gate in any ordering.
    m, nlp = 16, 14
    blocks = [2, -nlp]
    function space(kind)
        S = SDPA.SDPASparseLinearSpace{T}(1, nlp)
        S.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, kind)
        return S
    end
    C = space(:dense)
    SDPA.setelement_sdp!(C, 1, 1, 1, T(2))
    SDPA.setelement_sdp!(C, 1, 2, 2, one(T))
    A = [space(i == 2 || i == 3 ? :sparse : :dense) for i in 1:m]
    # Exercise raw dense writes as well as the setter, with no prior
    # sparse-to-dense conversion to populate the structural metadata.
    A[1].SDP_sp_block[1].de_ele[1, 1] = one(T)
    SDPA.setelement_sdp!(A[2], 1, 1, 1, one(T))
    SDPA.setelement_sdp!(A[2], 1, 2, 2, one(T))
    SDPA.setelement_sdp!(A[3], 1, 1, 1, one(T))
    SDPA.change_to_dense!(A[3].SDP_sp_block[1]; force=true)
    SDPA.setzero!(A[3].SDP_sp_block[1])  # a converted but now empty block
    for j in 1:nlp
        SDPA.setelement_lp!(C, j, one(T))
        SDPA.setelement_lp!(A[j + 2], j, one(T))
    end
    b = SDPA.SDPAVector{T}(T[1, 3, fill(one(T), nlp)...])
    return SDPA.SDPAProblem{T}(m, 2, blocks, b, C, A)
end

@testset "Direct dense SDP constraints participate in Schur assembly" begin
    for T in (Float64, BigFloat)
        problem = _manual_mixed_storage_problem(T)
        SDPA.initialize_index!(problem)
        @test problem.SDP_constraint[1] == [1, 2]
        @test problem.SDP_blockIndex[1] == [1, 1]
        @test problem.SDP_nConstraint == [2, 0]
        @test problem.A[1].SDP_sp_block[1].nonZeroEffect ==
              problem.A[3].SDP_sp_block[1].nonZeroEffect == 4

        newton = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
        work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)
        pt = SDPA.SDPASolution{T}(problem.m, problem.blockStruct, one(T))
        SDPA.compute_formula!(newton, problem)
        SDPA.compute_bmat_dense!(newton, problem, pt, work)
        @test newton.bmat.de_ele[1:2, 1:2] == T[1 1; 1 2]

        # Require the sparse assembly to actually run: force_sparse alone
        # can fall back to dense when the block profile exceeds its gate.
        params = SDPA.Parameters{T}(lambdaStar=one(T))
        p_dense = _manual_mixed_storage_problem(T)  # lazy index construction
        dense = SDPA.solve!(p_dense, params; verbose=false, reverse_primal_dual=false)
        sparse = @test_logs (:info, r"Block-level profile:") match_mode=:any SDPA.solve!(
            problem, params; verbose=true, io=devnull, reverse_primal_dual=false,
            force_sparse=true)
        @test dense.terminal_phase == sparse.terminal_phase == SDPA.pdOPT
        for log in (dense, sparse)
            @test isapprox(log.final_obj_primal, T(18); rtol=params.epsilonStar)
            @test isapprox(log.final_obj_dual, T(18); rtol=params.epsilonStar)
        end

        # Rebuilding after editing the authoritative dense payload removes
        # and adds memberships without relying on stale cached counts.
        SDPA.setzero!(problem.A[1].SDP_sp_block[1])
        problem.A[4].SDP_sp_block[1].de_ele[2, 2] = one(T)
        SDPA.initialize_index!(problem)
        @test problem.SDP_constraint[1] == [2, 4]
    end
end
