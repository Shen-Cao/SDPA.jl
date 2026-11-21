function _manual_mixed_storage_problem(::Type{T}; repeat_sparse=false) where {T<:Real}
    # x₁₁ = 1, tr(X) = 3, and each of fourteen independent LP entries is 1.
    # The objective is 2x₁₁ + x₂₂ + sum(lp) = 18.
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
    repeats = repeat_sparse ? 2 : 1
    for _ in 1:repeats
        SDPA.setelement_sdp!(A[2], 1, 1, 1, one(T) / repeats)
        SDPA.setelement_sdp!(A[2], 1, 2, 2, one(T) / repeats)
    end
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

        # force_sparse alone can fall back to dense, depending on the
        # width of the Schur profile in the ordering chosen by METIS.
        params = SDPA.Parameters{T}(lambdaStar=one(T))
        p_dense = _manual_mixed_storage_problem(T)  # lazy index construction
        dense = SDPA.solve!(p_dense, params; verbose=false, reverse_primal_dual=false)
        sparse = withenv("SDPA_FORCE_SPARSE_PROFILE" => "1") do
            # Both messages mark actual profile allocation, below or
            # above the density gate, not just entry to sparsity routing.
            @test_logs (:info, r"Block-level profile:|FORCED sparse profile") match_mode=:any SDPA.solve!(
                problem, params; verbose=true, io=devnull, reverse_primal_dual=false,
                force_sparse=true)
        end
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

@testset "Repeated sparse triplets can own a dense/sparse Schur pair" begin
    for T in (Float64, BigFloat)
        problem = _manual_mixed_storage_problem(T; repeat_sparse=true)
        SDPA.initialize_index!(problem)
        # Equal weights and the larger constraint index give the sparse
        # block ownership of the pair. F3 cannot consume the dense neighbor.
        @test problem.A[1].SDP_sp_block[1].nonZeroEffect ==
              problem.A[2].SDP_sp_block[1].nonZeroEffect == 4
        newton = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
        work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)
        pt = SDPA.SDPASolution{T}(problem.m, problem.blockStruct, one(T))
        SDPA.compute_formula!(newton, problem)
        @test newton.useFormula[2, 1] in (:F1, :F2)
        SDPA.compute_bmat_dense!(newton, problem, pt, work)
        @test newton.bmat.de_ele[1:2, 1:2] == T[1 1; 1 2]
    end
end
