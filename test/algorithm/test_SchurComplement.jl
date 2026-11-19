# _profile_from_dense: canonical definition in test/_profile_from_dense.jl,
# shared (single definition per process) with the sparse test files.
isdefined(@__MODULE__, :_profile_from_dense) ||
    include(joinpath(@__DIR__, "..", "_profile_from_dense.jl"))

@testset "SchurComplement" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            # Build a small 1-block SDP problem for testing.
            # blockStruct = [2], m = 2
            # C = diag(5, 7), A_1 = diag(1, 0), A_2 = [[0,1],[1,0]]
            blockStruct = [2]
            b = SDPA.SDPAVector{T}([T(1), T(1)])
            C = SDPA.SDPASparseLinearSpace{T}(1, 0)
            C.SDP_sp_index = [1]
            C.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 2)
            SDPA.setelement_sdp!(C, 1, 1, 1, T(5))
            SDPA.setelement_sdp!(C, 1, 2, 2, T(7))
            A1 = SDPA.SDPASparseLinearSpace{T}(1, 0)
            A1.SDP_sp_index = [1]
            A1.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 2)
            SDPA.setelement_sdp!(A1, 1, 1, 1, T(1))
            SDPA.setelement_sdp!(A1, 1, 2, 2, T(0))
            A2 = SDPA.SDPASparseLinearSpace{T}(1, 0)
            A2.SDP_sp_index = [1]
            A2.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 2)
            SDPA.setelement_sdp!(A2, 1, 1, 2, T(1))
            SDPA.setelement_sdp!(A2, 1, 2, 1, T(1))
            problem = SDPA.SDPAProblem{T}(2, 1, blockStruct, b, C, [A1, A2])
            SDPA.initialize_index!(problem)

            # Force dense form for all blocks (so F1 is selected).
            SDPA.change_to_dense!(problem.C; force=true)
            for Ai in problem.A
                SDPA.change_to_dense!(Ai; force=true)
            end

            # Set up current point with X = Z = I (lambdaStar = 1).
            current_pt = SDPA.SDPASolution{T}(2, blockStruct, T(1))
            params = SDPA.Parameters{T}()
            work = SDPA.SDPAWorkVariables{T}(blockStruct)
            newton = SDPA.NewtonState{T}(2, blockStruct)

            # compute_formula!: all blocks are dense → F1 or F2 (never F3).
            SDPA.compute_formula!(newton, problem)
            @test all(f === :F1 || f === :F2 for f in newton.useFormula)

            # compute_bmat_dense!: X = Z = I → Z⁻¹ = I.
            # B[i,j] = ⟨X · A_i · Z⁻¹, A_j⟩ = ⟨A_i, A_j⟩
            # A_1 = diag(1,0); A_2 = [[0,1],[1,0]]
            # ⟨A_1, A_1⟩ = 1; ⟨A_1, A_2⟩ = 0; ⟨A_2, A_2⟩ = 2
            SDPA.compute_bmat_dense!(newton, problem, current_pt, work)
            B = newton.bmat.de_ele
            @test B[1, 1] ≈ T(1) rtol=eps(T)^0.5
            @test B[1, 2] ≈ T(0) atol=eps(T)^0.5
            @test B[2, 2] ≈ T(2) rtol=eps(T)^0.5
            @test B[2, 1] ≈ B[1, 2] atol=eps(T)^0.5  # symmetric

            # Verify F1 directly: cal_f1!(G, Aj) where G = X*A_i*Z⁻¹.
            # For X=Z=I, G = A_i.
            # cal_f1!(A_1, A_1) should = ⟨A_1, A_1⟩ = 1.
            G1 = copy(A1.SDP_sp_block[1].de_ele)
            @test SDPA.cal_f1!(G1, problem.A[1].SDP_sp_block[1]) ≈ T(1) rtol=eps(T)^0.5

            # Test compute_rmat! predictor: r_zinv = βμ·Z⁻¹ - X = βμ·I - I
            mu = SDPA.AverageComplementarity{T}(T(1))
            mu.current = T(1)
            beta = SDPA.DirectionParameter{T}(T(0))
            SDPA.compute_rmat!(newton, :predictor, mu, beta, current_pt, work)
            # r_zinv = 0·I - I = -I
            @test newton.r_zinvMat.SDP_block[1].de_ele[1, 1] ≈ -T(1) rtol=eps(T)^0.5
            @test newton.r_zinvMat.SDP_block[1].de_ele[2, 2] ≈ -T(1) rtol=eps(T)^0.5
        end
    end
end

@testset "compute_formula! never selects F3 on dense blocks" begin
    problem = SDPA.read_dat_s(
        joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), Float64)
    # Blocks are already dense from _finalize_problem! (example1 has one
    # 2x2 SDP block, above the 20% density threshold).
    ns = SDPA.NewtonState{Float64}(problem.m, problem.blockStruct)
    SDPA.compute_formula!(ns, problem)
    # All formulas must be F1 or F2 (F3 never for dense blocks)
    @test all(f in (:F1, :F2) for f in ns.useFormula)
    # Manually corrupt: force F3 on a dense-block constraint
    ns.useFormula[1, 1] = :F3
    @test_throws AssertionError SDPA._check_no_F3_on_dense!(problem, ns)
    # Positive: restore valid formula
    ns.useFormula[1, 1] = :F1
    SDPA._check_no_F3_on_dense!(problem, ns)
end

# ---------------------------------------------------------------------------
# Location table correctness tests.
# ---------------------------------------------------------------------------
# Exercises make_aggrigate_index_sdp! directly. Verifies:
#   1. The 5 parallel arrays have equal length.
#   2. Length matches the per-block sum recorded in SDP_block_table_len.
#   3. Every SDP_location_sparse_bMat[k] is within [1, nnz].
#   4. Length(diagonalIndex) == m + 1.
#   5. Non-identity ordering is used (proves the table handles AMD).
#   6. The location-index formula matches the by-hand computation:
#      sp_ele[diagonalIndex[ii] + (jj - ii)] where ii=min(p_i,p_j).

@testset "make_aggrigate_index_sdp! location table correctness" begin
    T = Float64
    problem = SDPA.read_dat_s(
        joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T)
    SDPA.initialize_index!(problem)
    ns = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
    SDPA.compute_formula!(ns, problem)

    # Use a non-identity ordering (m=3 -> 6 permutations; pick [2,3,1]).
    @test problem.m == 3
    ordering = [2, 3, 1]
    ns.use_sparse = true
    ns.ordering = ordering
    ns.reverse_ordering = Vector{Int}(undef, problem.m)
    for (pos, orig) in enumerate(ordering)
        ns.reverse_ordering[orig] = pos
    end

    # Build sparse_bmat structure via the full upper-triangle profile
    # from a zero bmat (column_index pattern is value-independent).
    dummy_bmat = zeros(T, problem.m, problem.m)
    sparse_bmat, diagonalIndex = _profile_from_dense(dummy_bmat, ordering)
    ns.sparse_bmat = sparse_bmat
    ns.diagonalIndex = diagonalIndex

    # Build the location table.
    SDPA.make_aggrigate_index_sdp!(ns, problem, ordering)

    # 5 parallel arrays have equal length.
    n = length(ns.SDP_constraint1)
    @test n == length(ns.SDP_constraint2)
    @test n == length(ns.blockIndex1)
    @test n == length(ns.blockIndex2)
    @test n == length(ns.SDP_location_sparse_bMat)
    @test n > 0  # example1 has at least one SDP block with nonzeros

    # Per-block sum matches.
    @test sum(ns.SDP_block_table_len) == n

    # Every location index is in [1, nnz].
    nnz = ns.diagonalIndex[end] - 1
    @test all(1 .<= ns.SDP_location_sparse_bMat .<= nnz)

    # diagonalIndex has m+1 entries.
    @test length(ns.diagonalIndex) == problem.m + 1

    # Non-identity ordering in use (ordering != [1,2,3]).
    @test ns.ordering != collect(1:problem.m)

    # Location-index formula matches by-hand computation.
    # For each (i, j) recorded, the table position should equal
    # diagonalIndex[ii] + (jj - ii), where (ii, jj) = minmax(reverse_ordering).
    for k in 1:n
        i = ns.SDP_constraint1[k]
        j = ns.SDP_constraint2[k]
        p_i = ns.reverse_ordering[i]
        p_j = ns.reverse_ordering[j]
        ii, jj = p_i <= p_j ? (p_i, p_j) : (p_j, p_i)
        expected = ns.diagonalIndex[ii] + (jj - ii)
        @test ns.SDP_location_sparse_bMat[k] == expected
    end
end

# ---------------------------------------------------------------------------
# Sparse bmat numerical equivalence + NaN poisoning.
# ---------------------------------------------------------------------------
# Solve example1 via dense and sparse paths; assert same phase + close
# objective value at Float64 AND BigFloat(256).
# NaN poisoning — when use_sparse=true, compute_bmat_sparse_sdp!
# must NOT touch newton_state.bmat.de_ele. Poison with NaN before
# the call; assert NaN survives.

@testset "sparse vs dense bmat path agree on phase and objective" begin
    for T in (Float64, BigFloat)
        # Dense path.
        problem_d = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T)
        params_d = SDPA.Parameters{T}()
        log_d = SDPA.solve!(problem_d, params_d; verbose=false, io=devnull,
                             force_sparse=false)
        # Sparse path.
        problem_s = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T)
        params_s = SDPA.Parameters{T}()
        log_s = SDPA.solve!(problem_s, params_s; verbose=false, io=devnull,
                             force_sparse=true)

        @test log_s.terminal_phase == log_d.terminal_phase
        # Sparse vs dense: F1/F2/F3 formulas have slightly different
        # floating-point operation order, so allow rtol=1e-8 at Float64
        # and rtol=1e-30 at BigFloat(256).
        rtol = T === Float64 ? 1e-8 : BigFloat("1e-30")
        @test isapprox(log_s.final_obj_primal, log_d.final_obj_primal; rtol=rtol)
        @test isapprox(log_s.final_obj_dual,   log_d.final_obj_dual;   rtol=rtol)
    end
end

@testset "compute_bmat_sparse_sdp! never writes dense bmat (NaN poisoning)" begin
    # When use_sparse=true, compute_bmat_sparse_sdp! must NEVER touch
    # bmat.de_ele. Poison with NaN before the call; assert NaN survives.
    T = Float64
    problem = SDPA.read_dat_s(
        joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T)
    SDPA.initialize_index!(problem)
    ns = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
    SDPA.compute_formula!(ns, problem)
    ordering = [2, 3, 1]
    ns.use_sparse = true
    ns.ordering = ordering
    ns.reverse_ordering = Vector{Int}(undef, problem.m)
    for (pos, orig) in enumerate(ordering)
        ns.reverse_ordering[orig] = pos
    end
    # Build sparse_bmat structure (populates column_index + diagonalIndex).
    dummy_bmat = zeros(T, problem.m, problem.m)
    sparse_bmat, diagonalIndex = _profile_from_dense(dummy_bmat, ordering)
    ns.sparse_bmat = sparse_bmat
    ns.diagonalIndex = diagonalIndex
    SDPA.make_aggrigate_index_sdp!(ns, problem, ordering)

    # Build the current_pt + work needed for compute_bmat_sparse_sdp!.
    params = SDPA.Parameters{T}()
    current_pt = SDPA.SDPASolution{T}(
        problem.m, problem.blockStruct, params.lambdaStar)
    work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)

    # Poison bmat.de_ele with NaN.
    fill!(ns.bmat.de_ele, T(NaN))
    # Snapshot to detect any later write.
    poisoned_copy = copy(ns.bmat.de_ele)

    # Run sparse assembly.
    SDPA.compute_bmat_sparse_sdp!(ns, problem, current_pt, work)

    # Assertion 1: bmat.de_ele still equals the poisoned copy (NaN survives).
    # Use isequal for NaN-aware byte-identical comparison.
    @test isequal(ns.bmat.de_ele, poisoned_copy)
    # Assertion 2: sparse_bmat.sp_ele has no NaN (was actually populated).
    @test !any(isnan, ns.sparse_bmat.sp_ele)
    # Assertion 3: sparse_bmat.sp_ele has at least one nonzero value
    # (the assembly did something).
    @test any(x -> x != zero(T), ns.sparse_bmat.sp_ele)
end

# ---------------------------------------------------------------------------
# Symbolic factorization cached across iterations.
# ---------------------------------------------------------------------------
# diagonalIndex is NOT reallocated inside make_bmat!.
# sparse_bmat row_index/column_index object identity is preserved
# across iterations (proves the symbolic pattern is allocated once).

@testset "sparse symbolic pattern identity preserved across make_bmat! calls" begin
    T = Float64
    problem = SDPA.read_dat_s(
        joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T)
    params = SDPA.Parameters{T}(maxIteration=5)  # short run
    log = SDPA.solve!(problem, params; verbose=false, io=devnull,
                       force_sparse=true)

    # After the solve, sparse_bmat should still exist (set up before
    # the loop). The fact that the solver completed iterations proves
    # the cache worked — if sparse_bmat were reallocated each iteration,
    # the location table would be invalidated and the solver would
    # crash in compute_bmat_sparse_sdp! (out-of-bounds sp_ele writes).
    @test log.iterations > 1
    # Phase need not be pdOPT (maxIteration=5 may be too small); the
    # structural check below is the real test of caching.

    # Stronger structural test: re-run on a fresh problem and verify
    # diagonalIndex identity is preserved. We do this by checking that
    # the sparse_bmat structure (column_index) is consistent across
    # multiple make_bmat! invocations.
    problem2 = SDPA.read_dat_s(
        joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T)
    SDPA.initialize_index!(problem2)
    ns = SDPA.NewtonState{T}(problem2.m, problem2.blockStruct)
    SDPA.compute_formula!(ns, problem2)
    ordering = collect(1:problem2.m)  # identity (force_sparse path)
    ns.use_sparse = true
    ns.ordering = ordering
    ns.reverse_ordering = collect(1:problem2.m)
    dummy = zeros(T, problem2.m, problem2.m)
    sb, di = _profile_from_dense(dummy, ordering)
    ns.sparse_bmat = sb
    ns.diagonalIndex = di
    SDPA.make_aggrigate_index_sdp!(ns, problem2, ordering)

    # Capture object identities (proves no reallocation).
    sparse_bmat_id = objectid(ns.sparse_bmat)
    column_index_id = objectid(ns.sparse_bmat.column_index)
    row_index_id = objectid(ns.sparse_bmat.row_index)
    diagonalIndex_id = objectid(ns.diagonalIndex)

    # Now run two make_bmat! calls (simulating 2 iterations) and
    # verify the structure arrays are the same objects.
    current_pt = SDPA.SDPASolution{T}(
        problem2.m, problem2.blockStruct, params.lambdaStar)
    work = SDPA.SDPAWorkVariables{T}(problem2.blockStruct)
    SDPA.make_bmat!(ns, problem2, current_pt, work)
    @test objectid(ns.sparse_bmat) === sparse_bmat_id
    @test objectid(ns.sparse_bmat.column_index) === column_index_id
    @test objectid(ns.sparse_bmat.row_index) === row_index_id
    @test objectid(ns.diagonalIndex) === diagonalIndex_id

    SDPA.make_bmat!(ns, problem2, current_pt, work)
    @test objectid(ns.sparse_bmat) === sparse_bmat_id
    @test objectid(ns.sparse_bmat.column_index) === column_index_id
    @test objectid(ns.diagonalIndex) === diagonalIndex_id
end
