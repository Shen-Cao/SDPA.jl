# Build a tiny 2-block problem for testing SDPAProblem's accessors.
# Problem: m=2 constraints, nBlock=2, blockStruct=[2, -1]
#   SDP block 1: 2x2 symmetric
#   LP block 2:  1 scalar
#
# Constraint 1: A_1 = [[1,0],[0,0]] in SDP block 1, 0 in LP
# Constraint 2: A_2 = [[0,0],[0,2]] in SDP block 1, 3 in LP block 2
# Objective:    C  = [[5,0],[0,7]] in SDP block 1, 11 in LP block 2
# RHS:          b  = [1, 4]
#
# A hand-checkable instance: every Frobenius inner product has a clean
# closed form.

function _build_test_problem(::Type{T}) where {T<:Real}
    m = 2
    blockStruct = [2, -1]

    b = SDPA.SDPAVector{T}([T(1), T(4)])

    # C: SDP block at problem-index 1, LP block at problem-index 2.
    # blockStruct[1]=2 -> SDP; blockStruct[2]=-1 -> LP.
    C = SDPA.SDPASparseLinearSpace{T}(1, 1)
    C.SDP_sp_index = [1]
    C.LP_sp_index = [1]
    C.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 2)
    SDPA.setelement_sdp!(C, 1, 1, 1, T(5))
    SDPA.setelement_sdp!(C, 1, 2, 2, T(7))
    SDPA.setelement_lp!(C, 1, T(11))

    A1 = SDPA.SDPASparseLinearSpace{T}(1, 1)
    A1.SDP_sp_index = [1]
    A1.LP_sp_index = [1]
    A1.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
    SDPA.setelement_sdp!(A1, 1, 1, 1, T(1))
    SDPA.setelement_lp!(A1, 1, T(0))

    A2 = SDPA.SDPASparseLinearSpace{T}(1, 1)
    A2.SDP_sp_index = [1]
    A2.LP_sp_index = [1]
    A2.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
    SDPA.setelement_sdp!(A2, 1, 2, 2, T(2))
    SDPA.setelement_lp!(A2, 1, T(3))

    problem = SDPA.SDPAProblem{T}(m, 2, blockStruct, b, C, SDPA.SDPASparseLinearSpace{T}[A1, A2])
    SDPA.initialize_index!(problem)
    return problem
end

@testset "SDPAProblem" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            problem = _build_test_problem(T)
            @test problem.m == 2
            @test problem.nBlock == 2
            @test problem.blockStruct == [2, -1]
            @test length(problem.A) == 2

            # initialize_index! built correct per-block tables.
            @test problem.SDP_nConstraint == [2, 0]  # both constraints touch SDP block 1
            @test problem.SDP_constraint[1] == [1, 2]
            @test problem.SDP_blockIndex[1] == [1, 1]
            @test problem.LP_nConstraint == [1]  # only A_2 has a nonzero LP scalar (A_1's stored 0 is skipped)
            @test problem.LP_constraint[1] == [2]

            # multi_inner_product_to_a! with a known xMat.
            # xMat.SDP_block[1] = [[10, 0], [0, 20]] (2x2)
            # xMat.LP_block[1]  = 100
            xMat = SDPA.SDPADenseLinearSpace{T}([2, -1])
            xMat.SDP_block[1][1, 1] = T(10)
            xMat.SDP_block[1][2, 2] = T(20)
            xMat.LP_block[1] = T(100)

            ret = SDPA.SDPAVector{T}(2)
            SDPA.multi_inner_product_to_a!(ret, problem, xMat)
            # ret[1] = <A_1, xMat> = 1*10 + 0*20 + 0*100 = 10
            # ret[2] = <A_2, xMat> = 0*10 + 2*20 + 3*100 = 340
            @test ret[1] == T(10)
            @test ret[2] == T(340)

            # multi_plus_to_a! with a known xVec.
            # xVec = [1, 0.5]
            # Expected retMat.SDP_block[1] = 1*A1 + 0.5*A2 = [[1, 0], [0, 1]]
            # Expected retMat.LP_block[1]  = 1*0 + 0.5*3 = 1.5
            xVec = SDPA.SDPAVector{T}([T(1), T(1//2)])
            retMat = SDPA.SDPADenseLinearSpace{T}([2, -1])
            SDPA.multi_plus_to_a!(retMat, problem, xVec)
            @test retMat.SDP_block[1][1, 1] == T(1)
            @test retMat.SDP_block[1][2, 2] == T(1)
            @test retMat.SDP_block[1][1, 2] == T(0)
            @test retMat.LP_block[1] == T(3//2)
        end
    end

    # The dense-A_i accumulation branch must validate block sizes like
    # its sparse sibling multi_inner_product_to_a! (which throws
    # DimensionMismatch) — an unchecked @inbounds loop on a mismatched
    # dense block would read out-of-bounds memory. Exercised through
    # the public internal API SDPA.multi_plus_to_a! with a hand-built
    # problem whose A_1 dense block (3x3) disagrees with blockStruct ([2]
    # -> retMat block is 2x2).
    @testset "multi_plus_to_a! dense-block size mismatch" begin
        for T in (Float64, BigFloat)
            m = 1
            blockStruct = [2]
            b = SDPA.SDPAVector{T}([T(1)])
            C = SDPA.SDPASparseLinearSpace{T}(1, 0)
            C.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
            SDPA.setelement_sdp!(C, 1, 1, 1, T(1))

            A1 = SDPA.SDPASparseLinearSpace{T}(1, 0)
            A1.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(3, 3, :dense)  # wrong size vs [2]
            A1.SDP_sp_block[1].de_ele[1, 1] = T(1)

            problem = SDPA.SDPAProblem{T}(m, 1, blockStruct, b, C,
                                          SDPA.SDPASparseLinearSpace{T}[A1])
            SDPA.initialize_index!(problem)

            xVec = SDPA.SDPAVector{T}([T(1)])
            retMat = SDPA.SDPADenseLinearSpace{T}(blockStruct)
            err = try
                SDPA.multi_plus_to_a!(retMat, problem, xVec)
                nothing
            catch e
                e
            end
            @test err isa DimensionMismatch
            @test occursin("A_i block 1", err.msg)
            @test occursin("(3, 3)", err.msg)
            @test occursin("(2, 2)", err.msg)

            # Same problem with a correctly-sized dense block still works
            # (happy path through the new check).
            A1_ok = SDPA.SDPASparseLinearSpace{T}(1, 0)
            A1_ok.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :dense)
            A1_ok.SDP_sp_block[1].de_ele[1, 1] = T(3)
            A1_ok.SDP_sp_block[1].de_ele[2, 2] = T(5)
            problem_ok = SDPA.SDPAProblem{T}(m, 1, blockStruct, b, C,
                                             SDPA.SDPASparseLinearSpace{T}[A1_ok])
            SDPA.initialize_index!(problem_ok)
            retMat_ok = SDPA.SDPADenseLinearSpace{T}(blockStruct)
            SDPA.multi_plus_to_a!(retMat_ok, problem_ok, xVec)
            @test retMat_ok.SDP_block[1][1, 1] == T(3)
            @test retMat_ok.SDP_block[1][2, 2] == T(5)
        end
    end
end

# --- LP index tables: global-scalar-index contract (C++ parity) ---

function _build_lp_index_problem(::Type{T}) where {T<:Real}
    # blockStruct = [2, -3]: one 2×2 SDP block + 3 LP scalars.
    # A_1 touches LP scalar 3 only; A_2 touches LP scalars 2 and 3.
    m = 2
    blockStruct = [2, -3]
    b = SDPA.SDPAVector{T}([T(1), T(4)])
    C = SDPA.SDPASparseLinearSpace{T}(1, 3)
    C.SDP_sp_index = [1]
    C.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
    SDPA.setelement_sdp!(C, 1, 1, 1, T(5))
    SDPA.setelement_lp!(C, 1, T(11))

    A1 = SDPA.SDPASparseLinearSpace{T}(1, 3)
    A1.SDP_sp_index = [1]
    A1.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
    SDPA.setelement_sdp!(A1, 1, 1, 1, T(1))
    SDPA.setelement_lp!(A1, 3, T(7))

    A2 = SDPA.SDPASparseLinearSpace{T}(1, 3)
    A2.SDP_sp_index = [1]
    A2.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
    SDPA.setelement_sdp!(A2, 1, 2, 2, T(2))
    SDPA.setelement_lp!(A2, 2, T(5))
    SDPA.setelement_lp!(A2, 3, T(9))

    problem = SDPA.SDPAProblem{T}(m, 2, blockStruct, b, C,
                                  SDPA.SDPASparseLinearSpace{T}[A1, A2])
    SDPA.initialize_index!(problem)
    return problem
end

@testset "LP index tables store the global LP scalar index" begin
    # Contract lock: LP_blockIndex[l][k] is the problem-level LP SCALAR
    # index (== the table's own outer index l; C++ initialize_index_LP
    # parity). It is NOT a compact A_i-internal position — the supported
    # consumption pattern is get_lp_value(A[i], idx), never indexing
    # A[i].LP_sp_val with it.
    for T in (Float64, BigFloat)
        problem = _build_lp_index_problem(T)
        @test problem.LP_nConstraint == [0, 1, 2]
        @test problem.LP_constraint == [Int[], [2], [1, 2]]
        # Global-index semantics (compact positions would give [[], [1], [1, 2]]).
        @test problem.LP_blockIndex == [Int[], [2], [3, 3]]
        # The supported consumption pattern works for every table entry.
        for l in 2:3, k in eachindex(problem.LP_constraint[l])
            i = problem.LP_constraint[l][k]
            @test SDPA.get_lp_value(problem.A[i], problem.LP_blockIndex[l][k]) ==
                  SDPA.get_lp_value(problem.A[i], l)
        end
        @test SDPA.get_lp_value(problem.A[1], 3) == T(7)
        @test SDPA.get_lp_value(problem.A[2], 2) == T(5)
        @test SDPA.get_lp_value(problem.A[2], 3) == T(9)
    end
end
