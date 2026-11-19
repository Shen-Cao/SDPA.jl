@testset "linear space types" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            # SparseLinearSpace: 1 SDP block, 2 LP blocks
            S = SDPA.SDPASparseLinearSpace{T}(1, 2)
            @test S.SDP_sp_nBlock == 1
            @test S.SOCP_sp_nBlock == 0
            @test S.LP_sp_nBlock == 2
            @test isempty(S.SOCP_sp_block)
            # Compact LP storage: starts empty (no nonzeros).
            @test S.LP_nnz == 0
            @test isempty(S.LP_sp_index)
            @test isempty(S.LP_sp_val)

            # setElement_SDP - the block must be pre-sized (every parser-flow
            # caller does): give it a known shape (3x3) and small capacity.
            S.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(3, 3, :sparse, 4)
            SDPA.setelement_sdp!(S, 1, 1, 1, T(10))
            SDPA.setelement_sdp!(S, 1, 2, 2, T(20))
            @test S.SDP_sp_block[1].nonZeroEffect == 2
            @test S.SDP_sp_block[1].sp_ele[1] == T(10)

            SDPA.setelement_lp!(S, 2, T(99))
            @test S.LP_nnz == 1
            @test S.LP_sp_index == [2]
            @test SDPA.get_lp_value(S, 2) == T(99)
            @test SDPA.get_lp_value(S, 1) == T(0)   # absent -> 0

            # setzero! / setidentity! paths
            SDPA.setzero!(S)
            @test S.LP_nnz == 0                     # all-zero == empty compact
            @test all(SDPA.get_lp_value(S, k) == T(0) for k in 1:2)
            @test S.SDP_sp_block[1].nonZeroEffect == 0
            SDPA.setidentity!(S, T(2))
            @test S.LP_nnz == 2                     # every scalar stored
            @test all(SDPA.get_lp_value(S, k) == T(2) for k in 1:2)

            # DenseLinearSpace from blockStruct
            blockStruct = [2, -3, 4]
            D = SDPA.SDPADenseLinearSpace{T}(blockStruct)
            @test D.SDP_nBlock == 2  # [2, 4]
            @test D.LP_nBlock == 3   # [-3] = 3 separate LP scalars
            @test size(D.SDP_block[1]) == (2, 2)
            @test size(D.SDP_block[2]) == (4, 4)
            @test length(D.LP_block) == 3

            # setelement_sdp! / setelement_lp! round-trip
            SDPA.setelement_sdp!(D, 1, 1, 1, T(7))
            SDPA.setelement_sdp!(D, 2, 3, 4, T(8))
            SDPA.setelement_lp!(D, 1, T(42))
            @test D.SDP_block[1][1, 1] == T(7)
            @test D.SDP_block[2][3, 4] == T(8)
            @test D.LP_block[1] == T(42)

            # setidentity! fills diagonals
            SDPA.setidentity!(D, T(3))
            @test D.SDP_block[1][1, 1] == T(3)
            @test D.SDP_block[1][1, 2] == T(0)
            @test D.SDP_block[2][4, 4] == T(3)
            @test D.LP_block[1] == T(3)

            # copy round-trip
            D2 = SDPA.SDPADenseLinearSpace{T}([2, -3, 4])
            copyto!(D2, D)
            @test D2.SDP_block[1][1, 1] == T(3)
            @test D2.SDP_block[2][4, 4] == T(3)

            # The 2-arg constructor leaves SDP_block unassigned; the
            # first setelement_sdp! write must error with context naming
            # the block, not die on a raw UndefRefError.
            D4 = SDPA.SDPADenseLinearSpace{T}(1, 0)
            @test !isassigned(D4.SDP_block, 1)
            err = try
                SDPA.setelement_sdp!(D4, 1, 1, 1, T(5))
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("never sized", err.msg)
            @test occursin("block 1", err.msg)
            # A properly-constructed space (blockStruct ctor sizes the
            # blocks) still accepts writes — the guard costs one isassigned.
            D5 = SDPA.SDPADenseLinearSpace{T}([2])
            SDPA.setelement_sdp!(D5, 1, 1, 1, T(5))
            @test D5.SDP_block[1][1, 1] == T(5)

            # LP block vectors must use independent BigFloat elements
            # (distinct mpfr_t per slot) and value-copy semantics. Float64 is
            # isbits so `!==` can't distinguish objects; assert value behavior.
            @testset "LP block independence + value-copy" begin
                # Constructor: each LP slot is its own BigFloat object.
                D3 = SDPA.SDPADenseLinearSpace{T}([2, -3, 4])
                if T <: BigFloat
                    @test D3.LP_block[1] !== D3.LP_block[2]
                    @test D3.LP_block[1] !== D3.LP_block[3]
                end
                SDPA.setelement_lp!(D3, 1, T(5))
                @test D3.LP_block[1] == T(5)
                @test D3.LP_block[2] == T(0)
                @test D3.LP_block[3] == T(0)

                # setzero! / setidentity! keep elements independent.
                SDPA.setzero!(D3)
                @test all(iszero, D3.LP_block)
                if T <: BigFloat
                    @test D3.LP_block[1] !== D3.LP_block[2]
                end
                SDPA.setidentity!(D3, T(2))
                @test all(==(T(2)), D3.LP_block)
                if T <: BigFloat
                    @test D3.LP_block[1] !== D3.LP_block[2]
                    SDPA.mp_add!(D3.LP_block[1], D3.LP_block[1], BigFloat(1))
                    @test D3.LP_block[1] == BigFloat(3)
                    @test D3.LP_block[2] == BigFloat(2)
                end

                # copyto!: value copy, not reference copy (mutation-isolated).
                srcD = SDPA.SDPADenseLinearSpace{T}([2, -3, 4])
                SDPA.setidentity!(srcD, T(7))
                dstD = SDPA.SDPADenseLinearSpace{T}([2, -3, 4])
                copyto!(dstD, srcD)
                if T <: BigFloat
                    @test dstD.LP_block[1] !== srcD.LP_block[1]
                end
                @test dstD.LP_block == srcD.LP_block
                if T <: BigFloat
                    SDPA.mp_add!(dstD.LP_block[1], dstD.LP_block[1], BigFloat(1))
                    @test dstD.LP_block[1] == BigFloat(8)
                    @test srcD.LP_block[1] == BigFloat(7)
                else
                    dstD.LP_block[1] += T(1)
                    @test dstD.LP_block[1] == T(8)
                    @test srcD.LP_block[1] == T(7)
                end

                # Base.copy: independent copy.
                cD = copy(srcD)
                if T <: BigFloat
                    @test cD.LP_block[1] !== srcD.LP_block[1]
                end
                @test cD.LP_block == srcD.LP_block
                if T <: BigFloat
                    SDPA.mp_add!(cD.LP_block[1], cD.LP_block[1], BigFloat(1))
                    @test cD.LP_block[1] == BigFloat(8)
                    @test srcD.LP_block[1] == BigFloat(7)
                else
                    cD.LP_block[1] += T(1)
                    @test cD.LP_block[1] == T(8)
                    @test srcD.LP_block[1] == T(7)
                end

                # SparseLinearSpace compact LP storage: same guarantees.
                S3 = SDPA.SDPASparseLinearSpace{T}(0, 3)
                SDPA.setelement_lp!(S3, 1, T(5))
                @test S3.LP_nnz == 1
                @test S3.LP_sp_index == [1]
                @test SDPA.get_lp_value(S3, 1) == T(5)
                @test SDPA.get_lp_value(S3, 2) == T(0)
                SDPA.setzero!(S3)
                @test S3.LP_nnz == 0
                @test all(SDPA.get_lp_value(S3, k) == T(0) for k in 1:3)
                SDPA.setidentity!(S3, T(2))
                @test S3.LP_nnz == 3
                @test all(SDPA.get_lp_value(S3, k) == T(2) for k in 1:3)
                if T <: BigFloat
                    @test S3.LP_sp_val[1] !== S3.LP_sp_val[2]
                    SDPA.mp_add!(S3.LP_sp_val[1], S3.LP_sp_val[1], BigFloat(1))
                    @test S3.LP_sp_val[1] == BigFloat(3)
                    @test S3.LP_sp_val[2] == BigFloat(2)
                end

                # Sparse copyto!: value copy, not reference copy.
                srcS = SDPA.SDPASparseLinearSpace{T}(0, 3)
                SDPA.setidentity!(srcS, T(7))
                dstS = SDPA.SDPASparseLinearSpace{T}(0, 3)
                copyto!(dstS, srcS)
                if T <: BigFloat
                    @test dstS.LP_sp_val[1] !== srcS.LP_sp_val[1]
                end
                @test dstS.LP_sp_index == srcS.LP_sp_index
                @test dstS.LP_sp_val == srcS.LP_sp_val
                if T <: BigFloat
                    SDPA.mp_add!(dstS.LP_sp_val[1], dstS.LP_sp_val[1], BigFloat(1))
                    @test dstS.LP_sp_val[1] == BigFloat(8)
                    @test srcS.LP_sp_val[1] == BigFloat(7)
                else
                    dstS.LP_sp_val[1] += T(1)
                    @test dstS.LP_sp_val[1] == T(8)
                    @test srcS.LP_sp_val[1] == T(7)
                end
            end

            # change_to_dense! on SparseLinearSpace at high density
            # Use a 4x4 sparse matrix with 4 entries (density 0.25 > 0.20).
            S2 = SDPA.SDPASparseLinearSpace{T}(1, 0)
            S2.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(4, 4, :sparse, 4)
            SDPA.setidentity!(S2.SDP_sp_block[1])
            SDPA.change_to_dense!(S2)
            @test S2.SDP_sp_block[1].type === :dense

            # copy of a matrix whose nonZeroCount > nonZeroNumber (block was
            # allocated with 0 declared capacity and setelement_sdp! doubled
            # it): Base.copy must not throw a BoundsError on the index arrays.
            S0 = SDPA.SDPASparseLinearSpace{T}(1, 0)
            S0.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(3, 3, :sparse, 0)
            SDPA.setelement_sdp!(S0, 1, 1, 1, T(10))
            m0 = S0.SDP_sp_block[1]
            @test m0.nonZeroNumber == 0
            @test m0.nonZeroCount == 8
            @test m0.nonZeroEffect == 1
            c0 = copy(m0)
            @test c0.nonZeroCount == m0.nonZeroCount
            @test c0.nonZeroEffect == m0.nonZeroEffect
            nz = m0.nonZeroEffect
            @test c0.row_index[1:nz] == m0.row_index[1:nz]
            @test c0.column_index[1:nz] == m0.column_index[1:nz]
            @test c0.sp_ele[1:nz] == m0.sp_ele[1:nz]
            @test c0.sp_ele[1] == T(10)

            # Defense-in-depth bounds validation in setelement_sdp!
            # On an already-assigned block, out-of-range and
            # non-positive indices must throw; valid writes must still work.
            S = SDPA.SDPASparseLinearSpace{T}(1, 0)
            S.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(3, 3, :sparse, 4)
            @test_throws ErrorException SDPA.setelement_sdp!(S, 1, 4, 1, T(1))
            @test_throws ErrorException SDPA.setelement_sdp!(S, 1, 1, 4, T(1))
            @test_throws ErrorException SDPA.setelement_sdp!(S, 1, 0, 1, T(1))
            @test_throws ErrorException SDPA.setelement_sdp!(S, 1, 1, 0, T(1))
            @test_throws ErrorException SDPA.setelement_sdp!(S, 1, -1, 1, T(1))
            # Failed calls must not have mutated state.
            @test S.SDP_sp_block[1].nonZeroEffect == 0
            # Valid entries still work after the failed attempts.
            SDPA.setelement_sdp!(S, 1, 1, 1, T(10))
            SDPA.setelement_sdp!(S, 1, 3, 3, T(20))
            @test S.SDP_sp_block[1].nonZeroEffect == 2
            @test S.SDP_sp_block[1].sp_ele[1] == T(10)
            @test S.SDP_sp_block[1].sp_ele[2] == T(20)
            # An unassigned block must error loudly: every assigned block
            # is sized by its construction path, so bounds checks run
            # unconditionally and a block that never gained dimensions
            # fails at the write, not later inside change_to_dense!.
            S2 = SDPA.SDPASparseLinearSpace{T}(2, 0)
            @test !isassigned(S2.SDP_sp_block, 1)
            err = try
                SDPA.setelement_sdp!(S2, 1, 1, 1, T(1))
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("no dimensions", err.msg)
            @test occursin("block 1", err.msg)
            # A pre-assigned block in the same space still accepts writes.
            S2.SDP_sp_block[2] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
            SDPA.setelement_sdp!(S2, 2, 1, 1, T(3))
            @test S2.SDP_sp_block[2].nonZeroEffect == 1
            @test S2.SDP_sp_block[2].sp_ele[1] == T(3)

            # copyto! contract — a block unassigned in src must ZERO
            # the dst block, not leave stale contents from an earlier copy.
            @testset "copyto! zeroes dst blocks unassigned in src" begin
                # Sparse space: dst first copied from a fully-populated src.
                full = SDPA.SDPASparseLinearSpace{T}(2, 0)
                for k in 1:2
                    full.SDP_sp_block[k] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
                    SDPA.setelement_sdp!(full, k, 1, 1, T(10 + k))
                end
                dstS = SDPA.SDPASparseLinearSpace{T}(2, 0)
                copyto!(dstS, full)
                @test dstS.SDP_sp_block[1].nonZeroEffect == 1
                @test dstS.SDP_sp_block[2].nonZeroEffect == 1
                # Now copy from a partial src (block 2 never assigned).
                partial = SDPA.SDPASparseLinearSpace{T}(2, 0)
                partial.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
                SDPA.setelement_sdp!(partial, 1, 1, 1, T(3))
                copyto!(dstS, partial)
                @test dstS.SDP_sp_block[1].sp_ele[1] == T(3)   # assigned src copied
                @test dstS.SDP_sp_block[2].nonZeroEffect == 0  # unassigned -> zeroed
                # Same contract when dst's block had gone dense: setzero!
                # fills de_ele with independent zeros, not stale values.
                dstS.SDP_sp_block[2] = SDPA.SDPASparseMatrix{T}(2, 2, :dense)
                dstS.SDP_sp_block[2].de_ele[1, 1] = T(9)
                copyto!(dstS, partial)
                @test all(iszero, dstS.SDP_sp_block[2].de_ele)

                # Dense space: same contract.
                fullD = SDPA.SDPADenseLinearSpace{T}([2, 2])
                SDPA.setelement_sdp!(fullD, 1, 1, 1, T(5))
                SDPA.setelement_sdp!(fullD, 2, 2, 2, T(7))
                dstD = SDPA.SDPADenseLinearSpace{T}(2, 0)
                copyto!(dstD, fullD)
                @test dstD.SDP_block[2][2, 2] == T(7)
                partialD = SDPA.SDPADenseLinearSpace{T}(2, 0)
                partialD.SDP_block[1] = SDPA.SDPADenseMatrix{T}(2, 2)
                partialD.SDP_block[1][1, 1] = T(3)
                copyto!(dstD, partialD)               # block 2 unassigned in src
                @test dstD.SDP_block[1][1, 1] == T(3)
                @test all(iszero, dstD.SDP_block[2].de_ele)  # zeroed, not stale 7
            end
        end
    end
end

@testset "lp_scalar_count is the single LP-scalar classification" begin
    # Every place that derives an LP scalar count from blockStruct must
    # agree: the shared helper, the dense-space constructor, and the
    # SDP-block classification. Covers the tricky entries (0, 1, -k, ≥2).
    for T in (Float64, BigFloat)
        for bs in ([2, -3], [2, 1, -1, 0], [0, 1, -5, 3], Int[],
                   [1, 1], [-2, 4], [2, -1], [-7])
            D = SDPA.SDPADenseLinearSpace{T}(collect(bs))
            @test SDPA.lp_scalar_count(collect(bs)) == D.LP_nBlock
            @test D.SDP_nBlock == count(b -> b >= 2, bs)
        end
    end
end
