@testset "core types" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            # SDPAVector
            v = SDPA.SDPAVector{T}(5)
            @test length(v) == 5
            @test size(v) == (5,)
            @test eltype(v) === T
            @test all(iszero, v.ele)
            v[1] = T(7)
            @test v[1] == T(7)
            v2 = SDPA.SDPAVector{T}(5)
            copyto!(v2, v)
            @test v2[1] == T(7)
            @test v2 == v
            # == must be backed by consistent isequal/hash so that
            # value-equal vectors work as Dict keys / Set elements.
            v3 = SDPA.SDPAVector{T}(5)
            copyto!(v3, v)
            @test isequal(v, v3)
            @test hash(v) == hash(v3)
            @test get(Dict(v => 1), v3, :missing) === 1
            SDPA.setzero!(v2)
            @test all(iszero, v2.ele)
            SDPA.setidentity!(v2, T(3))
            @test all(x -> x == T(3), v2.ele)
            @test length(SDPA.SDPAVector{T}(0)) == 0

            # SDPABlockVector
            bs = [2, -3, 1]
            bv = SDPA.SDPABlockVector{T}(3, bs)
            @test bv.nBlock == 3
            @test bv.blockStruct == bs
            # SDP block (bs=2): 2*3/2 = 3 entries
            @test length(bv.ele[1]) == 3
            # LP block (bs=-3): 3 entries
            @test length(bv.ele[2]) == 3
            # LP block (bs=1): 1 entry
            @test length(bv.ele[3]) == 1
            bv.ele[1][1] = T(11)
            bv_copy = copy(bv)
            @test bv_copy.ele[1][1] == T(11)
            SDPA.setzero!(bv_copy)
            @test all(iszero, bv_copy.ele[1].ele)

            # SDPADenseMatrix
            m = SDPA.SDPADenseMatrix{T}(3, 4)
            @test size(m) == (3, 4)
            @test m.type === :dense
            @test all(iszero, m.de_ele)
            m[1, 2] = T(5)
            @test m[1, 2] == T(5)
            m_id = SDPA.SDPADenseMatrix{T}(3, 3)
            SDPA.setidentity!(m_id, T(2))
            @test m_id[1, 1] == T(2)
            @test m_id[2, 1] == T(0)
            @test m_id[3, 3] == T(2)
            m_completion = SDPA.SDPADenseMatrix{T}(2, 2, :completion)
            @test m_completion.type === :completion

            # SDPASparseMatrix
            sm = SDPA.SDPASparseMatrix{T}(4, 4, :sparse, 4)
            @test sm.type === :sparse
            @test sm.nonZeroNumber == 4
            @test sm.nonZeroEffect == 0
            SDPA.setidentity!(sm, T(1))
            @test sm.nonZeroEffect == 4
            # After identity: 4 diagonal entries, all = 1

            # change_to_dense! force path
            sm_force = SDPA.SDPASparseMatrix{T}(4, 4, :sparse, 4)
            SDPA.setidentity!(sm_force)
            SDPA.change_to_dense!(sm_force; force=true)
            @test sm_force.type === :dense
            expected = Matrix{T}(I, 4, 4)
            @test sm_force.de_ele == expected

            # change_to_dense! density gate: 4/16 = 0.25 > 0.20 should auto-trigger
            sm_gate = SDPA.SDPASparseMatrix{T}(4, 4, :sparse, 4)
            SDPA.setidentity!(sm_gate)  # 4 nonzeros out of 16 = density 0.25
            SDPA.change_to_dense!(sm_gate)  # no force
            @test sm_gate.type === :dense

            # change_to_dense! density gate: 1/100 = 0.01 should NOT auto-trigger
            sm_low = SDPA.SDPASparseMatrix{T}(10, 10, :sparse, 1)
            sm_low.row_index[1] = 1
            sm_low.column_index[1] = 1
            sm_low.sp_ele[1] = T(7)
            sm_low.nonZeroEffect = 1
            SDPA.change_to_dense!(sm_low)  # no force; density 0.01
            @test sm_low.type === :sparse

            # copyto! with a capacity-grown src. src's arrays were
            # resize!-doubled to capacity 8 by setelement_sdp! (5 stored
            # entries) while dst was constructed with capacity 2 — the old
            # code threw a raw BoundsError (dst arrays shorter than src's),
            # or on BigFloat died reading src's #undef capacity tail.
            let
                dst = SDPA.SDPASparseMatrix{T}(3, 3, :sparse, 2)
                Ssrc = SDPA.SDPASparseLinearSpace{T}(1, 0)
                Ssrc.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(3, 3, :sparse, 0)
                for (k, ij) in enumerate(((1, 1), (2, 2), (3, 3), (1, 2), (1, 3)))
                    SDPA.setelement_sdp!(Ssrc, 1, ij[1], ij[2], T(k))
                end
                src = Ssrc.SDP_sp_block[1]
                @test src.nonZeroCount == 8      # 0 -> max(8, 0) on first write
                @test src.nonZeroEffect == 5
                @test length(dst.row_index) == 2
                copyto!(dst, src)
                @test dst.nonZeroNumber == src.nonZeroNumber
                @test dst.nonZeroCount == src.nonZeroCount == 8
                @test dst.nonZeroEffect == 5
                @test dst.row_index == src.row_index
                @test dst.column_index == src.column_index
                @test dst.sp_ele[1:5] == src.sp_ele[1:5]
                @test dst.sp_ele[1:5] == T.(1:5)
                # Total copy: no #undef slots survive in dst (BigFloat
                # resize! growth leaves the tail undefined).
                @test all(isassigned(dst.sp_ele, k) for k in 1:length(dst.sp_ele))
                # A subsequent append lands in the copied capacity — no
                # BoundsError, values in the right slots.
                Sdst = SDPA.SDPASparseLinearSpace{T}(1, 0)
                Sdst.SDP_sp_block[1] = dst
                SDPA.setelement_sdp!(Sdst, 1, 3, 2, T(99))  # 6th entry, cap 8
                @test dst.nonZeroEffect == 6
                @test dst.row_index[6] == 3
                @test dst.column_index[6] == 2
                @test dst.sp_ele[6] == T(99)
            end
        end
    end
end
