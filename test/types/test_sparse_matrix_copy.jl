@testset "SDPASparseMatrix representation-aware copies" begin
    for T in (Float64, BigFloat)
        src = SDPA.SDPASparseMatrix{T}(3, 3, :sparse, 12)
        SDPA.setidentity!(src, T(3))
        @test length(src.row_index) == length(src.column_index) ==
              length(src.sp_ele) == src.nonZeroCount == 3
        @test copyto!(src, src) === src
        @test src.sp_ele == fill(T(3), 3)
        dst = SDPA.SDPASparseMatrix{T}(3, 3, :sparse, 11)
        @test copyto!(dst, src) === dst
        @test length(dst.sp_ele) == dst.nonZeroCount == dst.nonZeroEffect == 3
        @test dst.sp_ele == src.sp_ele
        space = SDPA.SDPASparseLinearSpace{T}(1, 0)
        space.SDP_sp_block[1] = dst
        SDPA.setelement_sdp!(space, 1, 1, 2, T(2))
        @test dst.nonZeroNumber == 3
        @test dst.nonZeroCount == 8 && dst.nonZeroEffect == 4
        grown = copy(dst)
        @test grown.nonZeroNumber == 3 && grown.nonZeroCount == 8
        @test grown.sp_ele[1:4] == T[3, 3, 3, 2]
        @test all(iszero, grown.sp_ele[5:end])

        SDPA.change_to_dense!(dst; force=true)
        SDPA.change_to_dense!(src; force=true)
        @test src.nonZeroEffect == dst.nonZeroEffect == 9
        @test copyto!(src, dst) === src
        fresh = copy(dst)
        @test src.de_ele == fresh.de_ele == T[3 2 0; 2 3 0; 0 0 3]
        @test src.nonZeroCount == fresh.nonZeroCount == 9
        @test isempty(src.row_index) && isempty(src.column_index) && isempty(src.sp_ele)
        @test isempty(fresh.row_index) && isempty(fresh.column_index) && isempty(fresh.sp_ele)
        if T === BigFloat
            SDPA.mp_add!(src.de_ele[1, 2], src.de_ele[1, 2], one(T))
            @test src.de_ele[1, 2] == 3
            @test src.de_ele[2, 1] == dst.de_ele[1, 2] == fresh.de_ele[1, 2] == 2
        end
    end
end

@testset "BigFloat matrix copies retain their destination precision convention" begin
    high = setprecision(BigFloat, 256) do
        m = SDPA.SDPASparseMatrix{BigFloat}(2, 2, :sparse)
        SDPA.setidentity!(m, 1 + BigFloat(2)^(-100))
        m
    end
    dense = setprecision(BigFloat, 256) do
        SDPA.change_to_dense!(copy(high); force=true)
    end
    dense_dst, sparse_dst = setprecision(BigFloat, 128) do
        SDPA.SDPASparseMatrix{BigFloat}(2, 2, :dense),
        SDPA.SDPASparseMatrix{BigFloat}(2, 2, :sparse, 2)
    end
    setprecision(BigFloat, 64) do
        copyto!(dense_dst, dense)
        @test precision(dense_dst.de_ele[1, 1]) == 128
        @test dense_dst.de_ele[1, 1] == dense.de_ele[1, 1] > 1
        low = copy(dense)
        @test precision(low.de_ele[1, 1]) == 64
        @test low.de_ele[1, 1] == 1
        copyto!(sparse_dst, high)
        @test precision(sparse_dst.sp_ele[1]) == 64
        @test sparse_dst.sp_ele[1] == 1 < high.sp_ele[1]
    end
end
