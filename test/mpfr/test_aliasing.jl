# test/mpfr/test_aliasing.jl
# Aliasing stress test.
# Verifies that _fill_independent! creates truly independent BigFloat elements
# that can be mutated in-place via mp_fma! without corrupting other elements.

using Test
using SDPA

@testset "BigFloat aliasing stress test" begin

    @testset "_independent_zeros creates independent elements" begin
        setprecision(BigFloat, 200) do
            M = SDPA._independent_zeros(BigFloat, 5, 5)
            # All elements should be 0 initially
            @test all(iszero, M)
            # Elements are independent objects (not shared MPFR Memory)
            @test M[1,1] !== M[1,2]
            @test M[1,1] !== M[2,2]
            @test M[1,1] !== M[5,5]
            # Mutate [1,1] via mp_fma!
            SDPA.mp_fma!(M[1,1], BigFloat(2), BigFloat(3), M[1,1])
            @test M[1,1] == 6
            # Other elements must be unaffected
            @test M[2,2] == 0
            @test M[5,5] == 0
            @test M[1,2] == 0  # neighbor of mutated element unaffected
            @test count(!iszero, M) == 1
        end
    end

    @testset "_fill_independent! creates independent elements (Matrix)" begin
        setprecision(BigFloat, 200) do
            # Start with shared elements (from zeros)
            M = zeros(BigFloat, 4, 4)
            SDPA._fill_independent!(M, zero(BigFloat))
            # Each element is a distinct BigFloat object
            @test M[1,1] !== M[1,2]
            @test M[1,1] !== M[4,4]
            # Mutate multiple elements
            SDPA.mp_fma!(M[1,1], BigFloat(2), BigFloat(3), M[1,1])
            SDPA.mp_mul!(M[2,2], BigFloat(7), BigFloat(6))
            SDPA.mp_add!(M[3,3], BigFloat(1), BigFloat(1))
            # Check independence
            @test M[1,1] == 6
            @test M[2,2] == 42
            @test M[3,3] == 2
            @test M[4,4] == 0  # untouched
            @test M[1,2] == 0  # untouched
        end
    end

    @testset "_fill_independent! creates independent elements (Vector)" begin
        setprecision(BigFloat, 200) do
            V = zeros(BigFloat, 5)
            SDPA._fill_independent!(V, zero(BigFloat))
            # Each element is a distinct BigFloat object
            @test V[1] !== V[2]
            @test V[1] !== V[5]
            SDPA.mp_fma!(V[1], BigFloat(2), BigFloat(3), V[1])
            SDPA.mp_mul!(V[3], BigFloat(5), BigFloat(5))
            @test V[1] == 6
            @test V[2] == 0
            @test V[3] == 25
            @test V[4] == 0
            @test V[5] == 0
        end
    end

    @testset "_fill_independent! precision inside setprecision block" begin
        setprecision(BigFloat, 200) do
            M = zeros(BigFloat, 3, 3)
            SDPA._fill_independent!(M, zero(BigFloat))
            for i in eachindex(M)
                @test precision(M[i]) == 200
            end
        end
    end

    @testset "Float64 fill is unaffected" begin
        M = zeros(Float64, 3, 3)
        SDPA._fill_independent!(M, zero(Float64))
        M[1,1] = 42.0
        @test M[2,2] == 0.0
    end

    @testset "SDPADenseMatrix setzero! produces independent elements" begin
        setprecision(BigFloat, 200) do
            m = SDPA.SDPADenseMatrix{BigFloat}(3, 3)
            SDPA.setzero!(m)
            # Mutate via mp_fma!
            SDPA.mp_fma!(m.de_ele[1,1], BigFloat(2), BigFloat(3), m.de_ele[1,1])
            @test m.de_ele[1,1] == 6
            @test m.de_ele[2,2] == 0
        end
    end

    @testset "_copyto_independent! copies values not references (Vector{BigFloat})" begin
        setprecision(BigFloat, 200) do
            src = SDPA._independent_zeros(BigFloat, 4)
            SDPA.mp_fma!(src[1], BigFloat(2), BigFloat(3), src[1])  # src[1] == 6
            SDPA.mp_mul!(src[3], BigFloat(5), BigFloat(5))         # src[3] == 25
            dst = SDPA._independent_zeros(BigFloat, 4)
            SDPA._copyto_independent!(dst, src)
            # Elements must be independent (not shared references) ...
            for i in eachindex(dst)
                @test dst[i] !== src[i]
            end
            # ... and value-equal.
            @test dst == src
            # Mutating dst must not affect src (in-place MPFR mutation).
            SDPA.mp_fma!(dst[1], BigFloat(1), BigFloat(1), dst[1])
            @test dst[1] == 7
            @test src[1] == 6
            @test src[3] == 25
        end
    end

    @testset "_copyto_independent! copies values (Vector{Float64})" begin
        src = [1.0, 2.0, 3.0]
        dst = zeros(Float64, 3)
        SDPA._copyto_independent!(dst, src)
        @test dst == src
        dst[1] = 42.0
        @test src[1] == 1.0
    end

    @testset "_independent_zeros creates independent elements (Vector{BigFloat})" begin
        setprecision(BigFloat, 200) do
            V = SDPA._independent_zeros(BigFloat, 5)
            @test all(iszero, V)
            @test V[1] !== V[2]
            @test V[1] !== V[5]
            SDPA.mp_add!(V[1], BigFloat(2), BigFloat(3))
            @test V[1] == 5
            @test V[2] == 0
            @test count(!iszero, V) == 1
        end
    end

    @testset "_independent_zeros (Float64) Matrix and Vector" begin
        M = SDPA._independent_zeros(Float64, 3, 4)
        @test size(M) == (3, 4)
        @test all(iszero, M)
        M[1,1] = 2.5
        @test M[1,2] == 0.0  # isbits: independence trivial, values must be right
        V = SDPA._independent_zeros(Float64, 4)
        @test length(V) == 4
        @test all(iszero, V)
        V[1] = 3.5
        @test V[2] == 0.0
    end

    @testset "_fill_independent! nonzero value (BigFloat Matrix + Vector)" begin
        setprecision(BigFloat, 200) do
            M = zeros(BigFloat, 3, 3)
            SDPA._fill_independent!(M, BigFloat(7))
            @test all(==(BigFloat(7)), M)
            @test M[1,1] !== M[1,2]
            SDPA.mp_mul!(M[1,1], BigFloat(2), BigFloat(2))
            @test M[1,1] == 4
            @test M[1,2] == 7  # neighbor stays at val
            V = zeros(BigFloat, 4)
            SDPA._fill_independent!(V, BigFloat(7))
            @test all(==(BigFloat(7)), V)
            @test V[1] !== V[2]
            SDPA.mp_mul!(V[1], BigFloat(3), BigFloat(3))
            @test V[1] == 9
            @test V[2] == 7
        end
    end

    @testset "_copyto_independent! copies values not references (Matrix{BigFloat})" begin
        setprecision(BigFloat, 200) do
            src = SDPA._independent_zeros(BigFloat, 3, 3)
            SDPA.mp_fma!(src[1,1], BigFloat(2), BigFloat(3), src[1,1])  # 6
            SDPA.mp_mul!(src[2,3], BigFloat(5), BigFloat(5))            # 25
            dst = SDPA._independent_zeros(BigFloat, 3, 3)
            SDPA._copyto_independent!(dst, src)
            # Elements must be independent (not shared references) ...
            for idx in eachindex(dst)
                @test dst[idx] !== src[idx]
            end
            # ... and value-equal.
            @test dst == src
            # Mutating dst must not affect src (in-place MPFR mutation).
            SDPA.mp_fma!(dst[1,1], BigFloat(1), BigFloat(1), dst[1,1])
            @test dst[1,1] == 7
            @test src[1,1] == 6
            @test src[2,3] == 25
        end
    end

    @testset "_fill_independent! (Float64 Vector)" begin
        V = zeros(Float64, 4)
        SDPA._fill_independent!(V, 2.0)
        @test all(==(2.0), V)
        V[1] = 42.0
        @test V[2] == 2.0
    end

    @testset "_copyto_independent! copies values (Matrix{Float64})" begin
        src = reshape([1.0, 2.0, 3.0, 4.0], 2, 2)
        dst = zeros(Float64, 2, 2)
        SDPA._copyto_independent!(dst, src)
        @test dst == src
        dst[1,1] = 42.0
        @test src[1,1] == 1.0
    end

    @testset "SDPAVector constructor creates independent BigFloat elements" begin
        setprecision(BigFloat, 200) do
            v = SDPA.SDPAVector{BigFloat}(5)
            @test length(v) == 5
            @test all(iszero, v.ele)
            # Each element must be its own BigFloat object (own mpfr_t).
            @test v.ele[1] !== v.ele[2]
            @test v.ele[1] !== v.ele[5]
            # In-place MPFR mutation of one element leaves the others alone.
            SDPA.mp_fma!(v.ele[1], BigFloat(2), BigFloat(3), v.ele[1])
            @test v.ele[1] == 6
            @test v.ele[2] == 0
            @test count(!iszero, v.ele) == 1

            # Nonzero value: also independent.
            w = SDPA.SDPAVector{BigFloat}(5; value=BigFloat(3))
            @test all(==(BigFloat(3)), w.ele)
            @test w.ele[1] !== w.ele[2]
            SDPA.mp_mul!(w.ele[1], w.ele[1], BigFloat(2))
            @test w.ele[1] == 6
            @test w.ele[2] == 3
        end
    end

    @testset "SDPAVector setzero!/setidentity! create independent elements" begin
        setprecision(BigFloat, 200) do
            v = SDPA.SDPAVector{BigFloat}(5; value=BigFloat(1))
            SDPA.setzero!(v)
            @test all(iszero, v.ele)
            @test v.ele[1] !== v.ele[2]
            SDPA.setidentity!(v, 2)
            @test all(==(BigFloat(2)), v.ele)
            @test v.ele[1] !== v.ele[2]
            SDPA.mp_add!(v.ele[1], v.ele[1], BigFloat(1))
            @test v.ele[1] == 3
            @test v.ele[2] == 2
        end
    end

    @testset "SDPAVector copyto! copies values not references" begin
        setprecision(BigFloat, 200) do
            src = SDPA.SDPAVector{BigFloat}(5; value=BigFloat(7))
            dst = SDPA.SDPAVector{BigFloat}(5)
            SDPA.copyto!(dst, src)
            for i in 1:5
                @test dst.ele[i] !== src.ele[i]
                @test dst.ele[i] == src.ele[i]
            end
            # Mutating dst must not affect src (in-place MPFR mutation).
            SDPA.mp_add!(dst.ele[1], dst.ele[1], BigFloat(1))
            @test dst.ele[1] == 8
            @test src.ele[1] == 7
        end
    end

    @testset "SDPAVector copy produces independent copy" begin
        setprecision(BigFloat, 200) do
            src = SDPA.SDPAVector{BigFloat}(5; value=BigFloat(7))
            c = copy(src)
            for i in 1:5
                @test c.ele[i] !== src.ele[i]
                @test c.ele[i] == src.ele[i]
            end
            SDPA.mp_add!(c.ele[1], c.ele[1], BigFloat(1))
            @test c.ele[1] == 8
            @test src.ele[1] == 7
        end
    end

    @testset "SDPABlockVector inherits independent elements via SDPAVector" begin
        setprecision(BigFloat, 200) do
            bv = SDPA.SDPABlockVector{BigFloat}(2, [2, -3]; value=BigFloat(4))
            # Block 1 is an SDP block of dim 2 -> 3 packed entries; block 2 is
            # an LP block of length 3.
            @test length(bv.ele[1]) == 3
            @test length(bv.ele[2]) == 3
            @test all(==(BigFloat(4)), bv.ele[1].ele)
            @test all(==(BigFloat(4)), bv.ele[2].ele)
            # Independent within a block ...
            @test bv.ele[1].ele[1] !== bv.ele[1].ele[2]
            @test bv.ele[2].ele[1] !== bv.ele[2].ele[2]
            # ... and across blocks.
            @test bv.ele[1].ele[1] !== bv.ele[2].ele[1]
            SDPA.mp_add!(bv.ele[1].ele[1], bv.ele[1].ele[1], BigFloat(1))
            @test bv.ele[1].ele[1] == 5
            @test bv.ele[1].ele[2] == 4
            @test bv.ele[2].ele[1] == 4
        end
    end

    @testset "SDPAVector Float64 correctness" begin
        v = SDPA.SDPAVector{Float64}(4)
        @test all(iszero, v.ele)
        w = SDPA.SDPAVector{Float64}(4; value=2.5)
        @test all(==(2.5), w.ele)
        dst = SDPA.SDPAVector{Float64}(4)
        SDPA.copyto!(dst, w)
        @test dst == w
        dst.ele[1] = 42.0
        @test w.ele[1] == 2.5
        c = copy(w)
        @test c == w
        c.ele[2] = -1.0
        @test w.ele[2] == 2.5
        s = similar(v)
        @test all(iszero, s.ele)
        SDPA.setidentity!(s, 3)
        @test all(==(3.0), s.ele)
    end

    @testset "SDPASparseMatrix dense branch independent elements" begin
        setprecision(BigFloat, 200) do
            # Dense constructor must not share BigFloat objects.
            m = SDPA.SDPASparseMatrix{BigFloat}(3, 3, :dense, 9)
            @test m.type === :dense
            @test all(iszero, m.de_ele)
            @test m.de_ele[1,1] !== m.de_ele[1,2]
            @test m.de_ele[1,1] !== m.de_ele[3,3]
            # In-place MPFR mutation of one element leaves others alone.
            SDPA.mp_fma!(m.de_ele[1,1], BigFloat(2), BigFloat(3), m.de_ele[1,1])
            @test m.de_ele[1,1] == 6
            @test m.de_ele[1,2] == 0
            @test m.de_ele[3,3] == 0

            # setzero! re-fills with independent elements.
            SDPA.setzero!(m)
            @test all(iszero, m.de_ele)
            @test m.de_ele[1,1] !== m.de_ele[1,2]
            SDPA.mp_mul!(m.de_ele[2,2], BigFloat(3), BigFloat(3))
            @test m.de_ele[2,2] == 9
            @test m.de_ele[1,1] == 0

            # setidentity!: diagonal == 1, off-diagonals independent zeros.
            SDPA.setidentity!(m)
            @test all(m.de_ele[i,i] == 1 for i in 1:3)
            @test all(m.de_ele[i,j] == 0 for i in 1:3 for j in 1:3 if i != j)
            @test m.de_ele[1,2] !== m.de_ele[1,3]
            @test m.de_ele[1,2] !== m.de_ele[3,1]
            SDPA.mp_add!(m.de_ele[1,2], m.de_ele[1,2], BigFloat(5))
            @test m.de_ele[1,2] == 5
            @test m.de_ele[1,3] == 0
            @test m.de_ele[2,2] == 1
        end
    end

    @testset "SDPASparseMatrix change_to_dense! independent de_ele" begin
        setprecision(BigFloat, 200) do
            sm = SDPA.SDPASparseMatrix{BigFloat}(4, 4, :sparse, 4)
            SDPA.setidentity!(sm)
            SDPA.change_to_dense!(sm; force=true)
            @test sm.type === :dense
            # Off-diagonal entries must be independent objects.
            @test sm.de_ele[1,1] !== sm.de_ele[1,2]
            @test sm.de_ele[2,1] !== sm.de_ele[2,3]
            SDPA.mp_fma!(sm.de_ele[1,2], BigFloat(1), BigFloat(1), sm.de_ele[1,2])
            @test sm.de_ele[1,2] == 1
            @test sm.de_ele[1,3] == 0
        end
    end

    @testset "SDPASparseMatrix change_to_dense! off-diagonal mirror independent" begin
        setprecision(BigFloat, 200) do
            # Sparse matrix with off-diagonal entries (1,2) and (2,3), so the
            # dense conversion mirrors them into (2,1) and (3,2).
            sm = SDPA.SDPASparseMatrix{BigFloat}(3, 3, :sparse, 2)
            sm.row_index[1] = 1; sm.column_index[1] = 2; sm.sp_ele[1] = BigFloat(7)
            sm.row_index[2] = 2; sm.column_index[2] = 3; sm.sp_ele[2] = BigFloat(5)
            sm.nonZeroEffect = 2
            SDPA.change_to_dense!(sm; force=true)
            @test sm.type === :dense
            # Values mirrored symmetrically.
            @test sm.de_ele[1, 2] == 7
            @test sm.de_ele[2, 1] == 7
            @test sm.de_ele[2, 3] == 5
            @test sm.de_ele[3, 2] == 5
            # Mirrored slots must be DISTINCT BigFloat objects (in-place
            # MPFR ops on one must not corrupt the other).
            @test sm.de_ele[1, 2] !== sm.de_ele[2, 1]
            @test sm.de_ele[2, 3] !== sm.de_ele[3, 2]
            # In-place mutation of one mirror leaves the other unchanged.
            SDPA.mp_add!(sm.de_ele[1, 2], sm.de_ele[1, 2], BigFloat(1))
            @test sm.de_ele[1, 2] == 8
            @test sm.de_ele[2, 1] == 7
            SDPA.mp_add!(sm.de_ele[3, 2], sm.de_ele[3, 2], BigFloat(2))
            @test sm.de_ele[3, 2] == 7
            @test sm.de_ele[2, 3] == 5
        end
    end

    @testset "SDPASparseMatrix copyto! copies sp_ele values not references" begin
        setprecision(BigFloat, 200) do
            src = SDPA.SDPASparseMatrix{BigFloat}(4, 4, :sparse, 4)
            SDPA.setidentity!(src)  # 4 diagonal sp_ele entries == 1
            dst = SDPA.SDPASparseMatrix{BigFloat}(4, 4, :sparse, 4)
            SDPA.copyto!(dst, src)
            for i in 1:4
                @test dst.sp_ele[i] !== src.sp_ele[i]
                @test dst.sp_ele[i] == src.sp_ele[i]
            end
            # Mutating dst must not affect src (in-place MPFR mutation).
            SDPA.mp_add!(dst.sp_ele[1], dst.sp_ele[1], BigFloat(1))
            @test dst.sp_ele[1] == 2
            @test src.sp_ele[1] == 1
        end
    end

    @testset "SDPASparseMatrix copy produces independent sp_ele" begin
        setprecision(BigFloat, 200) do
            src = SDPA.SDPASparseMatrix{BigFloat}(4, 4, :sparse, 4)
            SDPA.setidentity!(src)
            c = copy(src)
            for i in 1:4
                @test c.sp_ele[i] !== src.sp_ele[i]
                @test c.sp_ele[i] == src.sp_ele[i]
            end
            SDPA.mp_mul!(c.sp_ele[2], c.sp_ele[2], BigFloat(10))
            @test c.sp_ele[2] == 10
            @test src.sp_ele[2] == 1
        end
    end

    @testset "SDPADenseMatrix setidentity! off-diagonals independent" begin
        setprecision(BigFloat, 200) do
            m = SDPA.SDPADenseMatrix{BigFloat}(3, 3)
            SDPA.setidentity!(m)
            @test m.de_ele[1,1] == 1
            @test m.de_ele[2,2] == 1
            @test m.de_ele[3,3] == 1
            @test all(m.de_ele[i,j] == 0 for i in 1:3 for j in 1:3 if i != j)
            # Off-diagonal elements are independent objects.
            @test m.de_ele[1,2] !== m.de_ele[1,3]
            @test m.de_ele[1,2] !== m.de_ele[2,1]
            SDPA.mp_add!(m.de_ele[1,2], m.de_ele[1,2], BigFloat(5))
            @test m.de_ele[1,2] == 5
            @test m.de_ele[1,3] == 0
            @test m.de_ele[2,2] == 1
        end
    end

    @testset "SDPASparseMatrix dense branch Float64 correctness" begin
        m = SDPA.SDPASparseMatrix{Float64}(3, 3, :dense, 9)
        @test all(iszero, m.de_ele)
        SDPA.setzero!(m)
        @test all(iszero, m.de_ele)
        SDPA.setidentity!(m, 2.5)
        @test all(m.de_ele[i,i] == 2.5 for i in 1:3)
        @test all(m.de_ele[i,j] == 0.0 for i in 1:3 for j in 1:3 if i != j)
        src = SDPA.SDPASparseMatrix{Float64}(2, 2, :sparse, 2)
        SDPA.setidentity!(src)
        dst = SDPA.SDPASparseMatrix{Float64}(2, 2, :sparse, 2)
        SDPA.copyto!(dst, src)
        @test dst.sp_ele == src.sp_ele
        c = copy(src)
        @test c.sp_ele == src.sp_ele
        dst.sp_ele[1] = 42.0
        @test src.sp_ele[1] == 1.0
    end

    @testset "SDPADenseMatrix setidentity! Float64 correctness" begin
        m = SDPA.SDPADenseMatrix{Float64}(3, 3)
        SDPA.setidentity!(m, 2.0)
        @test all(m.de_ele[i,i] == 2.0 for i in 1:3)
        @test all(m.de_ele[i,j] == 0.0 for i in 1:3 for j in 1:3 if i != j)
    end

    @testset "get_inv_chol_and_inv! symmetrization keeps BigFloat pairs distinct" begin
        setprecision(BigFloat, 200) do
            # Symmetric PD input: M = [[3,1,0],[1,3,1],[0,1,3]].
            bs = [3, -1]
            M = SDPA.SDPADenseLinearSpace{BigFloat}(bs)
            M.SDP_block[1][1, 1] = BigFloat(3); M.SDP_block[1][1, 2] = BigFloat(1)
            M.SDP_block[1][1, 3] = BigFloat(0); M.SDP_block[1][2, 1] = BigFloat(1)
            M.SDP_block[1][2, 2] = BigFloat(3); M.SDP_block[1][2, 3] = BigFloat(1)
            M.SDP_block[1][3, 1] = BigFloat(0); M.SDP_block[1][3, 2] = BigFloat(1)
            M.SDP_block[1][3, 3] = BigFloat(3)
            M.LP_block[1] = BigFloat(4)
            inv_chol = SDPA.SDPADenseLinearSpace{BigFloat}(bs)
            inverse_mat = SDPA.SDPADenseLinearSpace{BigFloat}(bs)
            work_mat = SDPA.SDPADenseLinearSpace{BigFloat}(bs)
            ok = SDPA.get_inv_chol_and_inv!(inv_chol, inverse_mat, M, work_mat)
            @test ok == true
            inv_blk = inverse_mat.SDP_block[1].de_ele
            # Symmetrization must value-copy avg, NOT alias the transposed pair.
            for j in 1:3, i in 1:j-1
                @test inv_blk[i, j] !== inv_blk[j, i]
            end
            # In-place MPFR mutation of one entry must not corrupt the mirror.
            v21 = inv_blk[2, 1]; v31 = inv_blk[3, 1]; v32 = inv_blk[3, 2]
            SDPA.mp_add!(inv_blk[1, 2], inv_blk[1, 2], BigFloat(1))
            SDPA.mp_add!(inv_blk[2, 3], inv_blk[2, 3], BigFloat(1))
            @test inv_blk[2, 1] == v21
            @test inv_blk[3, 1] == v31
            @test inv_blk[3, 2] == v32
            @test inv_blk[1, 2] == v21 + 1
            @test inv_blk[2, 3] == v32 + 1
        end
    end
end
