using LinearAlgebra
using SDPA
using Test

@testset "compact Schur shape and checked addressing" begin
    compact = SDPA._SchurSkyline{Float64}([3, 3, 3, 6, 6, 6, 7])
    @test compact.offsets == [1, 4, 6, 7, 10, 12, 13, 14]
    @test propertynames(compact) == (:offsets, :sp_ele)
    @test all(iszero, compact.sp_ele)
    # This interval's neighbors are also occupied, so just testing the
    # resulting global index bounds would wrongly accept the missing pair.
    @test SDPA._schur_assembly_slot(compact, compact.offsets, 2, 3, "LP") == 5
    @test_throws "LP pair (ii=2, jj=4) outside block-level profile" SDPA._schur_assembly_slot(
        compact, compact.offsets, 2, 4, "LP")
    @test_throws ErrorException SDPA._schur_assembly_slot(compact, compact.offsets, 2, 1, "LP")
    @test_throws ErrorException SDPA._schur_assembly_slot(compact, compact.offsets, 2, typemin(Int), "LP")
    @test_throws ErrorException SDPA._schur_assembly_slot(compact, compact.offsets, 2, typemax(Int), "LP")
    @test_throws BoundsError SDPA._schur_assembly_slot(compact, compact.offsets, 0, 1, "LP")

    for reaches in ([0], [2], [1, 1], [3, 2, 3], [2, typemax(Int)], [typemin(Int)])
        @test_throws ArgumentError SDPA._SchurSkyline{Float64}(reaches)
    end
    # Test a real representability boundary without allocating the profile.
    @test SDPA._checked_skyline_end(typemax(Int) - 1, 1, 1, 1, 1) == typemax(Int)
    @test_throws OverflowError SDPA._checked_skyline_end(typemax(Int), 1, 1, 1, 1)
    n = Int((isqrt(1 + 8big(typemax(Int))) - 1) ÷ 2)
    @test SDPA._schur_triangle_slots(n) == (big(n) * (big(n) + 1)) ÷ 2
    @test_throws OverflowError SDPA._schur_triangle_slots(n + 1)
    @test_throws ArgumentError SDPA._schur_triangle_slots(-1)

    empty_profile = SDPA._SchurSkyline{Float64}(Int[])
    @test empty_profile.offsets == [1]
    @test isempty(empty_profile.sp_ele)
    @test SDPA.sparse_cholesky!(empty_profile)
    @test isempty(SDPA.solve_systems_sparse!(Float64[], empty_profile, Float64[]))
    @test_throws DimensionMismatch SDPA.solve_systems_sparse!([1.0], empty_profile, [1.0])

    # Own the symbolic shape regardless of subsequent edits to the input.
    reaches = [2, 2]
    bf = SDPA._SchurSkyline{BigFloat}(reaches)
    reaches .= 1
    SDPA.mp_set_si!(bf.sp_ele[1], 7)
    @test bf.offsets == [1, 3, 4]
    @test bf.sp_ele == [7, 0, 0]
end

@testset "compact factors and two RHS preserve explicit-coordinate arithmetic" begin
    # Separated ranges, an exact boundary at 128, only-diagonal gaps,
    # and a source reaching two later 64-column tiles.
    n = 197
    separated = collect(1:n)
    separated[1:128] .= 128
    separated[151:194] .= 194
    setprecision(BigFloat, 160) do
        for T in (Float64, BigFloat), reaches in (collect(1:n), separated, fill(n, n))
            @testset "$T, $(reaches[1]) first reach" begin
                compact = SDPA._SchurSkyline{T}(reaches)
                explicit, di = SDPA.profile_from_skyline!(reaches, T)
                @test compact.offsets == di
                A = SDPA._independent_zeros(T, n, n)
                k = 0
                for i in 1:n, j in i:reaches[i]
                    k += 1
                    v = i == j ? T(8) : T(mod(17i + 13j, 29) - 14) / T(1024)
                    A[i, j] = A[j, i] = v
                    if T === BigFloat
                        SDPA.mp_set!(compact.sp_ele[k], v)
                        SDPA.mp_set!(explicit.sp_ele[k], v)
                    else
                        compact.sp_ele[k] = explicit.sp_ele[k] = v
                    end
                end
                @test SDPA.sparse_cholesky!(explicit, di)
                @test SDPA.sparse_cholesky!(compact)
                @test isequal(compact.sp_ele, explicit.sp_ele)
                snapshot = SDPA._copy_independent(compact.sp_ele)
                for b in (T.(mod.(1:n, 11)), T.(mod.(1:n, 19) .- 9))
                    b0 = SDPA._copy_independent(b)
                    want, got = (SDPA._independent_zeros(T, n) for _ in 1:2)
                    SDPA.solve_systems_sparse!(want, explicit, b, di)
                    SDPA.solve_systems_sparse!(got, compact, b)
                    @test isequal(got, want)
                    @test isequal(b, b0)
                    @test norm(A * got - b, Inf) <= 1000 * eps(T) * norm(b, Inf)
                end
                @test isequal(compact.sp_ele, snapshot)
            end
            bad = SDPA._SchurSkyline{T}([1, 2])
            @test !SDPA.sparse_cholesky!(bad)  # Existing nonpositive-pivot contract.
        end
    end
end

@testset "explicit sparse matrices can still represent holes" begin
    # Column 1 has rows 1 and 4, not 1 and 2. The old explicit solve has
    # to honor actual row indices even when those differ from the new
    # implicit address arithmetic. (The true factor also has that hole.)
    for T in (Float64, BigFloat)
        mat = SDPA.SDPASparseMatrix{T}(4, 4, :sparse, 5)
        mat.nonZeroEffect = 5
        mat.row_index .= [1, 1, 2, 3, 4]
        mat.column_index .= [1, 4, 2, 3, 4]
        mat.sp_ele .= T[4, 1, 4, 4, 4]
        di = [1, 3, 4, 5, 6]
        @test SDPA.sparse_cholesky!(mat, di)
        b = T[1, 2, 3, 4]
        x = SDPA._independent_zeros(T, 4)
        SDPA.solve_systems_sparse!(x, mat, b, di)
        @test T[4 0 0 1; 0 4 0 0; 0 0 4 0; 1 0 0 4] * x ≈ b
    end
end

@testset "mixed sparse assembly in a permuted compact envelope" begin
    fixtures = joinpath(@__DIR__, "..", "fixtures")
    setprecision(BigFloat, 160) do
        for T in (Float64, BigFloat), name in ("sparse_lp_profile", "f4_mixed_dense_sparse", "lp_first_mixed")
            p = SDPA.read_dat_s(joinpath(fixtures, "$name.dat-s"), T)
            ordering = circshift(collect(p.m:-1:1), 2)
            inverse = invperm(ordering)
            reaches = SDPA.build_skyline_from_problem!(p, ordering, inverse)
            compact = SDPA._SchurSkyline{T}(reaches)
            explicit, di = SDPA.profile_from_skyline!(reaches, T)
            pt = SDPA.SDPASolution{T}(p.m, p.blockStruct, T(10))
            states = map(((compact, Int[]), (explicit, di))) do (storage, offsets)
                ns = SDPA.NewtonState{T}(p.m, p.blockStruct; allocate_dense_bmat=false)
                ns.use_sparse = true
                ns.sparse_bmat = storage
                ns.diagonalIndex = offsets
                ns.ordering, ns.reverse_ordering = ordering, inverse
                SDPA.compute_formula!(ns, p)
                SDPA.make_aggrigate_index_sdp!(ns, p, ordering)
                SDPA.make_bmat!(ns, p, pt, SDPA.SDPAWorkVariables{T}(p.blockStruct))
                return ns
            end
            @test states[1].SDP_location_sparse_bMat == states[2].SDP_location_sparse_bMat
            @test isequal(compact.sp_ele, explicit.sp_ele)
            @test states[1].bmat === nothing
            @test isempty(states[1].diagonalIndex)
            if name == "sparse_lp_profile"
                # At X=10I,Z=10I the only off-diagonal contribution is
                # the product of the two LP coefficients, not an SDP term.
                @test compact.sp_ele[2] == T(6)
                ns = states[1]
                ns.sparse_bmat = SDPA._SchurSkyline{T}(collect(1:p.m))
                # A valid *closed* shape can still be too short for the
                # problem. Confirm production assembly rejects the LP edge.
                SDPA.make_aggrigate_index_sdp!(ns, p, ordering)
                @test_throws "LP pair (ii=1, jj=2) outside block-level profile" SDPA.make_bmat!(
                    ns, p, pt, SDPA.SDPAWorkVariables{T}(p.blockStruct))
            elseif name == "f4_mixed_dense_sparse"
                ns = states[1]
                ns.sparse_bmat = SDPA._SchurSkyline{T}(collect(1:p.m))
                @test_throws "outside block-level profile" SDPA.make_aggrigate_index_sdp!(ns, p, ordering)
            end
        end
    end
end
