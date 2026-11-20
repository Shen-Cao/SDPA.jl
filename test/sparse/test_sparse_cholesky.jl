using SDPA
using Test
using LinearAlgebra
using Random

# Unit tests for sparse_cholesky! and solve_systems_sparse!.
#
# These tests verify the profile-form Cholesky factorization and sparse
# triangular solve directly, without requiring a full SDP solve. The
# end-to-end sparse path would trigger on problems like theta5 (density
# 0.0003) but those are too large for CI.

# _profile_from_dense: canonical definition in test/_profile_from_dense.jl,
# shared (single definition per process) with the other sparse test files.
isdefined(@__MODULE__, :_profile_from_dense) ||
    include(joinpath(@__DIR__, "..", "_profile_from_dense.jl"))

@testset "sparse_cholesky! factorization" begin
    # 4×4 tridiagonal SPD matrix
    A = [4.0 -1.0  0.0  0.0;
        -1.0  4.0 -1.0  0.0;
         0.0 -1.0  4.0 -1.0;
         0.0  0.0 -1.0  4.0]
    ordering = [1, 2, 3, 4]  # identity permutation

    # Convert to profile form (lower triangle)
    sparse_mat, diagonalIndex = _profile_from_dense(A, ordering)

    # Factorize in-place
    ok = SDPA.sparse_cholesky!(sparse_mat, diagonalIndex)
    @test ok

    # Verify: reconstruct L*L' = A
    # After factorization, diagonal stores 1/sqrt(d), sub-diagonal stores L[r,j]
    n = 4
    L = zeros(n, n)
    idx = 0
    for j in 1:n
        ia1 = diagonalIndex[j]
        ia2 = diagonalIndex[j + 1]
        # Diagonal: L[j,j] = 1/sp_ele[ia1] (since sp_ele stores 1/sqrt(d))
        diag_val = sparse_mat.sp_ele[ia1]
        if diag_val > 0
            L[j, j] = 1.0 / diag_val
        end
        # Sub-diagonal
        for k1 in (ia1 + 1):(ia2 - 1)
            row = sparse_mat.column_index[k1]
            col = sparse_mat.row_index[k1]
            L[row, col] = sparse_mat.sp_ele[k1]
        end
    end

    # Check L * L' ≈ A
    @test L * L' ≈ A rtol=1e-10
end

@testset "solve_systems_sparse! triangular solve" begin
    # 3×3 SPD matrix
    A = [4.0  1.0  0.0;
         1.0  3.0  1.0;
         0.0  1.0  2.0]
    ordering = [1, 2, 3]

    sparse_mat, diagonalIndex = _profile_from_dense(A, ordering)
    SDPA.sparse_cholesky!(sparse_mat, diagonalIndex)

    # Solve A * x = b
    b = [1.0, 2.0, 3.0]
    x = zeros(3)
    SDPA.solve_systems_sparse!(x, sparse_mat, b, diagonalIndex)

    # Verify: A * x ≈ b
    residual = norm(A * x - b)
    @test residual < 1e-10

    # Compare with Julia's built-in solve
    x_expected = A \ b
    @test x ≈ x_expected rtol=1e-10
end

@testset "sparse_cholesky! BigFloat support" begin
    # Same 3×3 SPD matrix at BigFloat precision
    A = BigFloat[4  1  0;
                 1  3  1;
                 0  1  2]
    ordering = [1, 2, 3]

    sparse_mat, diagonalIndex = _profile_from_dense(A, ordering)
    ok = SDPA.sparse_cholesky!(sparse_mat, diagonalIndex)
    @test ok

    # Solve at BigFloat precision
    b = BigFloat[1, 2, 3]
    # x must have INDEPENDENT BigFloat elements: solve_systems_sparse!
    # writes via in-place mp_set! (zero-alloc), which corrupts aliased
    # elements (zeros(BigFloat, n) shares ONE object).
    x = SDPA._independent_zeros(BigFloat, 3)
    SDPA.solve_systems_sparse!(x, sparse_mat, b, diagonalIndex)

    residual = norm(A * x - b)
    @test residual < BigFloat("1e-30")
end

@testset "sparse path integration: dense→profile→cholesky→solve matches dense" begin
    # End-to-end integration test: exercises the full sparse Schur path
    # (_profile_from_dense → sparse_cholesky! → solve_systems_sparse!)
    # and verifies results match the dense cholesky path.
    m = 10
    A = randn(m, m)
    A = A * A' + m * I  # SPD
    ordering = collect(1:m)

    # Sparse path
    sparse_mat, diagonalIndex = _profile_from_dense(A, ordering)
    SDPA.sparse_cholesky!(sparse_mat, diagonalIndex)
    b = collect(1.0:m)
    x_sparse = zeros(m)
    SDPA.solve_systems_sparse!(x_sparse, sparse_mat, b, diagonalIndex)

    # Dense path (ground truth)
    x_dense = A \ b

    @test x_sparse ≈ x_dense rtol=1e-10
    @test norm(A * x_sparse - b) < 1e-10
end

@testset "non-identity ordering permuted solve" begin
    # compute_dyvec! must permute gvec before
    # sparse solve and reverse-permute DyVec after. Without this, any
    # non-identity AMD ordering produces silently wrong Δy values.
    A = Float64[4 -1 0 0; -1 4 -1 0; 0 -1 4 -1; 0 0 -1 4]
    ordering = [3, 1, 4, 2]  # non-identity

    sparse_mat, diagonalIndex = _profile_from_dense(A, ordering)
    SDPA.sparse_cholesky!(sparse_mat, diagonalIndex)

    b = [1.0, 2.0, 3.0, 4.0]

    # CORRECT path: permute → solve → reverse-permute
    g_perm = similar(b)
    SDPA.permute_vec!(g_perm, b, ordering)
    x_perm = similar(b)
    SDPA.solve_systems_sparse!(x_perm, sparse_mat, g_perm, diagonalIndex)
    x_correct = similar(b)
    for k in eachindex(ordering)  # inline reverse permute (out[ordering[i]] = v[i])
        x_correct[ordering[k]] = x_perm[k]
    end
    @test x_correct ≈ A \ b rtol=1e-10

    # WRONG path: no permutation
    x_wrong = similar(b)
    SDPA.solve_systems_sparse!(x_wrong, sparse_mat, b, diagonalIndex)
    @test norm(A * x_wrong - b) > 1e-3  # proves the bug existed
end

@testset "sparse_cholesky! signals non-PD / singular diagonal" begin
    # Contract: a non-positive diagonal (≤ 0) is zeroed (C++ parity
    # for the factor state) AND the factorization signals failure by
    # returning false.
    for T in (Float64, BigFloat)
        @testset "$T" begin
            # (a) non-PD: a negative diagonal must be zeroed AND signaled.
            A = T[2 0 0; 0 -1 0; 0 0 3]
            sparse_mat, diagonalIndex = _profile_from_dense(A, [1, 2, 3])
            ok = SDPA.sparse_cholesky!(sparse_mat, diagonalIndex)
            @test ok == false
            @test sparse_mat.sp_ele[diagonalIndex[2]] == zero(T)

            # (b) exactly-zero diagonal: zeroed and signaled (never
            # silently succeeds with 1/sqrt(0) = +Inf).
            A0 = T[2 0 0; 0 3 0; 0 0 0]
            sparse_mat0, diagonalIndex0 = _profile_from_dense(A0, [1, 2, 3])
            ok0 = SDPA.sparse_cholesky!(sparse_mat0, diagonalIndex0)
            @test ok0 == false
            @test sparse_mat0.sp_ele[diagonalIndex0[3]] == zero(T)

            # (c) SPD: still returns true and factorizes correctly.
            Aok = T[4 -1 0; -1 4 -1; 0 -1 4]
            sparse_mat_ok, diagonalIndex_ok = _profile_from_dense(Aok, [1, 2, 3])
            ok_ok = SDPA.sparse_cholesky!(sparse_mat_ok, diagonalIndex_ok)
            @test ok_ok == true
        end
    end
end

@testset "non-fill-in-closed profile raises ErrorException (not @assert)" begin
    # The fill-in-closed guard in _sparse_profile_update! must be a real
    # error, not an @assert (Julia may elide asserts at higher optimization
    # levels; a silent skip would truncate the factor). This exercises the
    # user-reachable trigger: a truncated profile (the state
    # SDPA_FORCE_SPARSE_PROFILE=1 can produce), where an update target is
    # missing from the skyline.
    #
    # Construction route: hand-build the broken profile
    # via the public profile_from_skyline! with a NON-closed skyline
    # [3, 2, 3] — profile_from_skyline! performs no closure itself (the
    # production caller passes the closed block-level skyline). Range 1
    # holds rows {1,2,3} but range 2 stops at the diagonal, so factoring
    # entry (2,1) emits the rank-1 update targeting (3,2) — which is absent
    # → the converted `found || error(...)` must fire, in both the generic
    # (Float64) and BigFloat branches of _sparse_profile_update!.
    for T in (Float64, BigFloat)
        @testset "$T" begin
            sb, di = SDPA.profile_from_skyline!([3, 2, 3], T)
            # Positive values (per-slot independent, BigFloat-safe) so the
            # (1,1) diagonal factors and the update loop actually runs.
            for k in eachindex(sb.sp_ele)
                sb.sp_ele[k] = one(T)
            end
            @test_throws ErrorException SDPA.sparse_cholesky!(sb, di)
        end
    end
end


# A rank-one sweep visits every contribution directly in source-column
# order, independent of the blocked scheduling and its reach calculation.
function _unblocked_profile_reference!(mat, di)
    for j in 1:mat.nRow
        SDPA._sparse_cholesky_factor_range!(mat, di, j) || return false
        for k in (di[j] + 1):(di[j + 1] - 1)
            SDPA._sparse_profile_update!(mat, di, k)
        end
    end
    return true
end

@testset "bounded trailing tiles preserve factors and solutions" begin
    rng = MersenneTwister(26559)
    m = 197 # crosses three full 64-column block boundaries
    separated = collect(1:m)
    separated[60:145] .= 145  # spills into two later tiles
    separated[149:195] .= 195 # separate component, partial tile
    late_exact = collect(1:m)
    late_exact[63:128] .= 128 # a reach exactly at the end of a tile
    setprecision(BigFloat, 160) do
        for T in (Float64, BigFloat),
            reach in (collect(1:m), min.(m, (1:m) .+ 3), separated, late_exact, fill(m, m))
            A = SDPA._independent_zeros(T, m, m)
            for j in 1:m
                A[j, j] = T(8)
                for i in (j + 1):reach[j]
                    val = T(rand(rng, -16:16)) / T(1024)
                    A[i, j] = val
                    A[j, i] = val
                end
            end
            profile, di = SDPA.profile_from_skyline!(reach, T)
            SDPA._fill_sparse_values!(profile, A, collect(1:m), di)
            expected = copy(profile)
            @test _unblocked_profile_reference!(expected, di)
            @test SDPA.sparse_cholesky!(profile, di)
            @test isequal(profile.sp_ele, expected.sp_ele)
            # Predictor/corrector use two right hand sides without touching
            # the existing factor. Check each residual against the original matrix.
            for _ in 1:2
                rhs = T.(rand(rng, -8:8, m))
                x = SDPA._independent_zeros(T, m)
                SDPA.solve_systems_sparse!(x, profile, rhs, di)
                tol = 1000 * eps(T)
                @test norm(A * x - rhs, Inf) <= tol * norm(rhs, Inf)
            end
            @test isequal(profile.sp_ele, expected.sp_ele)
        end
    end
end
