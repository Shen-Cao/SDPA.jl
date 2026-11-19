using SDPA
using Test
using LinearAlgebra

# C++ parity minors: Lanczos restart and Cholesky diagonal contract.
#
# `min_eigen_lanczos`'s periodic eigen-restart must fire when
# `k > sqrt((mpf_class)nDim + 9)` (sdpa_linear.cpp:69 — sqrt of the SUM,
# arbitrary precision), NOT `k > floor(sqrt(nDim)) + 9`. The OUTER
# iteration cap `iter_cap = min(n, floor(Int, sqrt(n)) + 10)`
# (min_eigen.jl:81/107, C++ linear.cpp:43 `k < sqrt(nDim) + 10`) is a
# DIFFERENT expression and must not change.
#
# `sparse_cholesky!` non-positive-diagonal contract.
# Deliberate divergence from C++ `getCholesky` (sdpa_linear.cpp:
# 409-415, `diag < 0` → set to 0; else `1/sqrt(diag)`): C++ stores
# +Inf for an EXACTLY ZERO diagonal (silent garbage) and ALWAYS
# returns true. This repo guards with `diag <= 0` (zeroed AND
# `sparse_cholesky!` returns false), so the caller can detect a
# singular/non-PD B matrix.

@testset "Lanczos restart threshold matches C++ sqrt(n+9)" begin
    setprecision(BigFloat, 200) do
        # Discriminating (k, n) pairs: the C++ formula `k > sqrt(n + 9)` and
        # the alternative floored formula `k > floor(sqrt(n)) + 9` DISAGREE,
        # e.g. n = 6: sqrt(15) ≈ 3.873 vs floor(sqrt(6)) + 9 = 11 (k = 5
        # fires under C++, not under the floored formula).
        discriminating = [(4, 6), (5, 6), (7, 30), (10, 30), (12, 100), (19, 100)]
        # Agreement pairs: both formulas agree (guards against the
        # boundary moving in the other direction).
        agreeing = [(3, 6), (6, 30), (20, 100)]

        # The code's expression (via lanczos_restart_threshold) must match
        # the C++ expression evaluated at 200-bit.
        for (k, n) in vcat(discriminating, agreeing)
            cpp = k > sqrt(BigFloat(n + 9))
            code = k > SDPA.lanczos_restart_threshold(n)
            @test code == cpp
        end
        # The discriminating pairs must genuinely distinguish the
        # alternative floored formula from the C++ formula.
        for (k, n) in discriminating
            old = k > floor(Int, sqrt(n)) + 9
            cpp = k > sqrt(BigFloat(n + 9))
            @test old != cpp
            @test (k > SDPA.lanczos_restart_threshold(n)) == cpp
        end
    end

    # No-regression sanity: Lanczos on a 50×50 diagonal matrix still yields a
    # valid lower bound on λ_min = 1 (full convergence covered by the suite:
    # example1/arch0 step lengths).
    n = 50
    Xdiag = Matrix(Diagonal(Float64.(1:n)))
    L_I = Matrix{Float64}(I, n, n)
    lam = SDPA.min_eigen_dispatch(L_I, Xdiag)
    @test lam <= 1.0 + 1e-3
    @test lam >= 0.0
end

@testset "sparse_cholesky! zeroes non-positive diagonals and returns false" begin
    # Test-local full upper-triangle profile construction (LIVE path:
    # profile_from_skyline! + _fill_sparse_values!).
    function _profile_from_dense(A::AbstractMatrix{T}, ordering) where {T<:Real}
        m = size(A, 1)
        sb, di = SDPA.profile_from_skyline!(fill(m, m), T)
        SDPA._fill_sparse_values!(sb, A, ordering, di)
        return sb, di
    end

    for T in (Float64, BigFloat)
        @testset "$T" begin
            # 3×3 with a NEGATIVE second diagonal and an EXACTLY ZERO third
            # diagonal (the full profile stores all three diagonal entries on
            # sp_ele[diagonalIndex[i]]).
            # Contract: non-positive diagonal (≤ 0) → zeroed + returns false.
            A = T[2 0 0; 0 -1 0; 0 0 0]
            sparse_mat, diagonalIndex = _profile_from_dense(A, [1, 2, 3])
            ok = SDPA.sparse_cholesky!(sparse_mat, diagonalIndex)
            @test ok == false

            d1 = sparse_mat.sp_ele[diagonalIndex[1]]
            d2 = sparse_mat.sp_ele[diagonalIndex[2]]
            d3 = sparse_mat.sp_ele[diagonalIndex[3]]

            # Positive diagonal: unchanged 1/sqrt(2) path.
            @test isfinite(d1) && d1 > zero(T)
            @test d1 ≈ one(T) / sqrt(T(2)) rtol=eps(T)^0.5
            # Non-positive diagonal (negative OR exactly zero): the C++
            # zeroing of the factor state is kept, but both now signal
            # failure. (Deliberate divergence: C++ takes the 1/sqrt(0) =
            # +Inf path for an exactly-zero diagonal and always returns true.)
            @test d2 == zero(T)
            @test d3 == zero(T)
        end
    end
end
