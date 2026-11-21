@testset "min_eigen" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            # --- Lanczos must not return the 1e50 sentinel when the ---
            # --- starting vector is an exact eigenvector.            ---
            # Regression: the restart test `kk >= floor(sqrt(max(1,k)))`
            # suppressed the C++ k=0 restart (0 >= 0, always true), so a
            # Krylov-invariant start (residual <= 1e-16 at k=1) broke out
            # with min_val still at its 1e50 sentinel — which StepLength
            # reads as "no negative curvature" and awards a full alphaBD
            # step. With the C++ `floor(sqrt(k))` restart the first Ritz
            # value is computed at k=0 and the true eigenvalue returns.
            begin
                n = 21  # above the Lanczos dispatch threshold (n > 20)
                u = T[sqrt(T(1 / n)) for _ in 1:n]
                X = Matrix{T}(I, n, n) - T(6) * u * u'  # λ_min = -5, ones is its eigenvector
                L = Matrix{T}(I, n, n)
                @test SDPA.min_eigen_lanczos(L, X) ≈ T(-5) rtol=1e-6
                @test SDPA.min_eigen_dispatch(L, X) ≈ T(-5) rtol=1e-6
                # Identity X: every vector is an exact eigenvector (λ = 1).
                # BigFloat exact arithmetic makes this case deterministic.
                @test SDPA.min_eigen_lanczos(L, Matrix{T}(I, n, n)) ≈ T(1) rtol=1e-6
                @test SDPA.min_eigen_dispatch(L, Matrix{T}(I, n, n)) ≈ T(1) rtol=1e-6
            end

            # --- min_eigen_qr_block (exact tests on small matrices) ---
            M = T[3 0 0; 0 1 0; 0 0 2]
            @test SDPA.min_eigen_qr_block(M) ≈ T(1) rtol=eps(T)^0.5

            M2 = T[2 1; 1 2]
            @test SDPA.min_eigen_qr_block(M2) ≈ T(1) rtol=eps(T)^0.5

            # --- min_eigen_dispatch (small matrices use QR path) ---
            L = Matrix{T}(I, 3, 3)
            X = T[3 0 0; 0 1 0; 0 0 2]
            @test SDPA.min_eigen_dispatch(L, X) ≈ T(1) rtol=eps(T)^0.5

            # --- min_eigen_lanczos on larger matrices (n > 20) ---
            # Here every diagonal eigendirection overlaps the seed. Check
            # the quality of the corrected Ritz estimate for this matrix;
            # it is not a certified lower bound for an arbitrary matrix.
            n = 50
            Xdiag = Matrix(Diagonal(T.(1:n)))
            L_I = Matrix{T}(I, n, n)
            lam_qr = SDPA.min_eigen_qr_block(copy(Xdiag))
            @test lam_qr ≈ T(1) rtol=T(1e-10)

            lam_lanczos = SDPA.min_eigen_dispatch(L_I, Xdiag)
            @test lam_lanczos <= T(1) + T(1e-3)
            @test lam_lanczos >= T(0)

            # --- Lanczos work vectors must be independent BigFloat
            # elements. Repeat calls on the same inputs must be bit-identical
            # (no cross-call shared-buffer corruption). ---
            n2 = 24  # > 20 ⇒ Lanczos path; small enough to stay fast
            X2 = Matrix(Diagonal(T.(1:n2)))
            L2 = Matrix{T}(I, n2, n2)
            r1 = SDPA.min_eigen_lanczos(L2, X2)
            r2 = SDPA.min_eigen_lanczos(L2, X2)
            @test r1 == r2

            # --- min_eigen_qr_block must not mutate its input. Use an
            # independent value copy as the reference so aliasing (BigFloat
            # reference copy) cannot mask in-place mutation of `M`. ---
            M3 = T[3 0 0; 0 1 0; 0 0 2]
            M3ref = SDPA._copy_independent(M3)
            lam3 = SDPA.min_eigen_qr_block(M3)
            @test lam3 ≈ T(1) rtol=eps(T)^0.5
            @test M3 == M3ref
        end
    end

    # --- BigFloat Lanczos iteration loops must be zero-allocation. ---
    # The four per-iteration loops (q normalization, alpha dot-product,
    # 3-term recurrence, beta norm) use the in-place mp_* idiom
    # (mp_mul!, mp_add_mul!, mp_sub_mul!, mp_set!); this test locks the
    # allocation budget.
    #
    # Measurement method: cap iterations at 1 (`max_iter=1`) so the
    # tridiagonal BigFloat eigen restart (which dominates the default-run
    # allocation, ~5.3 MB per solve at n=30) runs exactly once; the four
    # loops are then the dominant measurable difference. Warm up first so
    # `@allocated` excludes compilation.
    #
    # Current measurement (200-bit): @allocated ≈ 31,472 bytes for n=30,
    # max_iter=1. The 45,000 threshold leaves ample headroom over that
    # budget.
    setprecision(BigFloat, 200) do
        n = 30
        T = BigFloat
        X2 = Matrix(Diagonal(T.(1:n)))
        L2 = Matrix{T}(I, n, n)
        SDPA.min_eigen_lanczos(L2, X2, 1)  # warm up (compilation)
        a = @allocated SDPA.min_eigen_lanczos(L2, X2, 1)
        # +10% headroom (bytes) against allocator-level noise on Julia
        # point releases.
        @test a < 49_500
    end
end

@testset "dimension checks and error paths" begin
    # Non-square inputs must throw a descriptive error, NOT UndefVarError.
    @test_throws ErrorException SDPA.min_eigen_lanczos(rand(2, 3), rand(2, 2))
    @test_throws ErrorException SDPA.min_eigen_lanczos(rand(2, 2), rand(2, 3))
    # min_eigen_dispatch: full dimension checks on both matrices.
    @test_throws DimensionMismatch SDPA.min_eigen_dispatch(rand(2, 3), rand(2, 2))
    @test_throws DimensionMismatch SDPA.min_eigen_dispatch(rand(2, 2), rand(2, 3))
    @test_throws DimensionMismatch SDPA.min_eigen_dispatch(rand(3, 2), rand(2, 2))
end
