using SDPA
using Test
using LinearAlgebra
using GenericLinearAlgebra  # BigFloat cholesky!/inv/backslash references

@testset "Rlapack" begin
    setprecision(BigFloat, 256) do
        @testset "Rpotf2! upper" begin
            for n in (1, 2, 3, 5)
                # Build SPD matrix M = Q'*Q for random Q.
                Q = rand(BigFloat, n, n) .* BigFloat(10)
                M = Q' * Q + n * one(BigFloat) * I  # add regularization
                M_u = deepcopy(M)
                # Symmetrize upper triangle (the lower will be ignored).
                for j in 1:n, i in 1:j
                    M_u[i, j] = M[i, j]
                end
                # Reference
                F_ref = cholesky!(Hermitian(deepcopy(M), :U); check=false)
                @test issuccess(F_ref)
                # Rpotf2!
                info = SDPA.Rpotf2!("U", n, M_u)
                @test info == 0
                # Compare with reference (upper-triangular factor).
                # F_ref.factors is the U factor.
                for j in 1:n, i in 1:j
                    @test abs(M_u[i, j] - F_ref.factors[i, j]) < BigFloat("1e-30")
                end
            end
        end

        @testset "Rpotf2! lower" begin
            for n in (2, 3)
                Q = rand(BigFloat, n, n) .* BigFloat(10)
                M = Q' * Q + n * one(BigFloat) * I
                M_l = SDPA._copy_independent(M)
                for j in 1:n, i in j:n
                    SDPA.mp_set!(M_l[i, j], M[i, j])
                end
                info = SDPA.Rpotf2!("L", n, M_l)
                @test info == 0
                # Verify L * L' ≈ M
                L = LowerTriangular(M_l)
                reconstructed = L * L'
                @test maximum(abs.(reconstructed .- M)) < BigFloat("1e-30")
            end
        end

        @testset "Rpotf2! non-SPD" begin
            n = 3
            M = zeros(BigFloat, n, n)
            M[1, 1] = -1  # negative diagonal
            info = SDPA.Rpotf2!("U", n, M)
            @test info > 0  # should fail
        end

        @testset "Rtrsm! upper no-trans" begin
            # Solve U * X = B where U is upper triangular.
            for n in (2, 3)
                U_raw = rand(BigFloat, n, n) .* BigFloat(10)
                # Make it upper triangular with nonzero diagonal.
                U = UpperTriangular(U_raw)
                for i in 1:n
                    U_raw[i, i] += one(BigFloat)
                end
                B = rand(BigFloat, n, n) .* BigFloat(10)
                B_expected = copy(B)
                # Reference: U \ B
                X_ref = U \ B
                # Actual: Rtrsm!("L", "U", "N", "N", n, n, one, U_raw, B)
                SDPA.Rtrsm!("L", "U", "N", "N", n, n, one(BigFloat), U_raw, B)
                @test maximum(abs.(B .- X_ref)) < BigFloat("1e-30")
            end
        end

        @testset "Rtrsm! lower trans" begin
            # Solve L' * X = B where L is lower triangular.
            for n in (2, 3)
                L_raw = rand(BigFloat, n, n) .* BigFloat(10)
                for i in 1:n
                    L_raw[i, i] += one(BigFloat)
                end
                L = LowerTriangular(L_raw)
                B = rand(BigFloat, n, n) .* BigFloat(10)
                X_ref = L' \ B
                SDPA.Rtrsm!("L", "L", "T", "N", n, n, one(BigFloat), L_raw, B)
                @test maximum(abs.(B .- X_ref)) < BigFloat("1e-30")
            end
        end

        @testset "Float64 dispatch" begin
            n = 5
            Q = rand(n, n)
            M = Q' * Q + n * I
            info = SDPA.Rpotf2!("U", n, M)
            @test info == 0
        end

        @testset "Rtrsm! right side BigFloat" begin
            # Solve X * op(A) = alpha * B for all four (uplo, transa) combos.
            for (uplo, transa) in (("L", "N"), ("L", "T"), ("U", "N"), ("U", "T"))
                n = 5
                m = 4
                A_raw = rand(BigFloat, n, n) .* BigFloat(10)
                for i in 1:n
                    A_raw[i, i] += BigFloat(10)
                end
                if uplo == "U"
                    for j in 1:n, i in (j + 1):n
                        A_raw[i, j] = zero(BigFloat)
                    end
                else
                    for j in 1:n, i in 1:(j - 1)
                        A_raw[i, j] = zero(BigFloat)
                    end
                end
                B_orig = rand(BigFloat, m, n) .* BigFloat(10)
                alpha = BigFloat("0.75")
                B = deepcopy(B_orig)
                X = SDPA.Rtrsm!("R", uplo, transa, "N", m, n, alpha, A_raw, B)
                @test X === B
                # Ground truth: X * op(A) == alpha * B_orig.
                opA = transa == "N" ? copy(A_raw) : Matrix(transpose(A_raw))
                @test maximum(abs.(X * opA .- alpha .* B_orig)) < BigFloat("1e-60")
                # Independent dense reference via inv.
                X_ref = (alpha .* B_orig) * inv(opA)
                @test maximum(abs.(X .- X_ref)) < BigFloat("1e-60")
            end
        end

        @testset "Rtrsm! right unit diag" begin
            # diag="U": stored diagonal must be ignored (treated as 1).
            n = 5
            m = 4
            A_raw = rand(BigFloat, n, n) .* BigFloat(10)
            for j in 1:n, i in 1:(j - 1)
                A_raw[i, j] = zero(BigFloat)
            end
            for i in 1:n
                A_raw[i, i] = BigFloat("12345.6789")  # garbage: must be ignored
            end
            B_orig = rand(BigFloat, m, n) .* BigFloat(10)
            alpha = BigFloat("1.5")
            B = deepcopy(B_orig)
            X = SDPA.Rtrsm!("R", "L", "N", "U", m, n, alpha, A_raw, B)
            L_unit = copy(A_raw)
            for i in 1:n
                L_unit[i, i] = one(BigFloat)
            end
            X_ref = (alpha .* B_orig) * inv(L_unit)
            @test maximum(abs.(X .- X_ref)) < BigFloat("1e-60")
        end

        @testset "Rpotrf! lower blocked regression" begin
            # Blocked lower Cholesky calls Rtrsm!("R", "L", "T", ...).
            n = 70
            M = Matrix{BigFloat}(rand(n, n) * 0.1)
            M = M' * M + Matrix{BigFloat}(LinearAlgebra.I, n, n)
            # Value-copy (plain copy() aliases BigFloat buffers on Julia 1.12).
            M_orig = SDPA._copy_independent(M)
            info = SDPA.Rpotrf!("L", n, M; nb=64)
            @test info == 0
            L = LowerTriangular(M)
            @test maximum(abs.(L * L' .- M_orig)) < BigFloat("1e-30")
        end

        @testset "Rtrsm! Float64 fallback" begin
            # The fallback must honor side, transa, diag and alpha.
            # For side="L" A is m×m; for side="R" A is n×n.
            for (side, uplo, transa) in (
                ("L", "U", "N"), ("L", "U", "T"),
                ("L", "L", "N"), ("L", "L", "T"),
                ("R", "U", "N"), ("R", "U", "T"),
                ("R", "L", "N"), ("R", "L", "T"),
            )
                m = 4
                n = 5
                dim = side == "L" ? m : n
                A_raw = rand(dim, dim) .* 10.0
                for i in 1:dim
                    A_raw[i, i] += 10.0
                end
                if uplo == "U"
                    for j in 1:dim, i in (j + 1):dim
                        A_raw[i, j] = 0.0
                    end
                else
                    for j in 1:dim, i in 1:(j - 1)
                        A_raw[i, j] = 0.0
                    end
                end
                B = rand(m, n) .* 10.0
                B_orig = copy(B)
                alpha = 0.5
                SDPA.Rtrsm!(side, uplo, transa, "N", m, n, alpha, A_raw, B)
                opA = transa == "N" ? copy(A_raw) : Matrix(transpose(A_raw))
                X_ref = side == "L" ? (opA \ (alpha .* B_orig)) : ((alpha .* B_orig) * inv(opA))
                @test maximum(abs.(B .- X_ref)) < 1e-9
            end
            # diag="U": diagonal treated as identity.
            for (side, uplo, transa) in (("R", "L", "N"), ("L", "U", "T"))
                m = 4
                n = 5
                dim = side == "L" ? m : n
                A_raw = rand(dim, dim) .* 10.0
                if uplo == "U"
                    for j in 1:dim, i in (j + 1):dim
                        A_raw[i, j] = 0.0
                    end
                else
                    for j in 1:dim, i in 1:(j - 1)
                        A_raw[i, j] = 0.0
                    end
                end
                for i in 1:dim
                    A_raw[i, i] = 12345.6789  # garbage: must be ignored
                end
                B = rand(m, n) .* 10.0
                B_orig = copy(B)
                SDPA.Rtrsm!(side, uplo, transa, "U", m, n, 1.0, A_raw, B)
                A_unit = copy(A_raw)
                for i in 1:dim
                    A_unit[i, i] = 1.0
                end
                opA = transa == "N" ? copy(A_unit) : Matrix(transpose(A_unit))
                X_ref = side == "L" ? (opA \ B_orig) : (B_orig * inv(opA))
                @test maximum(abs.(B .- X_ref)) < 1e-9
            end
        end

        @testset "invalid control strings throw" begin
            # A typo like "n" must error, not be silently treated as its
            # closest valid flag.
            A = rand(BigFloat, 3, 3)
            for i in 1:3
                A[i, i] += BigFloat(5)
            end
            B = rand(BigFloat, 3, 3)
            for (side, uplo, transa, diag) in (
                ("x", "L", "N", "N"), ("L", "x", "N", "N"),
                ("L", "L", "x", "N"), ("L", "L", "N", "x"),
                ("l", "L", "N", "N"), ("R", "L", "n", "N"),
                ("", "L", "N", "N"), ("L", "U", "C", "u"),
            )
                @test_throws ErrorException SDPA.Rtrsm!(
                    side, uplo, transa, diag, 3, 3, BigFloat(1), A, B)
            end
            # Rpotf2!: uplo validation on BOTH the BigFloat and the
            # Float64/32 methods.
            Mb = rand(BigFloat, 3, 3)
            @test_throws ErrorException SDPA.Rpotf2!("u", 3, Mb)
            @test_throws ErrorException SDPA.Rpotf2!("", 3, Mb)
            Mf = rand(3, 3)
            @test_throws ErrorException SDPA.Rpotf2!("u", 3, Mf)
            @test_throws ErrorException SDPA.Rpotf2!("l", 3, Mf)
            @test_throws ErrorException SDPA.Rpotf2!("", 3, Mf)
            # Rsyrk!: uplo/trans validation. Only the BLAS short forms
            # are accepted; internal callers (Rpotrf!) pass the BLAS
            # short forms.
            As = rand(BigFloat, 3, 2)
            C = SDPA._independent_zeros(BigFloat, 3, 3)
            for (uplo, trans) in (
                ("u", "N"), ("U", "t"), ("", "N"), ("U", ""),
                ("Lower", "N"), ("U", "Transpose"), ("x", "N"), ("U", "x"),
            )
                @test_throws ErrorException SDPA.Rsyrk!(
                    uplo, trans, 3, 2, BigFloat(1), As, BigFloat(0), C)
            end
            # "C" transa stays valid for Rtrsm! (same as "T" for real).
            SDPA.Rtrsm!("R", "U", "C", "N", 3, 3, BigFloat(1), A, B)
        end

        @testset "NaN pivots are failures" begin
            # BigFloat Rpotf2!: a NaN pivot must return the pivot index, not
            # 0 (ajj <= 0 is false for NaN, which would read as success).
            Mb = BigFloat[4 1; 1 3]
            Mb[1, 1] = NaN
            @test SDPA.Rpotf2!("U", 2, Mb) == 1
            @test SDPA.Rpotrf!("U", 2, Mb) == 1
            Ml = BigFloat[4 1; 1 3]
            Ml[2, 2] = NaN
            @test SDPA.Rpotf2!("L", 2, Ml) == 2
            # Float64 fallback: the NaN pre-scan must report failure too.
            Mf = [4.0 1.0; 1.0 3.0]
            Mf[1, 1] = NaN
            @test SDPA.Rpotf2!("U", 2, Mf) == 1
            @test SDPA.Rpotrf!("U", 2, Mf) == 1
            Mf2 = [4.0 1.0; 1.0 3.0]
            Mf2[2, 2] = NaN
            @test SDPA.Rpotf2!("L", 2, Mf2) == 2
            @test SDPA.Rpotrf!("L", 2, Mf2) == 2
        end

        @testset "dimension checks" begin
            @test_throws DimensionMismatch SDPA.Rpotf2!("U", 3, rand(BigFloat, 2, 3))
            @test_throws DimensionMismatch SDPA.Rpotf2!("U", 3, rand(2, 3))
            @test_throws DimensionMismatch SDPA.Rpotrf!("U", 3, rand(2, 3))
            @test_throws DimensionMismatch SDPA.Rtrsm!(
                "L", "U", "N", "N", 3, 3, BigFloat(1), rand(BigFloat, 2, 2), rand(BigFloat, 3, 3))
            @test_throws DimensionMismatch SDPA.Rtrsm!(
                "L", "U", "N", "N", 3, 3, BigFloat(1), rand(BigFloat, 3, 3), rand(BigFloat, 2, 2))
            @test_throws DimensionMismatch SDPA.Rgemm!(
                "N", "N", 2, 2, 2, BigFloat(1), rand(BigFloat, 2, 3), rand(BigFloat, 2, 2),
                BigFloat(0), SDPA._independent_zeros(BigFloat, 2, 2))
        end

        @testset "Float64 fallback control-string validation" begin
            A = [4.0 1.0; 1.0 3.0]
            B = rand(2, 2)
            @test_throws ErrorException SDPA.Rtrsm!("x", "U", "N", "N", 2, 2, 1.0, A, B)
            @test_throws ErrorException SDPA.Rtrsm!("L", "x", "N", "N", 2, 2, 1.0, A, B)
            @test_throws ErrorException SDPA.Rtrsm!("L", "U", "n", "N", 2, 2, 1.0, A, B)
            @test_throws ErrorException SDPA.Rtrsm!("L", "U", "N", "u", 2, 2, 1.0, A, B)
            @test_throws ErrorException SDPA.Rgemm!("n", "N", 2, 2, 2, 1.0, A, B, 0.0, B)
            @test_throws ErrorException SDPA.Rgemm!("N", "n", 2, 2, 2, 1.0, A, B, 0.0, B)
        end

        @testset "Rsyrk! triangle-only semantics" begin
            # xSYRK contract: only the triangle of C named by uplo is read
            # (beta scaling) and written; the opposite triangle is left
            # untouched. A is a general matrix (read in full).
            n, k = 4, 3
            for T in (Float64, BigFloat)
                alpha = T(1.5)
                beta = T(0.5)
                rtol = T === BigFloat ? BigFloat("1e-30") : 1e-12
                for (uplo, trans) in (("U", "N"), ("U", "T"), ("L", "N"), ("L", "T"))
                    A = rand(T, trans == "N" ? (n, k) : (k, n)) .* T(10)
                    C0 = SDPA._copy_independent(rand(T, n, n) .* T(10))
                    C = SDPA._copy_independent(C0)
                    SDPA.Rsyrk!(uplo, trans, n, k, alpha, A, beta, C)
                    opA = trans == "N" ? A : transpose(A)
                    expected = alpha * opA * opA' + beta * C0
                    if uplo == "U"
                        for j in 1:n, i in 1:j
                            @test isapprox(C[i, j], expected[i, j]; rtol=rtol)
                        end
                        for j in 1:n, i in (j + 1):n
                            @test C[i, j] == C0[i, j]  # untouched triangle
                        end
                    else
                        for j in 1:n, i in j:n
                            @test isapprox(C[i, j], expected[i, j]; rtol=rtol)
                        end
                        for j in 1:n, i in 1:(j - 1)
                            @test C[i, j] == C0[i, j]  # untouched triangle
                        end
                    end
                end
            end
            # Dimension checks (parity with the checks above).
            @test_throws DimensionMismatch SDPA.Rsyrk!(
                "U", "N", 4, 3, BigFloat(1), rand(BigFloat, 3, 3), BigFloat(0),
                SDPA._independent_zeros(BigFloat, 4, 4))
            @test_throws DimensionMismatch SDPA.Rsyrk!(
                "U", "N", 4, 3, BigFloat(1), rand(BigFloat, 4, 3), BigFloat(0),
                SDPA._independent_zeros(BigFloat, 3, 3))
        end

        @testset "Rsyrk! beta=0 does not read C" begin
            # BLAS beta=0 contract: C := alpha*op(A)*op(A)' with OVERWRITE
            # semantics — C is not read: undef entries must not throw
            # UndefRefError, and NaNs prefilled in C must not propagate.
            n, k = 3, 2
            for T in (BigFloat, Float64)
                alpha = T(1.5)
                rtol = T === BigFloat ? BigFloat("1e-30") : 1e-12
                for (uplo, trans) in (("U", "N"), ("U", "T"), ("L", "N"), ("L", "T"))
                    A = rand(T, trans == "N" ? (n, k) : (k, n)) .* T(10)
                    opA = trans == "N" ? A : transpose(A)
                    ref = alpha * opA * opA'
                    # Named-triangle index pairs (re-iterated in (a)-(c)).
                    tri = Tuple{Int,Int}[]
                    if uplo == "U"
                        for j in 1:n, i in 1:j; push!(tri, (j, i)); end
                    else
                        for j in 1:n, i in j:n; push!(tri, (j, i)); end
                    end
                    # (a) undef C, beta=0: must not throw UndefRefError
                    # and must write alpha*op(A)*op(A)'
                    # into the named triangle without reading C.
                    C = Matrix{T}(undef, n, n)
                    SDPA.Rsyrk!(uplo, trans, n, k, alpha, A, zero(T), C)
                    for (j, i) in tri
                        @test isapprox(C[i, j], ref[i, j]; rtol=rtol)
                    end
                    # (b) NaN-prefilled C, beta=0: BLAS overwrite semantics
                    # — NaNs in the named triangle are overwritten (finite
                    # alpha*A'A values), never propagated.
                    C2 = fill!(Matrix{T}(undef, n, n), T(NaN))
                    SDPA.Rsyrk!(uplo, trans, n, k, alpha, A, zero(T), C2)
                    for (j, i) in tri
                        @test isfinite(C2[i, j])
                        @test isapprox(C2[i, j], ref[i, j]; rtol=rtol)
                    end
                    # (c) beta=one behavior unchanged:
                    # alpha*op(A)*op(A)' + C_before on the named triangle.
                    A3 = rand(T, trans == "N" ? (n, k) : (k, n)) .* T(10)
                    opA3 = trans == "N" ? A3 : transpose(A3)
                    C3 = SDPA._copy_independent(rand(T, n, n) .* T(10))
                    C3_before = T === BigFloat ? SDPA._copy_independent(C3) :
                                               copy(C3)
                    expected3 = alpha * opA3 * opA3' + C3_before
                    SDPA.Rsyrk!(uplo, trans, n, k, alpha, A3, one(T), C3)
                    for (j, i) in tri
                        @test isapprox(C3[i, j], expected3[i, j]; rtol=rtol)
                    end
                end
            end
        end
    end
end
