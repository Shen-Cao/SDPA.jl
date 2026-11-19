using SDPA
using Test
using LinearAlgebra
using Printf

@testset "Rgemm" begin
    setprecision(BigFloat, 256) do
        # ----------------------------------------------------------------
        # Functional tests: compare against LinearAlgebra.mul!
        # ----------------------------------------------------------------
        @testset "NN (no transpose)" begin
            for (m, n, k) in [(1, 1, 1), (2, 2, 2), (3, 4, 5), (5, 3, 4)]
                A = rand(BigFloat, m, k) .* BigFloat(10)
                B = rand(BigFloat, k, n) .* BigFloat(10)
                alpha = BigFloat(1.5)
                beta = BigFloat(0.5)
                # Expected
                # Actual
                C_actual = SDPA._independent_zeros(BigFloat, m, n)
                # Init C to 1 to test beta
                SDPA._fill_independent!(C_actual, BigFloat(1))
                C_expected_with_beta = alpha * A * B + beta * ones(BigFloat, m, n)
                SDPA.Rgemm!("N", "N", m, n, k, alpha, A, B, beta, C_actual)
                @test maximum(abs.(C_actual .- C_expected_with_beta)) < BigFloat("1e-30")
            end
        end

        @testset "TN (transpose A)" begin
            for (m, n, k) in [(2, 2, 2), (3, 4, 5)]
                A = rand(BigFloat, k, m) .* BigFloat(10)
                B = rand(BigFloat, k, n) .* BigFloat(10)
                alpha = BigFloat(1.0)
                beta = BigFloat(0.0)
                C_expected = alpha * transpose(A) * B
                C_actual = SDPA._independent_zeros(BigFloat, m, n)
                SDPA.Rgemm!("T", "N", m, n, k, alpha, A, B, beta, C_actual)
                @test maximum(abs.(C_actual .- C_expected)) < BigFloat("1e-30")
            end
        end

        @testset "NT (transpose B)" begin
            for (m, n, k) in [(2, 2, 2), (3, 4, 5)]
                A = rand(BigFloat, m, k) .* BigFloat(10)
                B = rand(BigFloat, n, k) .* BigFloat(10)
                alpha = BigFloat(1.0)
                beta = BigFloat(0.0)
                C_expected = alpha * A * transpose(B)
                C_actual = SDPA._independent_zeros(BigFloat, m, n)
                SDPA.Rgemm!("N", "T", m, n, k, alpha, A, B, beta, C_actual)
                @test maximum(abs.(C_actual .- C_expected)) < BigFloat("1e-30")
            end
        end

        @testset "TT (both transposed)" begin
            for (m, n, k) in [(2, 2, 2), (3, 4, 5)]
                A = rand(BigFloat, k, m) .* BigFloat(10)
                B = rand(BigFloat, n, k) .* BigFloat(10)
                alpha = BigFloat(1.0)
                beta = BigFloat(0.0)
                C_expected = alpha * transpose(A) * transpose(B)
                C_actual = SDPA._independent_zeros(BigFloat, m, n)
                SDPA.Rgemm!("T", "T", m, n, k, alpha, A, B, beta, C_actual)
                @test maximum(abs.(C_actual .- C_expected)) < BigFloat("1e-30")
            end
        end

        @testset "Float64 dispatch" begin
            A = rand(3, 3)
            B = rand(3, 3)
            C = zeros(3, 3)
            SDPA.Rgemm!("N", "N", 3, 3, 3, 1.0, A, B, 0.0, C)
            @test maximum(abs.(C .- A * B)) < 1e-12
        end

        @testset "invalid control strings throw" begin
            # A typo like "n" must error, not be silently treated as
            # transpose. Both transa and transb are validated (BigFloat
            # path); "C" stays valid for real types.
            A = rand(BigFloat, 2, 2)
            B = rand(BigFloat, 2, 2)
            C = SDPA._independent_zeros(BigFloat, 2, 2)
            for (ta, tb) in (
                ("n", "N"), ("t", "N"), ("N", "n"),
                ("X", "N"), ("N", "X"), ("", "N"), ("N", ""),
                ("transpose", "N"), ("N", "Tran"), ("NN", "N"),
            )
                @test_throws ErrorException SDPA.Rgemm!(
                    ta, tb, 2, 2, 2, BigFloat(1), A, B, BigFloat(0), C)
            end
            # "C" remains accepted (same as "T" for real types).
            SDPA.Rgemm!("C", "C", 2, 2, 2, BigFloat(1), A, B, BigFloat(0), C)
        end

        @testset "edge cases" begin
            # alpha=0: C should be just beta*C
            A = ones(BigFloat, 2, 2)
            B = ones(BigFloat, 2, 2)
            C = BigFloat.([1 2; 3 4])
            C_orig = copy(C)
            SDPA.Rgemm!("N", "N", 2, 2, 2, zero(BigFloat), A, B,
                        BigFloat(2), C)
            @test C == 2 * C_orig

            # k=0: no update
            C = BigFloat.([1 2; 3 4])
            C_orig = copy(C)
            SDPA.Rgemm!("N", "N", 2, 2, 0, one(BigFloat), A, B,
                        one(BigFloat), C)
            @test C == C_orig

            # alpha=0, beta=0: C must be zeroed with DISTINCT BigFloat
            # objects per element. `fill!(C, zero(T))` stores the SAME
            # object everywhere, so a later in-place MPFR op on C[1,1]
            # (e.g. mp_add!) would silently corrupt C[1,2] as well.
            A = ones(BigFloat, 2, 2)
            B = ones(BigFloat, 2, 2)
            C = SDPA._independent_zeros(BigFloat, 2, 2)
            @test C[1, 1] !== C[1, 2]  # pre-condition: independent objects
            SDPA.Rgemm!("N", "N", 2, 2, 2, zero(BigFloat), A, B,
                        zero(BigFloat), C)
            @test all(iszero, C)
            @test C[1, 1] !== C[1, 2]  # must not be fill!-aliased
            SDPA.mp_add!(C[1, 1], C[1, 1], one(BigFloat))
            @test isone(C[1, 1])
            @test iszero(C[1, 2])  # unchanged: not corrupted by C[1,1] add
        end

        @testset "aliasing rejection" begin
            A = rand(BigFloat, 2, 2)
            B = rand(BigFloat, 2, 2)
            @test_throws ArgumentError SDPA.Rgemm!(
                "N", "N", 2, 2, 2, BigFloat(1), A, B, BigFloat(0), A)  # C === A
            @test_throws ArgumentError SDPA.Rgemm!(
                "N", "N", 2, 2, 2, BigFloat(1), A, B, BigFloat(0), B)  # C === B
            @test_throws ArgumentError SDPA.Rgemm!(
                "T", "N", 2, 2, 2, BigFloat(1), A, B, BigFloat(0), A)  # transposed path
            # View aliasing into the same parent is detected too.
            parent = rand(BigFloat, 4, 4)
            @test_throws ArgumentError SDPA.Rgemm!(
                "N", "N", 2, 2, 2, BigFloat(1),
                view(parent, 1:2, 1:2), B, BigFloat(0),
                view(parent, 1:2, 1:2))
            # Disjoint views are NOT aliases: the call must succeed.
            C = SDPA._independent_zeros(BigFloat, 2, 2)
            SDPA.Rgemm!("N", "N", 2, 2, 2, BigFloat(1),
                        view(parent, 1:2, 1:2), B, BigFloat(0), C)
            @test C ≈ view(parent, 1:2, 1:2) * B
            # Float64 path enforces the same contract.
            Af = rand(2, 2); Bf = rand(2, 2)
            @test_throws ArgumentError SDPA.Rgemm!(
                "N", "N", 2, 2, 2, 1.0, Af, Bf, 0.0, Af)
            # On the BLAS quick-return paths (k=0 here) the operands
            # are never referenced, so C === A must be ACCEPTED: the call
            # returns beta*A without throwing.
            Aq = BigFloat.([1 2; 3 4])
            Bq = BigFloat.([5 6; 7 8])
            expected = 2 .* Aq
            out = SDPA.Rgemm!("N", "N", 2, 2, 0, BigFloat(3), Aq, Bq,
                              BigFloat(2), Aq)  # k=0, C === A, beta != 1
            @test out === Aq
            @test Aq == expected  # C := beta*C = beta*A
            # beta=1 hits the first quick-return branch (untouched C).
            @test SDPA.Rgemm!("N", "N", 2, 2, 0, BigFloat(3), Aq, Bq,
                              BigFloat(1), Aq) === Aq
            @test Aq == expected
            # alpha=0 (beta=1) is also a quick-return: aliased C accepted.
            @test SDPA.Rgemm!("N", "N", 2, 2, 2, zero(BigFloat), Aq, Bq,
                              BigFloat(1), Bq) === Bq
        end

        # ----------------------------------------------------------------
        # Allocation budget: the hard assertion is a byte budget on
        # Rgemm! itself. Rgemm! allocates far less than
        # LinearAlgebra.mul! (which creates per-element temporaries for
        # BigFloat); the mul! figure is measured and printed for
        # reference only — it is not asserted.
        # ----------------------------------------------------------------
        @testset "allocation" begin
            n = 10
            A = rand(BigFloat, n, n)
            B = rand(BigFloat, n, n)
            C = zeros(BigFloat, n, n)
            # Warm up
            SDPA.Rgemm!("N", "N", n, n, n, one(BigFloat), A, B,
                        zero(BigFloat), C)
            mul!(C, A, B)
            # Measure Rgemm
            allocs_rgemm = @allocated SDPA.Rgemm!("N", "N", n, n, n,
                                                  one(BigFloat), A, B,
                                                  zero(BigFloat), C)
            # Measure LinearAlgebra.mul!
            allocs_mul = @allocated mul!(C, A, B)
            @printf("  n=%d: Rgemm=%d bytes, mul!=%d bytes, ratio=%.2f\n",
                    n, allocs_rgemm, allocs_mul, allocs_rgemm / allocs_mul)
            # Zero-alloc direct-write pattern with headroom for
            # allocator-level noise.
            @test allocs_rgemm ≤ 1100
        end
    end

    # ----------------------------------------------------------------
    # The parallel @tasks branch (m*n*k > 4096) must be bit-identical
    # across thread counts.
    #
    # This test solves the same problem in subprocesses at
    # JULIA_NUM_THREADS=1 and =4 and requires byte-identical output.
    # ----------------------------------------------------------------
    @testset "Parallel @tasks bit-identity across thread counts" begin
        proj_root = dirname(dirname(@__DIR__))
        julia_bin = joinpath(Sys.BINDIR, "julia")
        script = """
        using SDPA
        setprecision(BigFloat, 200) do
            m, n, k = 64, 64, 64   # 64³ = 262144 > 4096 → parallel @tasks branch
            out = String[]
            for (name, ta, tb) in (("NN","N","N"), ("TN","T","N"),
                                   ("NT","N","T"), ("TT","T","T"))
                Am = ta == "N" ? (m, k) : (k, m)
                Bm = tb == "N" ? (k, n) : (n, k)
                A = SDPA._independent_zeros(BigFloat, Am...)
                B = SDPA._independent_zeros(BigFloat, Bm...)
                # Deterministic pseudo-random data (no Random import: the
                # Pkg.test sandbox only exposes declared deps, and the two
                # subprocesses must build identical matrices).
                for i in 1:Am[1], l in 1:Am[2]
                    v = (sin(i * 1.7 + l * 0.3) + 1) * 0.5 + 0.1
                    SDPA.mp_set!(A[i, l], BigFloat(v))
                end
                for l in 1:Bm[1], j in 1:Bm[2]
                    v = (sin(l * 2.3 + j * 1.1) + 1) * 0.5 + 0.1
                    SDPA.mp_set!(B[l, j], BigFloat(v))
                end
                C = SDPA._independent_zeros(BigFloat, m, n)
                SDPA.Rgemm!(ta, tb, m, n, k, BigFloat(1), A, B, BigFloat(0), C)
                for i in 1:m, j in 1:n
                    push!(out, string(C[i, j]))
                end
            end
            println(join(out, "\\n"))
        end
        """
        env1 = copy(ENV)
        env1["JULIA_PKG_SERVER"] = ""  # offline: load the resolved project only
        env1["JULIA_NUM_THREADS"] = "1"
        cmd1 = Cmd(`$julia_bin --project=$proj_root -e $script`; env=env1)
        env4 = copy(ENV)
        env4["JULIA_PKG_SERVER"] = ""  # offline: load the resolved project only
        env4["JULIA_NUM_THREADS"] = "4"
        cmd4 = Cmd(`$julia_bin --project=$proj_root -e $script`; env=env4)
        out1 = read(cmd1, String)
        out4 = read(cmd4, String)
        @test !isempty(out1)
        @test out1 == out4  # bit-identical across thread counts
    end
end
