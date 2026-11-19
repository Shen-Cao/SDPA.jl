@testset "linalg dispatch" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            # update_eigen!: factor a known symmetric matrix, verify eigenvalues.
            # M = diag(3, 1, 2) has eigenvalues {1, 2, 3}.
            M = T[3 0 0; 0 1 0; 0 0 2]
            vals = SDPA.update_eigen!(M)
            @test sort([vals...]) ≈ [T(1), T(2), T(3)]

            if T === BigFloat
                # update_eigen! must VALUE-copy F.vectors into X. The
                # BigFloat overload writes via mp_set! (in-place mutation of
                # X's EXISTING element objects), so the object identity of
                # X's elements is preserved across the call. A reference copy
                # (default copyto!) would REPLACE the slots with F.vectors'
                # objects instead — detectable by object identity.
                M4 = T[3 0 0; 0 1 0; 0 0 2]
                orig11 = M4[1, 1]  # keep a handle on the original element object
                SDPA.update_eigen!(M4)
                @test M4[1, 1] === orig11
            end

            # posdef_fact!: SPD factorization round-trip.
            # A = [4 2; 2 3] has eigenvalues {1.38, 5.62}, both positive.
            A = T[4 2; 2 3]
            fact = SDPA.posdef_fact!(Symmetric(A, :U))
            @test issuccess(fact)

            # Indefinite matrix: posdef_fact! should fail.
            B = T[1 2; 2 1]  # eigenvalues {-1, 3}
            fact_bad = SDPA.posdef_fact!(Symmetric(B, :U))
            @test !issuccess(fact_bad)

            # symm_fact!: factor an indefinite matrix directly.
            D = T[0 1; 1 0]
            fact_symm = SDPA.symm_fact!(Symmetric(D, :U))
            @test issuccess(fact_symm)

            # increase_diag!: verify the diagonal gets bumped.
            E = T[0.5 1; 1 0.5]
            SDPA.increase_diag!(E)
            @test E[1, 1] > T(0.5)
            @test E[2, 2] > T(0.5)
            # Off-diagonal unchanged.
            @test E[1, 2] == T(1)
            @test E[2, 1] == T(1)
        end
    end
end

@testset "increase_diag! increments by the documented step" begin
    # dispatch.jl documents a single ladder step (no loop):
    #     A[j,j] <- (1 + 1e-5) * max(A[j,j], 1000*eps(T))
    # The testset above only checks "> 0.5"; pin the EXACT increment
    # magnitude and both legs of the max — the plain-scale leg and the
    # 1000*eps(T) floor leg (entries at/below the floor are floored
    # BEFORE the scaling).
    for T in (Float64, BigFloat)
        step = 1 + T(1e-5)
        floor_val = 1000 * eps(T)

        # Plain leg: a healthy diagonal grows by exactly the factor
        # (1 + 1e-5) (same expression as the source, so == is exact;
        # mirroring the documented semantics, not a re-implementation).
        E = T[0.5 1; 1 0.5]
        SDPA.increase_diag!(E)
        @test E[1, 1] == step * T(0.5)
        @test E[2, 2] == step * T(0.5)
        # Off-diagonal structure untouched.
        @test E[1, 2] == T(1)
        @test E[2, 1] == T(1)

        # Floor leg: a zero and a negative diagonal both land on the
        # scaled floor (1 + 1e-5) * 1000*eps(T) — not on a scaled
        # version of their original value.
        F = T[0 2; 2 -1]
        SDPA.increase_diag!(F)
        @test F[1, 1] == step * floor_val
        @test F[2, 2] == step * floor_val
        @test F[1, 2] == T(2)
        @test F[2, 1] == T(2)
        # The floor is genuinely tiny relative to the plain leg — the
        # documented 1000*eps magnitude, not something coarse like 1e-5.
        @test step * floor_val < T(1e-5) * T(1e-5)
    end
end
