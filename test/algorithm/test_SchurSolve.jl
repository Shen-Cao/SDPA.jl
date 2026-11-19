@testset "SchurSolve" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            # --- cholesky_schur! on SPD matrix ---
            ns = SDPA.NewtonState{T}(3, [2])
            # Set bmat to a known SPD matrix: [[4,2,1],[2,5,3],[1,3,6]].
            ns.bmat.de_ele[1, 1] = T(4); ns.bmat.de_ele[1, 2] = T(2); ns.bmat.de_ele[1, 3] = T(1)
            ns.bmat.de_ele[2, 1] = T(2); ns.bmat.de_ele[2, 2] = T(5); ns.bmat.de_ele[2, 3] = T(3)
            ns.bmat.de_ele[3, 1] = T(1); ns.bmat.de_ele[3, 2] = T(3); ns.bmat.de_ele[3, 3] = T(6)

            ok = SDPA.cholesky_schur!(ns)
            @test ok == true
            @test ns.cholesky_factorization !== nothing

            # --- cholesky_schur! on indefinite matrix ---
            ns_bad = SDPA.NewtonState{T}(2, [2])
            ns_bad.bmat.de_ele[1, 1] = T(0); ns_bad.bmat.de_ele[1, 2] = T(1)
            ns_bad.bmat.de_ele[2, 1] = T(1); ns_bad.bmat.de_ele[2, 2] = T(0)
            ok_bad = SDPA.cholesky_schur!(ns_bad)
            # This matrix (eigenvalues -1, +1) is indefinite, so the
            # ladder does NOT recover: Float64 fails cholesky!(check=false),
            # BigFloat fails plain Rpotrf (info != 0, no diagonal
            # adjust). Empirically verified at both T: returns false and
            # leaves no cached factorization.
            @test ok_bad isa Bool
            @test ok_bad == false
            @test ns_bad.cholesky_factorization === nothing

            # --- compute_dyvec! end-to-end solve ---
            # B·Δy = g where B is SPD and g is known.
            ns2 = SDPA.NewtonState{T}(3, [2])
            B = T[4 2 1; 2 5 3; 1 3 6]
            g = T[7; 10; 10]
            SDPA._copyto_independent!(ns2.bmat.de_ele, B)
            copyto!(ns2.gvec.ele, g)
            ok2 = SDPA.compute_dyvec!(ns2, true)  # is_predictor
            @test ok2 == true
            # Verify residual ||B·Δy - g|| / ||g|| < eps(T)^0.5
            dy = ns2.DyVec.ele
            residual = B * dy - g
            rel_res = norm(residual) / max(norm(g), one(T))
            @test rel_res < eps(T)^0.5

            # --- compute_dyvec! reuses factorization for corrector ---
            # Change gvec (simulating corrector's new g), re-solve.
            g2 = T[1; 2; 3]
            copyto!(ns2.gvec.ele, g2)
            ok3 = SDPA.compute_dyvec!(ns2, false)  # corrector
            @test ok3 == true
            dy2 = ns2.DyVec.ele
            residual2 = B * dy2 - g2
            rel_res2 = norm(residual2) / max(norm(g2), one(T))
            @test rel_res2 < eps(T)^0.5

            # --- compute_dzmat! ---
            # Build a small problem so multi_plus_to_a! works.
            blockStruct = [2]
            b = SDPA.SDPAVector{T}([T(1)])
            C = SDPA.SDPASparseLinearSpace{T}(1, 0)
            C.SDP_sp_index = [1]
            C.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
            SDPA.setelement_sdp!(C, 1, 1, 1, T(1))
            A1 = SDPA.SDPASparseLinearSpace{T}(1, 0)
            A1.SDP_sp_index = [1]
            A1.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
            SDPA.setelement_sdp!(A1, 1, 1, 1, T(2))
            problem = SDPA.SDPAProblem{T}(1, 1, blockStruct, b, C, [A1])
            SDPA.initialize_index!(problem)
            SDPA.change_to_dense!(problem.C; force=true)
            SDPA.change_to_dense!(problem.A[1]; force=true)

            current_pt = SDPA.SDPASolution{T}(1, blockStruct, T(1))
            res = SDPA.SDPAResiduals{T}(problem, current_pt)
            SDPA.compute!(res, problem, current_pt)

            ns3 = SDPA.NewtonState{T}(1, blockStruct)
            ns3.DyVec.ele[1] = T(3)
            SDPA.compute_dzmat!(ns3, problem, res, SDPA.noINFO)
            # DzMat = -Δy_1 * A_1 + dualMat = -3*[[2,0],[0,0]] + dualMat
            @test ns3.DzMat.SDP_block[1].de_ele[1, 1] ≈ T(-6) + res.dualMat.SDP_block[1].de_ele[1, 1]

            # --- compute_dxmat! ---
            # Set up r_zinvMat and DzMat for a hand-checkable case.
            ns4 = SDPA.NewtonState{T}(1, [2])
            current_pt4 = SDPA.SDPASolution{T}(1, [2], T(1))
            work4 = SDPA.SDPAWorkVariables{T}([2])
            # r_zinvMat = βμ·Z⁻¹ - X = βμ·I - I (with βμ = 0)
            SDPA.setidentity!(ns4.r_zinvMat.SDP_block[1], T(0))  # zero out
            # Set r_zinvMat to identity for simplicity.
            ns4.r_zinvMat.SDP_block[1].de_ele[1, 1] = T(1)
            ns4.r_zinvMat.SDP_block[1].de_ele[2, 2] = T(1)
            # DzMat = zero
            SDPA.setzero!(ns4.DzMat)
            # Δx = r_zinvMat - X·Δz·Z⁻¹ = I - I·0·I = I
            SDPA.compute_dxmat!(ns4, current_pt4, work4)
            @test ns4.DxMat.SDP_block[1].de_ele[1, 1] ≈ T(1)
            @test ns4.DxMat.SDP_block[1].de_ele[2, 2] ≈ T(1)
            @test isapprox(ns4.DxMat.SDP_block[1].de_ele[1, 2], T(0); atol=eps(T)^0.5)
        end
    end

    # --- the symmetrize step must value-copy, not alias, the mirrored
    # transposed slots of a BigFloat block ---
    @testset "compute_dxmat! symmetrize value-copies mirrored BigFloat slots" begin
        T = BigFloat
        ns4 = SDPA.NewtonState{T}(1, [2])
        current_pt4 = SDPA.SDPASolution{T}(1, [2], T(1))
        work4 = SDPA.SDPAWorkVariables{T}([2])
        # DzMat = 0 → Δx = r_zinvMat - 0, then symmetrized.
        SDPA.setzero!(ns4.DzMat)
        SDPA.setzero!(ns4.r_zinvMat)
        # Asymmetric off-diagonals: the sym step averages (2+4)/2 = 3.
        ns4.r_zinvMat.SDP_block[1].de_ele[1, 1] = T(3)
        ns4.r_zinvMat.SDP_block[1].de_ele[1, 2] = T(2)
        ns4.r_zinvMat.SDP_block[1].de_ele[2, 1] = T(4)
        ns4.r_zinvMat.SDP_block[1].de_ele[2, 2] = T(5)
        SDPA.compute_dxmat!(ns4, current_pt4, work4)
        blk = ns4.DxMat.SDP_block[1].de_ele
        @test blk[1, 2] == 3
        @test blk[2, 1] == 3
        # Mirrored slots must be distinct BigFloat objects.
        @test blk[1, 2] !== blk[2, 1]
        # In-place mutation of one mirror leaves the other unchanged.
        SDPA.mp_add!(blk[1, 2], blk[1, 2], T(1))
        @test blk[1, 2] == 4
        @test blk[2, 1] == 3
    end

    # --- Rpotrf_with_adjust! = plain Rpotrf! + failure (no diagonal adjust) ---
    # C++ parity: the active path (sdpa_linear.cpp:352-361) is plain Rpotrf +
    # FAILURE; the diagonal-adjust algorithm is `#if 0` dead code. The adjust
    # loop read diagonals already overwritten by the first factorization.
    @testset "Rpotrf_with_adjust! performs no diagonal adjust (C++ parity)" begin
        T = BigFloat
        # Indefinite 3×3 whose second pivot is exactly -(2^-40) < 0. All
        # entries are exact powers of 2, so the pivot is exactly negative at
        # any BigFloat precision.
        A = T[1          T(2^-10)      T(2^-10);
              T(2^-10)   T(2^-20) - T(2^-40)   T(2^-10);
              T(2^-10)   T(2^-10)      1]
        # NOTE: `copy` on Matrix{BigFloat} is shallow (shared elements);
        # use `deepcopy` so the in-place factorization cannot mutate A.
        B = deepcopy(A)
        info = SDPA.Rpotrf_with_adjust!("U", 3, B; nb=64)
        @test info != 0

        # Clearly negative pivot (eigenvalues -1, 3) also fails.
        C = T[1 2; 2 1]
        info2 = SDPA.Rpotrf_with_adjust!("U", 2, C; nb=64)
        @test info2 != 0

        # Caller-level failure propagation: cholesky_schur! returns false.
        ns_bad = SDPA.NewtonState{T}(3, [2])
        SDPA._copyto_independent!(ns_bad.bmat.de_ele, A)
        ok_bad = SDPA.cholesky_schur!(ns_bad)
        @test ok_bad == false
        @test ns_bad.cholesky_factorization === nothing

        # No-regression: PD 3×3 → info == 0 and U'U ≈ A.
        A2 = T[2 0.5 0.5; 0.5 2 0.5; 0.5 0.5 2]
        B2 = deepcopy(A2)
        info3 = SDPA.Rpotrf_with_adjust!("U", 3, B2; nb=64)
        @test info3 == 0
        @test triu(B2)' * triu(B2) ≈ A2

        ns_good = SDPA.NewtonState{T}(3, [2])
        SDPA._copyto_independent!(ns_good.bmat.de_ele, A2)
        @test SDPA.cholesky_schur!(ns_good) == true
    end
end
