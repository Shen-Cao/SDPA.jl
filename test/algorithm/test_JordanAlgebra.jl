@testset "JordanAlgebra" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            # Build two simple 2-block DLS instances for testing:
            #   SDP block: 2x2 symmetric matrix
            #   LP block: 1 scalar
            blockStruct = [2, -1]

            A = SDPA.SDPADenseLinearSpace{T}([2, -1])
            B = SDPA.SDPADenseLinearSpace{T}([2, -1])
            R = SDPA.SDPADenseLinearSpace{T}([2, -1])
            W = SDPA.SDPADenseLinearSpace{T}([2, -1])

            # A.SDP_block[1] = [[1, 2], [2, 3]], A.LP_block[1] = 5
            A.SDP_block[1][1, 1] = T(1); A.SDP_block[1][1, 2] = T(2)
            A.SDP_block[1][2, 1] = T(2); A.SDP_block[1][2, 2] = T(3)
            A.LP_block[1] = T(5)

            # B.SDP_block[1] = [[4, 1], [1, 6]], B.LP_block[1] = 2
            B.SDP_block[1][1, 1] = T(4); B.SDP_block[1][1, 2] = T(1)
            B.SDP_block[1][2, 1] = T(1); B.SDP_block[1][2, 2] = T(6)
            B.LP_block[1] = T(2)

            # --- multiply! ---
            # A*B = [[1*4+2*1, 1*1+2*6], [2*4+3*1, 2*1+3*6]] = [[6, 13], [11, 20]]
            SDPA.multiply!(R, A, B)
            @test R.SDP_block[1][1, 1] == T(6)
            @test R.SDP_block[1][1, 2] == T(13)
            @test R.SDP_block[1][2, 1] == T(11)
            @test R.SDP_block[1][2, 2] == T(20)
            @test R.LP_block[1] == T(10)  # 5*2

            # --- multiply! with scalar ---
            SDPA.multiply!(R, A, B, T(2))
            @test R.SDP_block[1][1, 1] == T(12)
            @test R.LP_block[1] == T(20)

            # --- ns_jordan_triple_product! ---
            # A*B*C with C=A → A*B*A
            mat_c3 = SDPA.SDPADenseLinearSpace{T}([2, -1])
            mat_c3.SDP_block[1][1, 1] = T(1); mat_c3.SDP_block[1][1, 2] = T(2)
            mat_c3.SDP_block[1][2, 1] = T(2); mat_c3.SDP_block[1][2, 2] = T(3)
            mat_c3.LP_block[1] = T(5)  # same as A
            SDPA.ns_jordan_triple_product!(R, A, B, mat_c3, W)
            @test R.SDP_block[1][1, 1] == T(32)
            @test R.SDP_block[1][2, 2] == T(82)

            # --- get_inv_chol! ---
            # Build a positive-definite matrix: M = [[2,1],[1,2]]
            # Cholesky: L = [[sqrt(2), 0], [1/sqrt(2), sqrt(3/2)]]
            M = SDPA.SDPADenseLinearSpace{T}([2, -1])
            M.SDP_block[1][1, 1] = T(2); M.SDP_block[1][1, 2] = T(1)
            M.SDP_block[1][2, 1] = T(1); M.SDP_block[1][2, 2] = T(2)
            M.LP_block[1] = T(4)
            inv_chol = SDPA.SDPADenseLinearSpace{T}([2, -1])
            work_mat = SDPA.SDPADenseLinearSpace{T}([2, -1])
            ok = SDPA.get_inv_chol!(inv_chol, M, work_mat)
            @test ok == true
            # L*L' should reconstruct M
            L_inv = inv_chol.SDP_block[1].de_ele
            # inv_chol is L⁻¹. Check that L⁻¹ is lower-triangular.
            @test L_inv[1, 2] == zero(T)
            # L⁻¹ * M * (L⁻¹)' should = I (since L⁻¹ L L' (L⁻¹)' = I).
            reconstructed = L_inv * M.SDP_block[1].de_ele * L_inv'
            @test reconstructed[1, 1] ≈ one(T)
            @test reconstructed[2, 2] ≈ one(T)
            @test isapprox(reconstructed[1, 2], zero(T); atol=eps(T)^0.5)
            # LP block: 1/sqrt(4) = 0.5
            @test inv_chol.LP_block[1] == T(1) / T(2)

            # --- get_inv_chol_and_inv! ---
            inverse_mat = SDPA.SDPADenseLinearSpace{T}([2, -1])
            ok = SDPA.get_inv_chol_and_inv!(inv_chol, inverse_mat, M, work_mat)
            @test ok == true
            # M⁻¹ = (1/3)*[[2,-1],[-1,2]]
            inv_blk = inverse_mat.SDP_block[1].de_ele
            @test inv_blk[1, 1] ≈ T(2) / T(3)
            @test inv_blk[2, 2] ≈ T(2) / T(3)
            @test inv_blk[1, 2] ≈ -T(1) / T(3)
            # LP inverse: 1/4
            @test inverse_mat.LP_block[1] == T(1) / T(4)

            # --- Cholesky failure on indefinite matrix ---
            bad = SDPA.SDPADenseLinearSpace{T}([2, -1])
            bad.SDP_block[1][1, 1] = T(0); bad.SDP_block[1][1, 2] = T(1)
            bad.SDP_block[1][2, 1] = T(1); bad.SDP_block[1][2, 2] = T(0)
            bad.LP_block[1] = T(4)
            ok_bad = SDPA.get_inv_chol!(inv_chol, bad, work_mat)
            @test ok_bad == false

            # --- LP block exactly zero: guarded, no silent Inf ---
            zero_lp = SDPA.SDPADenseLinearSpace{T}([2, -1])
            zero_lp.SDP_block[1][1, 1] = T(2); zero_lp.SDP_block[1][1, 2] = T(1)
            zero_lp.SDP_block[1][2, 1] = T(1); zero_lp.SDP_block[1][2, 2] = T(2)
            zero_lp.LP_block[1] = T(0)  # exactly zero LP entry
            # get_inv_chol!: returns false (no silent 1/sqrt(0) = Inf).
            ok_zero = SDPA.get_inv_chol!(inv_chol, zero_lp, work_mat)
            @test ok_zero == false
            # get_inv_chol_and_inv!: returns false (no silent 1/0 = Inf).
            ok_zero2 = SDPA.get_inv_chol_and_inv!(inv_chol, inverse_mat, zero_lp, work_mat)
            @test ok_zero2 == false
        end
    end
end
