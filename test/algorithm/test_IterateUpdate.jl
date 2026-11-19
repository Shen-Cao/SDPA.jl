@testset "IterateUpdate" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            # --- update! with hand-crafted alpha and newton ---
            blockStruct = [2]
            current_pt = SDPA.SDPASolution{T}(1, blockStruct, T(1))
            work = SDPA.SDPAWorkVariables{T}(blockStruct)
            newton = SDPA.NewtonState{T}(1, blockStruct)

            # Set Δx = [[1, 0], [0, 1]], Δy = [2], Δz = [[3, 0], [0, 3]]
            newton.DxMat.SDP_block[1].de_ele[1, 1] = T(1)
            newton.DxMat.SDP_block[1].de_ele[2, 2] = T(1)
            newton.DyVec.ele[1] = T(2)
            newton.DzMat.SDP_block[1].de_ele[1, 1] = T(3)
            newton.DzMat.SDP_block[1].de_ele[2, 2] = T(3)

            alpha = SDPA.StepLength{T}(T(1), T(1))

            # After update: xMat = I + 1*I = 2*I, yVec = 0 + 1*2 = 2,
            # zMat = I + 1*3*I = 4*I
            ok = SDPA.update!(current_pt, alpha, newton, work)
            @test ok == true
            @test current_pt.xMat.SDP_block[1].de_ele[1, 1] == T(2)
            @test current_pt.xMat.SDP_block[1].de_ele[2, 2] == T(2)
            @test current_pt.yVec.ele[1] == T(2)
            @test current_pt.zMat.SDP_block[1].de_ele[1, 1] == T(4)
            @test current_pt.zMat.SDP_block[1].de_ele[2, 2] == T(4)

            # After update, inverse Cholesky should be recomputed.
            # X = 2*I → invCholeskyX = I/sqrt(2)
            @test current_pt.invCholeskyX.SDP_block[1].de_ele[1, 1] ≈ T(1) / sqrt(T(2))
            # Z = 4*I → invzMat = I/4
            @test current_pt.invzMat.SDP_block[1].de_ele[1, 1] ≈ T(1) / T(4)

            # --- "cannot move" safeguard ---
            # Both alpha < 1e-4 → returns false
            current_pt2 = SDPA.SDPASolution{T}(1, blockStruct, T(1))
            alpha_tiny = SDPA.StepLength{T}(T(1e-5), T(1e-5))
            ok2 = SDPA.update!(current_pt2, alpha_tiny, newton, work)
            @test ok2 == false

            # --- compute_inverse! standalone ---
            current_pt3 = SDPA.SDPASolution{T}(1, blockStruct, T(4))
            ok3 = SDPA.compute_inverse!(current_pt3, work)
            @test ok3 == true
            # X = 4*I → invCholeskyX = I/2 (since L = 2*I, L⁻¹ = 0.5*I)
            @test current_pt3.invCholeskyX.SDP_block[1].de_ele[1, 1] ≈ T(1) / T(2)
        end
    end
end
