@testset "constraint-free optimum returns its dual slack" begin
    for T in (Float64, BigFloat), mixed in (false, true)
        blocks = mixed ? [-3, 2, 1] : [-3, 1]
        C = SDPA.SDPASparseLinearSpace{T}(Int(mixed), 4)
        if mixed
            C.SDP_sp_index[1] = 2
            C.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
            SDPA.setelement_sdp!(C, 1, 1, 1, T(3))
        end
        SDPA.setelement_lp!(C, 2, T(5))
        SDPA.setelement_lp!(C, 4, T(7))
        problem = SDPA.SDPAProblem{T}(0, length(blocks), blocks,
            SDPA.SDPAVector{T}(0), C, SDPA.SDPASparseLinearSpace{T}[])

        log = SDPA.solve!(problem, SDPA.Parameters{T}(); verbose=false,
                          reverse_primal_dual=false)
        @test log.terminal_phase == SDPA.pdOPT && log.iterations == 0
        @test all(iszero, log.xMat.LP_block)
        @test log.zMat.LP_block == T[0, 5, 0, 7]
        @test log.final_dual_res == maximum(abs, T[0, 5, 0, 7] - log.zMat.LP_block)
        if mixed
            @test log.zMat.SDP_block[1].de_ele == T[3 0; 0 0]
        end
        if T === BigFloat
            # The result is independently owned, including its mutable MPFR slots.
            SDPA.mp_zero!(log.zMat.LP_block[2])
            @test SDPA.get_lp_value(C, 2) == 5
        end
    end
end
