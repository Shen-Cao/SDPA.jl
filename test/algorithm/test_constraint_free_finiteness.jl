@testset "constraint-free nonfinite objectives cannot report optimality" begin
    for T in (Float64, BigFloat), bad in (NaN, Inf, -Inf), reverse in (false, true)
        C = SDPA.SDPASparseLinearSpace{T}(0, 1)
        SDPA.setelement_lp!(C, 1, T(bad))
        p = SDPA.SDPAProblem{T}(0, 1, [-1], SDPA.SDPAVector{T}(0), C,
                                 SDPA.SDPASparseLinearSpace{T}[])
        @test_throws ArgumentError SDPA.solve!(p, SDPA.Parameters{T}();
                                               verbose=false, reverse_primal_dual=reverse)
    end
    for T in (Float64, BigFloat), storage in (:lp, :sparse, :dense), bad in (NaN, Inf, -Inf)
        # Mixed blocks also cover an invalid coefficient encountered after
        # a negative direction: invalid input is not a solver exit status.
        C = SDPA.SDPASparseLinearSpace{T}(1, 1)
        C.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 2)
        SDPA.setelement_sdp!(C, 1, 1, 1, storage === :lp ? -one(T) : T(bad))
        SDPA.setelement_lp!(C, 1, storage === :lp ? T(bad) : one(T))
        storage === :dense && SDPA.change_to_dense!(C; force=true)
        p = SDPA.SDPAProblem{T}(0, 2, [2, -1], SDPA.SDPAVector{T}(0), C,
                                 SDPA.SDPASparseLinearSpace{T}[])
        for reverse in (false, true)
            err = try
                SDPA.solve!(p, SDPA.Parameters{T}(maxIteration=1);
                            verbose=false, reverse_primal_dual=reverse)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError && occursin("constraint-free objective", sprint(showerror, err))
        end
    end
end

@testset "constraint-free residual uses the actual returned precision" begin
    setprecision(BigFloat, 256) do
        T = BigFloat
        delta = T(2)^(-100)
        for storage in (:lp, :sparse, :dense)
            C = SDPA.SDPASparseLinearSpace{T}(1, 1)
            C.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
            SDPA.setelement_sdp!(C, 1, 1, 1, storage === :lp ? one(T) : one(T) + delta)
            SDPA.setelement_lp!(C, 1, storage === :lp ? one(T) + delta : one(T))
            storage === :dense && SDPA.change_to_dense!(C; force=true)
            p = SDPA.SDPAProblem{T}(0, 2, [2, -1], SDPA.SDPAVector{T}(0), C,
                                     SDPA.SDPASparseLinearSpace{T}[])
            for tolerance in (2delta, delta/2)
                log = SDPA.solve!(p, SDPA.Parameters{T}(precision=64, epsilonDash=tolerance);
                                  verbose=false, reverse_primal_dual=false)
                @test log.zMat.LP_block[1] == log.zMat.SDP_block[1].de_ele[1,1] == one(T)
                @test log.final_dual_res == delta
                @test log.terminal_phase === (tolerance >= delta ? SDPA.pdOPT : SDPA.pFEAS)
                @test log.exit_reason === (tolerance >= delta ? SDPA.OPTIMAL_TERMINAL : SDPA.CANNOT_MOVE)
            end
        end
    end
end
