@testset "a finite positive trial with unrepresentable inverse is not accepted" begin
    T = Float64
    lo = nextfloat(zero(T))
    z0 = 2 * floatmin(T)
    for sdp in (false, true)
        blocks = sdp ? [2] : [-1]
        point = SDPA.SDPASolution{T}(1, blocks, one(T))
        newton = SDPA.NewtonState{T}(1, blocks)
        work = SDPA.SDPAWorkVariables{T}(blocks)
        entry(M) = sdp ? M.SDP_block[1].de_ele[1, 1] : M.LP_block[1]
        function setentry!(M, val)
            if sdp
                M.SDP_block[1].de_ele[1, 1] = val
            else
                M.LP_block[1] = val
            end
        end
        setentry!(point.zMat, z0)
        @test SDPA.compute_inverse!(point, work)
        old_inverse = entry(point.invzMat)
        @test isfinite(old_inverse)

        # All trial coordinates are finite and positive, but 1/lo overflows.
        # A tiny invalid proposal cannot be shortened further; restore every
        # cache of the old point, including caches touched before Z failed.
        tiny = T(2)^(-14)
        setentry!(newton.DzMat, (lo - z0) / tiny)
        @test z0 + tiny * entry(newton.DzMat) == lo
        a = SDPA.StepLength{T}(tiny, tiny)
        @test SDPA._update_iterate!(point, a, newton, work) === SDPA._NONFINITE_STEP
        @test entry(point.zMat) == z0 && entry(point.invzMat) == old_inverse
        @test isfinite(entry(point.invCholeskyZ))

        # The same boundary point from a full step is recoverable by halving.
        setentry!(newton.DzMat, lo - z0)
        a = SDPA.StepLength{T}(one(T), one(T))
        @test SDPA.update!(point, a, newton, work)
        @test 0 < a.dual < 1
        @test entry(point.zMat) == z0 + a.dual*(lo - z0)
        @test isfinite(entry(point.invzMat))
        @test entry(point.invzMat) ≈ inv(entry(point.zMat))

        # Warm-start inverse construction must also reject the invalid cache.
        setentry!(point.zMat, lo)
        @test !SDPA.compute_inverse!(point, work)
    end
end

@testset "inverse cleanup preserves representable extreme entries" begin
    for threaded in (false, true)
        point = SDPA.SDPASolution{Float64}(0, [2], 1)
        point.zMat.SDP_block[1].de_ele[1, 1] = 1e-308
        work = SDPA.SDPAWorkVariables{Float64}([2])
        kernel = threaded ? SDPA.threaded_get_inv_chol_and_inv! : SDPA.get_inv_chol_and_inv!
        @test kernel(point.invCholeskyZ, point.invzMat, point.zMat, work.DLS2)
        @test point.invzMat.SDP_block[1].de_ele[1, 1] ≈ 1e308 rtol=5eps(Float64)
    end
    # Keep the usual operation order at the other end of the exponent
    # range: unconditionally halving operands would erase a subnormal.
    tiny = nextfloat(0.0)
    A = [1e308 1.4e308; 1.2e308 9e307]
    SDPA._symmetrize_inverse!(A, nothing)
    @test A[1,1] == 1e308 && A[2,2] == 9e307
    @test A[1,2] == A[2,1] == (1.4e308/2 + 1.2e308/2)
    A[1,2] = A[2,1] = tiny
    SDPA._symmetrize_inverse!(A, nothing)
    @test A[1,2] == A[2,1] == tiny
end
