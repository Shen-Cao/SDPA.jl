@testset "DirectionParameter" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            params = SDPA.Parameters{T}()
            beta = SDPA.DirectionParameter{T}(params.betaStar)
            @test beta.value == params.betaStar

            # Predictor: pdFEAS -> value = 0
            SDPA.mehrotra_predictor!(beta, SDPA.pdFEAS, SDPA.Switch(SDPA.REDUCTION_ON), params)
            @test beta.value == zero(T)

            # Predictor: noINFO + reduction ON -> value = betaBar
            SDPA.mehrotra_predictor!(beta, SDPA.noINFO, SDPA.Switch(SDPA.REDUCTION_ON), params)
            @test beta.value == params.betaBar

            # Predictor: noINFO + reduction OFF -> value = 2.0
            SDPA.mehrotra_predictor!(beta, SDPA.noINFO, SDPA.Switch(SDPA.REDUCTION_OFF), params)
            @test beta.value == T(2)

            # Corrector: build a tiny solution + directions, verify clamping.
            # Set up current_pt with lambda=1, so X=Z=I, mu.current = 1.
            sol = SDPA.SDPASolution{T}(2, [2], T(1))
            mu = SDPA.AverageComplementarity{T}(T(1))
            mu.current = T(1)  # nDim * mu/nDim

            # Δx = 0, Δz = 0: mu_target = mu.current = 1, value = 1, no squaring
            dx_mat = SDPA.SDPADenseLinearSpace{T}([2])
            dz_mat = SDPA.SDPADenseLinearSpace{T}([2])
            SDPA.mehrotra_corrector!(beta, SDPA.pdFEAS, T(0), T(0), sol, dx_mat, dz_mat, mu, params)
            # value = 1/1 = 1, no squaring (1 not < 1), pdFEAS clamps to [betaStar, 1]
            # so value stays 1.
            @test beta.value == one(T)

            # Δx = 0, Δz = 0, with very small alpha so mu_target ~ mu.current.
            # But verify pdFEAS clamps value below betaStar up to betaStar.
            SDPA.mehrotra_corrector!(beta, SDPA.pdFEAS, T(0.5), T(0.5), sol, dx_mat, dz_mat, mu, params)
            @test beta.value >= params.betaStar
            @test beta.value <= one(T)

            # Non-pdFEAS phase: lower clamp is betaBar instead of betaStar.
            SDPA.mehrotra_corrector!(beta, SDPA.noINFO, T(0.5), T(0.5), sol, dx_mat, dz_mat, mu, params)
            @test beta.value >= params.betaBar

            # Corrector, mu.current == 0 (degenerate): value forced to 1,
            # never NaN/Inf.
            mu.current = zero(T)
            SDPA.mehrotra_corrector!(beta, SDPA.pdFEAS, T(0), T(0), sol, dx_mat, dz_mat, mu, params)
            @test beta.value == one(T)
            @test !isnan(beta.value)
        end
    end
end
