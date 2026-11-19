@testset "StepLength" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            # --- StepLength struct ---
            alpha = SDPA.StepLength{T}()
            @test alpha.primal == zero(T)
            @test alpha.dual == zero(T)

            # --- compute_step_length! on identity matrices ---
            # X = I, Δx = -0.5*I → L_x·Δx·L_x' = -0.5*I → λ_min = -0.5
            # → primal = -1/(-0.5) = 2.0
            blockStruct = [2]
            current_pt = SDPA.SDPASolution{T}(1, blockStruct, T(1))
            work = SDPA.SDPAWorkVariables{T}(blockStruct)
            newton = SDPA.NewtonState{T}(1, blockStruct)

            # Set Δx = -0.5*I (negative definite → positive step)
            newton.DxMat.SDP_block[1].de_ele[1, 1] = T(-1) / T(2)
            newton.DxMat.SDP_block[1].de_ele[2, 2] = T(-1) / T(2)
            # Set Δz = -0.5*I similarly
            newton.DzMat.SDP_block[1].de_ele[1, 1] = T(-1) / T(2)
            newton.DzMat.SDP_block[1].de_ele[2, 2] = T(-1) / T(2)

            SDPA.compute_step_length!(alpha, current_pt, newton, work)
            # With L_x = I and Δx = -0.5*I: λ_min(L_x·Δx·L_x') = -0.5
            # primal = -1/(-0.5) = 2.0
            @test alpha.primal ≈ T(2) rtol=T(1e-3)
            @test alpha.dual ≈ T(2) rtol=T(1e-3)

            # --- cap at alphaBD = 100 ---
            # If λ_min is tiny (near zero), step should be capped at 100.
            # Set Δx = -1e-5 * I → λ_min ≈ -1e-5 → step huge → capped.
            newton.DxMat.SDP_block[1].de_ele[1, 1] = T(-1) / T(1e5)
            newton.DxMat.SDP_block[1].de_ele[2, 2] = T(-1) / T(1e5)
            newton.DzMat.SDP_block[1].de_ele[1, 1] = T(-1) / T(1e5)
            newton.DzMat.SDP_block[1].de_ele[2, 2] = T(-1) / T(1e5)
            SDPA.compute_step_length!(alpha, current_pt, newton, work)
            @test alpha.primal <= T(100) + T(1e-3)
            @test alpha.dual <= T(100) + T(1e-3)
        end
    end

    # --- NaN min-eigen estimate must be a LOUD error ---
    # A NaN block in the Newton direction makes λ_min(L·Δ·L') NaN. The
    # NaN must fail loudly (throw) rather than silently awarding the full
    # alphaBD = 100 step as if the direction were genuinely safe.
    @testset "compute_step_length! throws on NaN min-eigenvalue estimate" begin
        for T in (Float64, BigFloat)
            @testset "$T" begin
                blockStruct = [2]
                current_pt = SDPA.SDPASolution{T}(1, blockStruct, T(1))
                work = SDPA.SDPAWorkVariables{T}(blockStruct)
                newton = SDPA.NewtonState{T}(1, blockStruct)
                alpha = SDPA.StepLength{T}()

                # Primal: poison the whole 2×2 Dx block with NaN; keep the
                # dual direction sane (λ_min = -0.5) to isolate the guard.
                for i in 1:2, j in 1:2
                    newton.DxMat.SDP_block[1].de_ele[i, j] = T(NaN)
                end
                newton.DzMat.SDP_block[1].de_ele[1, 1] = T(-1) / T(2)
                newton.DzMat.SDP_block[1].de_ele[2, 2] = T(-1) / T(2)
                err = try
                    SDPA.compute_step_length!(alpha, current_pt, newton, work)
                    nothing
                catch e
                    e
                end
                @test err isa ErrorException
                @test occursin("min-eigenvalue estimate is NaN", err.msg)
                @test occursin("primal", err.msg)

                # Dual twin: sane primal direction, NaN-poisoned Dz block.
                # (Reset ALL Dx entries — the primal injection above left
                # NaN off-diagonals that would trip the primal guard.)
                newton.DxMat.SDP_block[1].de_ele[1, 1] = T(-1) / T(2)
                newton.DxMat.SDP_block[1].de_ele[1, 2] = T(0)
                newton.DxMat.SDP_block[1].de_ele[2, 1] = T(0)
                newton.DxMat.SDP_block[1].de_ele[2, 2] = T(-1) / T(2)
                for i in 1:2, j in 1:2
                    newton.DzMat.SDP_block[1].de_ele[i, j] = T(NaN)
                end
                err = try
                    SDPA.compute_step_length!(alpha, current_pt, newton, work)
                    nothing
                catch e
                    e
                end
                @test err isa ErrorException
                @test occursin("min-eigenvalue estimate is NaN", err.msg)
                @test occursin("dual", err.msg)

                # LP-tail twin: a NaN LP entry of the direction
                # DLS gives l²·x = NaN, which must hit the same loud
                # error as the SDP-block cases above (NaN comparisons
                # never silently award the full alphaBD step).
                blockStruct_lp = [2, -1]
                current_pt_lp = SDPA.SDPASolution{T}(1, blockStruct_lp, T(1))
                work_lp = SDPA.SDPAWorkVariables{T}(blockStruct_lp)
                newton_lp = SDPA.NewtonState{T}(1, blockStruct_lp)
                # Sane SDP direction (λ_min = -0.5); NaN only in the LP tail.
                newton_lp.DxMat.SDP_block[1].de_ele[1, 1] = T(-1) / T(2)
                newton_lp.DxMat.SDP_block[1].de_ele[1, 2] = T(0)
                newton_lp.DxMat.SDP_block[1].de_ele[2, 1] = T(0)
                newton_lp.DxMat.SDP_block[1].de_ele[2, 2] = T(-1) / T(2)
                newton_lp.DxMat.LP_block[1] = T(NaN)
                newton_lp.DzMat.SDP_block[1].de_ele[1, 1] = T(-1) / T(2)
                newton_lp.DzMat.SDP_block[1].de_ele[1, 2] = T(0)
                newton_lp.DzMat.SDP_block[1].de_ele[2, 1] = T(0)
                newton_lp.DzMat.SDP_block[1].de_ele[2, 2] = T(-1) / T(2)
                newton_lp.DzMat.LP_block[1] = T(-1) / T(2)
                err = try
                    SDPA.compute_step_length!(alpha, current_pt_lp, newton_lp, work_lp)
                    nothing
                catch e
                    e
                end
                @test err isa ErrorException
                @test occursin("min-eigenvalue estimate is NaN", err.msg)
                @test occursin("primal", err.msg)
            end
        end
    end

    # --- Lanczos sentinel regression at the step-length layer ---
    # A 21×21 direction whose Lanczos starting vector (ones) is an exact
    # eigenvector used to make min_eigen return the 1e50 sentinel; the
    # step length then read "no negative curvature" and awarded the full
    # alphaBD = 100 cap instead of 1/|λ_min| = 0.2.
    @testset "step length uses the true λmin on a Krylov-invariant start" begin
        for T in (Float64, BigFloat)
            n = 21
            blockStruct = [n]
            current_pt = SDPA.SDPASolution{T}(0, blockStruct, T(1))
            work = SDPA.SDPAWorkVariables{T}(blockStruct)
            newton = SDPA.NewtonState{T}(0, blockStruct)
            alpha = SDPA.StepLength{T}()
            u = T[sqrt(T(1 / n)) for _ in 1:n]
            for i in 1:n, j in 1:n
                newton.DxMat.SDP_block[1].de_ele[i, j] =
                    (i == j ? T(1) : T(0)) - T(6) * u[i] * u[j]
                newton.DzMat.SDP_block[1].de_ele[i, j] =
                    (i == j ? T(-1) / T(2) : T(0))
            end
            SDPA.compute_step_length!(alpha, current_pt, newton, work)
            @test alpha.primal ≈ T(1 // 5) rtol=T(1e-6)  # 1/|λ_min|, not 100
            @test alpha.dual ≈ T(2) rtol=T(1e-6)
        end
    end
end
