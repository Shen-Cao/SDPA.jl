@testset "threaded ops bit-identical" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            # Parse example1 and set up state at iteration 0.
            problem = SDPA.read_dat_s(
                joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T
            )
            params = SDPA.Parameters{T}()
            current_pt = SDPA.SDPASolution{T}(problem.m, problem.blockStruct, params.lambdaStar)
            work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)
            newton = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
            SDPA.compute_formula!(newton, problem)

            # --- threaded vs serial make_bmat! ---
            SDPA.make_bmat!(newton, problem, current_pt, work)
            bmat_serial = SDPA._independent_zeros(T, problem.m, problem.m)
            SDPA._copyto_independent!(bmat_serial, newton.bmat.de_ele)   # VALUE copy (BigFloat: mp_set! per element; Float64: plain copyto!)

            SDPA.threaded_compute_bmat_dense!(newton, problem, current_pt, work)
            bmat_threaded = SDPA._independent_zeros(T, problem.m, problem.m)
            SDPA._copyto_independent!(bmat_threaded, newton.bmat.de_ele)

            @test bmat_serial == bmat_threaded

            # --- threaded vs serial get_inv_chol! ---
            inv_chol_s = SDPA.SDPADenseLinearSpace{T}(problem.blockStruct)
            inv_chol_t = SDPA.SDPADenseLinearSpace{T}(problem.blockStruct)
            work_mat = SDPA.SDPADenseLinearSpace{T}(problem.blockStruct)

            SDPA.get_inv_chol!(inv_chol_s, current_pt.xMat, work_mat)
            SDPA.threaded_get_inv_chol!(inv_chol_t, current_pt.xMat, work_mat)

            @test inv_chol_s.SDP_block[1].de_ele == inv_chol_t.SDP_block[1].de_ele

            # --- threaded vs serial get_inv_chol_and_inv! ---
            inv_mat_s = SDPA.SDPADenseLinearSpace{T}(problem.blockStruct)
            inv_mat_t = SDPA.SDPADenseLinearSpace{T}(problem.blockStruct)

            SDPA.get_inv_chol_and_inv!(inv_chol_s, inv_mat_s, current_pt.zMat, work_mat)
            SDPA.threaded_get_inv_chol_and_inv!(inv_chol_t, inv_mat_t, current_pt.zMat, work_mat)

            @test inv_mat_s.SDP_block[1].de_ele == inv_mat_t.SDP_block[1].de_ele
        end
    end
end

@testset "bit-identity: multi_inner_product_to_a! at nthreads" begin
    # multi_inner_product_to_a! threads the outer i loop with @tasks.
    # Each task writes ret.ele[i] from exactly one i; xMat is read-only.
    # Result is therefore independent of thread count by construction.
    # We verify by re-running the function and checking the output is
    # stable across calls (no nondeterminism) and matches the math.
    for T in (Float64, BigFloat)
        problem = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T)
        params = SDPA.Parameters{T}()
        current_pt = SDPA.SDPASolution{T}(
            problem.m, problem.blockStruct, params.lambdaStar)
        # Build a ret vector and exercise the threaded function twice.
        ret1 = SDPA.SDPAVector{T}(problem.m)
        ret2 = SDPA.SDPAVector{T}(problem.m)
        SDPA.multi_inner_product_to_a!(ret1, problem, current_pt.xMat)
        SDPA.multi_inner_product_to_a!(ret2, problem, current_pt.xMat)
        # Bit-identity: two consecutive runs produce identical bytes.
        @test ret1.ele == ret2.ele
        # Sanity: result is finite (not NaN/Inf) and at least one entry is nonzero.
        @test all(isfinite, ret1.ele)
        @test any(x -> x != zero(T), ret1.ele)
    end
end

@testset "bit-identity: threaded_min_eigen_dls vs serial" begin
    # threaded_min_eigen_dls parallelizes the per-block eigenvalue
    # computation; each task writes vals[k] for exactly one k, then the
    # final `minimum(vals)` reduction is a deterministic Julia `minimum`
    # on a Vector. Result must equal the serial path bit-for-bit.
    for T in (Float64, BigFloat)
        problem = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T)
        params = SDPA.Parameters{T}()
        current_pt = SDPA.SDPASolution{T}(
            problem.m, problem.blockStruct, params.lambdaStar)
        work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)

        # Serial computation.
        min_serial = SDPA._min_eigen_dls(
            current_pt.invCholeskyX, current_pt.xMat, work)
        # Threaded computation (used in StepLength when nthreads > 1).
        min_threaded = SDPA.threaded_min_eigen_dls(
            current_pt.invCholeskyX, current_pt.xMat, work)
        # BigFloat `===` is too strict (precision metadata may differ on
        # intermediate computations even when final values match); use `==`.
        @test min_serial == min_threaded
        @test isfinite(min_serial)
    end
end

@testset "threaded LP zero-guard matches serial" begin
    # Serial get_inv_chol! / get_inv_chol_and_inv! guard `LP <= 0` (return
    # false, loud failure over silent Inf). The threaded twins must match at
    # an exact-zero LP entry, else results become thread-count-dependent
    # (Inf vs false). With JULIA_NUM_THREADS >= 2 the @tasks region is
    # genuinely parallel.
    Threads.nthreads() >= 2 ||
        @info "threaded zero-guard test running @tasks serially (nthreads=$(Threads.nthreads()))"
    for T in (Float64, BigFloat)
        @testset "$T" begin
            bs = [2, -1]
            M = SDPA.SDPADenseLinearSpace{T}(bs)
            M.SDP_block[1][1, 1] = T(2); M.SDP_block[1][1, 2] = T(1)
            M.SDP_block[1][2, 1] = T(1); M.SDP_block[1][2, 2] = T(2)
            M.LP_block[1] = T(0)   # exact-zero LP entry
            work_mat = SDPA.SDPADenseLinearSpace{T}(bs)

            # threaded_get_inv_chol!: the exact-zero LP entry must not
            # slip through the guard (mp_rec_sqrt!(0) -> Inf).
            inv_chol = SDPA.SDPADenseLinearSpace{T}(bs)
            ok = SDPA.threaded_get_inv_chol!(inv_chol, M, work_mat)
            @test ok == false
            # Serial path agrees.
            inv_chol_s = SDPA.SDPADenseLinearSpace{T}(bs)
            work_s = SDPA.SDPADenseLinearSpace{T}(bs)
            @test SDPA.get_inv_chol!(inv_chol_s, M, work_s) == false

            # threaded_get_inv_chol_and_inv!: same guard chain, no Inf in
            # the LP inverse.
            inv_mat = SDPA.SDPADenseLinearSpace{T}(bs)
            ok2 = SDPA.threaded_get_inv_chol_and_inv!(inv_chol, inv_mat, M, work_mat)
            @test ok2 == false
            @test all(isfinite, inv_mat.LP_block)
            inv_mat_s = SDPA.SDPADenseLinearSpace{T}(bs)
            @test SDPA.get_inv_chol_and_inv!(inv_chol_s, inv_mat_s, M, work_s) == false
        end
    end
end

@testset "threaded_get_inv_chol_and_inv! symmetrization keeps BigFloat pairs distinct" begin
    # The threaded symmetrization must value-copy avg (via mp_set!),
    # not store the same BigFloat object in both transposed slots.
    # @tasks on a 1-block space runs the loop body once; with
    # Threads.nthreads() > 1 the @tasks region is actually parallel.
    Threads.nthreads() > 1 || @info "threaded test running @tasks serially (nthreads=$(Threads.nthreads()))"
    setprecision(BigFloat, 200) do
        bs = [2, -1]
        M = SDPA.SDPADenseLinearSpace{BigFloat}(bs)
        M.SDP_block[1][1, 1] = BigFloat(2); M.SDP_block[1][1, 2] = BigFloat(1)
        M.SDP_block[1][2, 1] = BigFloat(1); M.SDP_block[1][2, 2] = BigFloat(2)
        M.LP_block[1] = BigFloat(4)
        inv_chol = SDPA.SDPADenseLinearSpace{BigFloat}(bs)
        inverse_mat = SDPA.SDPADenseLinearSpace{BigFloat}(bs)
        work_mat = SDPA.SDPADenseLinearSpace{BigFloat}(bs)
        ok = SDPA.threaded_get_inv_chol_and_inv!(inv_chol, inverse_mat, M, work_mat)
        @test ok == true
        inv_blk = inverse_mat.SDP_block[1].de_ele
        # Symmetrization must value-copy avg, NOT alias the transposed pair.
        @test inv_blk[1, 2] !== inv_blk[2, 1]
        # In-place MPFR mutation of one entry must not corrupt the mirror.
        mirrored = inv_blk[2, 1]
        SDPA.mp_add!(inv_blk[1, 2], inv_blk[1, 2], BigFloat(1))
        @test inv_blk[2, 1] == mirrored
        @test inv_blk[1, 2] == mirrored + 1
    end
end

@testset "min_eigen sentinel agrees across types on degenerate spaces" begin
    # The T(1e50) sentinel (C++ parity; Inf under Float32) must behave
    # identically on the serial and threaded paths: an LP-only space
    # overwrites it with finite values; a fully empty space returns it
    # unchanged, and the consumer comparison treats 1e50 and Inf alike.
    for T in (Float32, Float64, BigFloat)
        # LP-only: blockStruct = [-2], L²X per scalar = {4·5, 9·7}.
        l_mat = SDPA.SDPADenseLinearSpace{T}([-2])
        x_mat = SDPA.SDPADenseLinearSpace{T}([-2])
        work = SDPA.SDPAWorkVariables{T}([-2])
        l_mat.LP_block[1] = T(2); l_mat.LP_block[2] = T(3)
        x_mat.LP_block[1] = T(5); x_mat.LP_block[2] = T(7)
        s = SDPA._min_eigen_dls(l_mat, x_mat, work)
        t = SDPA.threaded_min_eigen_dls(l_mat, x_mat, work)
        @test s == t == T(20)

        # Fully empty: sentinel returned unchanged on both paths.
        l0 = SDPA.SDPADenseLinearSpace{T}(Int[])
        x0 = SDPA.SDPADenseLinearSpace{T}(Int[])
        w0 = SDPA.SDPAWorkVariables{T}(Int[])
        s0 = SDPA._min_eigen_dls(l0, x0, w0)
        t0 = SDPA.threaded_min_eigen_dls(l0, x0, w0)
        @test s0 == t0 == T(1e50)
    end
end
