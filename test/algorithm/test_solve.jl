@testset "solve!" begin
    # The BLAS-policy testsets below set_num_threads(2), and the
    # concurrency testset's single-core branch sets (1); save the
    # process-global count on entry and restore it after the last inner
    # testset so later includes in runtests.jl see the pre-test value.
    _blas_threads_on_entry = BLAS.get_num_threads()

    @testset "example1 e2e @ Float64" begin
        problem = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), Float64
        )
        params = SDPA.Parameters{Float64}()
        log = SDPA.solve!(problem, params; verbose=false, io=devnull)

        @test log.terminal_phase == SDPA.pdOPT
        @test log.iterations <= 200
        @test log.final_obj_primal ≈ Float64(-4.19e1) rtol=1e-7
        @test log.final_obj_dual   ≈ Float64(-4.19e1) rtol=1e-7
        # Float64 defaults use the reachable 1e-7 tolerances (upstream
        # SDPA double presets). Assert the stopping criterion itself
        # (relative duality gap <= epsilonStar, with rounding slack) and a
        # mu bound consistent with it (observed ~1.9e-6).
        gap = abs(log.final_obj_primal - log.final_obj_dual)
        dominator = max(1.0, (abs(log.final_obj_primal) + abs(log.final_obj_dual)) / 2)
        @test gap / dominator <= 1.1e-7
        @test log.final_mu < 1e-5
    end

    @testset "example1 e2e @ BigFloat(256)" begin
        problem = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), BigFloat
        )
        params = SDPA.Parameters{BigFloat}(precision=256)
        log = SDPA.solve!(problem, params; verbose=false, io=devnull)

        @test log.terminal_phase == SDPA.pdOPT
        @test log.iterations <= 200
        # C++ reference was captured at 200-bit precision ≈ 60 digits.
        # rtol=1e-30 verifies our 256-bit solver achieves ~100 bits.
        @test isapprox(log.final_obj_primal, BigFloat("-4.19e+01"); rtol=BigFloat("1e-30"))
        @test isapprox(log.final_obj_dual,   BigFloat("-4.19e+01"); rtol=BigFloat("1e-30"))
        # mu at termination is ~2.2e-30 (matching C++ reference). The
        # convergence criterion is on the duality gap (PDgap/mean ≤
        # epsilonStar), not on mu directly, so mu can be slightly above
        # epsilonStar at termination.
        @test log.final_mu < BigFloat("1e-29")

        # Iteration count should match the C++ reference within ±5.
        # C++ reference takes 39 iterations (from test/reference/example1.json).
        expected_iter = 39
        @test abs(log.iterations - expected_iter) <= 5
    end

    @testset "example1 e2e @ BigFloat(512) precision scaling" begin
        setprecision(BigFloat, 512) do
            problem = SDPA.read_dat_s(
                joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), BigFloat
            )
            params = SDPA.Parameters{BigFloat}(precision=512, epsilonStar=BigFloat("1e-100"))
            log = SDPA.solve!(problem, params; verbose=false, io=devnull,
                              reverse_primal_dual=false)

            @test log.terminal_phase == SDPA.pdOPT
            # At 512 bits (~154 digits), objVal should be right to ~140 digits.
            @test isapprox(log.final_obj_primal, BigFloat("-4.19e+01"); rtol=BigFloat("1e-140"))
            @test log.final_mu < BigFloat("1e-99")
        end
    end

    @testset "SolutionLog field copies are deep and mutation-isolated" begin
        # minimal_sdp_lp: 1×2×2 SDP block + 1 LP scalar (blockStruct [2, -1]).
        problem = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "minimal_sdp_lp.dat-s"), BigFloat
        )
        params = SDPA.Parameters{BigFloat}()
        log = SDPA.solve!(problem, params; verbose=false, io=devnull)
        @test log.terminal_phase == SDPA.pdOPT
        @test log.xMat.SDP_nBlock >= 1
        @test !isempty(log.xMat.LP_block)

        # xMat: copy(log.xMat) must be a VALUE copy — independent BigFloat
        # elements, so mutating the copy leaves solver state unchanged.
        xc = copy(log.xMat)
        @test xc !== log.xMat
        @test xc.SDP_block[1] !== log.xMat.SDP_block[1]
        @test xc.SDP_block[1].de_ele !== log.xMat.SDP_block[1].de_ele
        @test xc.SDP_block[1].de_ele[1, 1] !== log.xMat.SDP_block[1].de_ele[1, 1]
        @test xc.LP_block[1] !== log.xMat.LP_block[1]
        orig_x11 = deepcopy(log.xMat.SDP_block[1].de_ele[1, 1])
        orig_lp = deepcopy(log.xMat.LP_block[1])
        SDPA.mp_add!(xc.SDP_block[1].de_ele[1, 1], xc.SDP_block[1].de_ele[1, 1], one(BigFloat))
        SDPA.mp_add!(xc.LP_block[1], xc.LP_block[1], one(BigFloat))
        @test log.xMat.SDP_block[1].de_ele[1, 1] == orig_x11  # state unchanged
        @test log.xMat.LP_block[1] == orig_lp

        # yVec.
        yc = copy(log.yVec)
        @test yc !== log.yVec
        @test yc.ele[1] !== log.yVec.ele[1]
        orig_y1 = deepcopy(log.yVec.ele[1])
        SDPA.mp_add!(yc.ele[1], yc.ele[1], one(BigFloat))
        @test log.yVec.ele[1] == orig_y1

        # zMat.
        zc = copy(log.zMat)
        @test zc !== log.zMat
        @test zc.SDP_block[1].de_ele[1, 1] !== log.zMat.SDP_block[1].de_ele[1, 1]
        @test zc.LP_block[1] !== log.zMat.LP_block[1]
        orig_z11 = deepcopy(log.zMat.SDP_block[1].de_ele[1, 1])
        orig_zlp = deepcopy(log.zMat.LP_block[1])
        SDPA.mp_add!(zc.SDP_block[1].de_ele[1, 1], zc.SDP_block[1].de_ele[1, 1], one(BigFloat))
        SDPA.mp_add!(zc.LP_block[1], zc.LP_block[1], one(BigFloat))
        @test log.zMat.SDP_block[1].de_ele[1, 1] == orig_z11
        @test log.zMat.LP_block[1] == orig_zlp
    end

    @testset "exception safety: BLAS policy restored and problem.C intact" begin
        example1 = joinpath(@__DIR__, "..", "fixtures", "example1.dat-s")

        # Record all nonzero entries of C (SDP blocks in either storage mode
        # plus LP entries) so we can assert exact restoration after an
        # exception.
        function _record_C_entries(c)
            entries = Float64[]
            for sdp_k in 1:c.SDP_sp_nBlock
                blk = c.SDP_sp_block[sdp_k]
                if blk.type === :sparse
                    for nz in 1:blk.nonZeroEffect
                        push!(entries, blk.sp_ele[nz])
                    end
                else
                    for k in eachindex(blk.de_ele)
                        push!(entries, blk.de_ele[k])
                    end
                end
            end
            append!(entries, c.LP_sp_val)
            return entries
        end

        @testset "BLAS threads restored on normal exit" begin
            BLAS.set_num_threads(2)
            problem = SDPA.read_dat_s(example1, Float64)
            params = SDPA.Parameters{Float64}()
            log = SDPA.solve!(problem, params; verbose=false, io=devnull)
            @test log.terminal_phase == SDPA.pdOPT
            @test BLAS.get_num_threads() == 2
        end

        @testset "BLAS threads restored after solve! throws" begin
            BLAS.set_num_threads(2)
            problem = SDPA.read_dat_s(example1, Float64)
            params = SDPA.Parameters{Float64}()
            @test_throws ErrorException SDPA.solve!(
                problem, params; verbose=false, io=devnull,
                _test_throw_after_blas=true,
            )
            @test BLAS.get_num_threads() == 2
        end

        @testset "problem.C never mutated under reverse_primal_dual" begin
            problem = SDPA.read_dat_s(example1, Float64)
            params = SDPA.Parameters{Float64}()
            original = _record_C_entries(problem.C)
            # Throw path: the internal negated copy must not leak into the
            # caller's problem.
            @test_throws ErrorException SDPA.solve!(
                problem, params; verbose=false, io=devnull,
                reverse_primal_dual=true, _test_throw_after_blas=true,
            )
            @test _record_C_entries(problem.C) == original
            # Success path: C is untouched even after a full REVERSE solve.
            log = SDPA.solve!(problem, params; verbose=false, io=devnull,
                              reverse_primal_dual=true)
            @test log.terminal_phase == SDPA.pdOPT
            @test _record_C_entries(problem.C) == original
        end

        @testset "verbose iteration table renders scientific notation" begin
            # The GMP-style "%Fe" specifier (C++ SDPA-GMP) renders in Julia
            # Printf as fixed-point plus a stray literal 'e' with no exponent
            # digits. The verbose console table must use proper "%e".
            problem = SDPA.read_dat_s(example1, Float64)
            params = SDPA.Parameters{Float64}(maxIteration=5)
            buf = IOBuffer()
            SDPA.solve!(problem, params; verbose=true, io=buf)
            s = String(take!(buf))
            # No bare 'e' (digit-e followed by whitespace/end-of-line).
            @test !occursin(r"[0-9]e(\s|$)", s)
            # Scientific exponents with 2-digit exponent present.
            @test occursin(r"[0-9]\.[0-9]e[+-][0-9]{2}", s)
        end
    end

    @testset "concurrent subprocess solves keep BLAS policies isolated" begin
        # The race must run in two concurrent Julia SUBPROCESSES (a
        # parent-process thread gate would degrade to two sequential
        # solves under a 1-thread CI), each forced to a fixed subprocess
        # thread count regardless of the parent test process's: one solves
        # example1 at BigFloat(200), the other at Float64. The two types
        # install DIFFERENT process-global BLAS policies (BigFloat -> 1
        # BLAS thread, Float64 -> Threads.nthreads(); see src/solve.jl
        # _solve_inner_locked!), which is exactly the state the
        # _BLAS_POLICY_LOCK set/restore section protects. A ready-file
        # handshake aligns the two solves so their wall-clock intervals
        # genuinely overlap, and each subprocess reports its PID,
        # start/end timestamps, and BLAS thread counts before/after/
        # sampled-during the solve.
        # Subprocess thread count: >= 2 for genuine concurrency, capped at
        # 4 to avoid oversubscribing small machines, derived from the
        # machine rather than hardcoded.
        nt_sub = clamp(Sys.CPU_THREADS, 2, 4)
        example1 = joinpath(@__DIR__, "..", "fixtures", "example1.dat-s")
        proj_root = dirname(dirname(@__DIR__))
        julia_bin = joinpath(Sys.BINDIR, "julia")

        if Sys.CPU_THREADS == 1
            # Genuinely single-core machine (NOT Threads.nthreads() == 1,
            # which is the CI default this test defends against): two
            # overlapping solves cannot add parallelism here, so keep the
            # sequential assertions — loudly.
            @info "single-core machine — concurrency gate downgraded to sequential"
            BLAS.set_num_threads(1)
            pb = SDPA.read_dat_s(example1, BigFloat)
            pf = SDPA.read_dat_s(example1, Float64)
            params_b = SDPA.Parameters{BigFloat}(precision=200)
            params_f = SDPA.Parameters{Float64}()
            log_b = SDPA.solve!(
                pb, params_b; verbose=false, io=devnull, reverse_primal_dual=false)
            log_f = SDPA.solve!(
                pf, params_f; verbose=false, io=devnull, reverse_primal_dual=false)
            @test log_b.terminal_phase == SDPA.pdOPT
            @test log_f.terminal_phase == SDPA.pdOPT
            @test isapprox(log_b.final_obj_primal, BigFloat("-4.19e+01"); rtol=BigFloat("1e-30"))
            @test log_f.final_obj_primal ≈ -4.19e1 rtol=1e-7
            @test BLAS.get_num_threads() == 1
        else
            mktempdir() do tmpdir
                out_b = joinpath(tmpdir, "bigfloat.out")
                out_f = joinpath(tmpdir, "float64.out")
                ready_b = joinpath(tmpdir, "bigfloat.ready")
                ready_f = joinpath(tmpdir, "float64.ready")

                # Subprocess script (type_name: "BigFloat" or "Float64"):
                # load + parse the fixture first, handshake on the
                # sibling's ready file so both solves START together, then
                # solve while a sampler task records the live BLAS thread
                # count. Machine-readable output lines: PID / TSTART /
                # TEND / RESULT key=value.
                function _concurrent_script(type_name, ready, sibling)
                    params_ctor = type_name == "BigFloat" ?
                        "SDPA.Parameters{BigFloat}(precision=200)" :
                        "SDPA.Parameters{Float64}()"
                    policy_expr = type_name == "BigFloat" ? "1" :
                        "Threads.nthreads()"
                    return """
                    using SDPA, LinearAlgebra
                    const BLAS = LinearAlgebra.BLAS
                    problem = SDPA.read_dat_s($(repr(example1)), $(type_name))
                    params = $(params_ctor)
                    touch($(repr(ready)))
                    t_barrier = time()
                    while !isfile($(repr(sibling))) && time() - t_barrier < 60.0
                        sleep(0.05)
                    end
                    if !isfile($(repr(sibling)))
                        # Barrier timeout: the sibling never showed up
                        # (e.g. cold package cache made it start late).
                        # Exit NONZERO so the parent's success-check fails
                        # loudly with this stdout shown — instead of the
                        # solve proceeding unaligned and the OVERLAP
                        # assertion failing with a confusing message.
                        println("BARRIER_TIMEOUT")
                        exit(1)
                    end
                    println("PID ", getpid())
                    println("TSTART ", time())
                    blas_before = BLAS.get_num_threads()
                    samples = Int[]
                    done = Ref(false)
                    ts = Threads.@spawn while !done[]
                        push!(samples, BLAS.get_num_threads())
                        sleep(0.001)
                    end
                    log = SDPA.solve!(problem, params; verbose=false, io=devnull,
                                      reverse_primal_dual=false)
                    done[] = true
                    wait(ts)
                    println("TEND ", time())
                    blas_after = BLAS.get_num_threads()
                    policy = $(policy_expr)
                    println("RESULT type=", $(type_name), " phase=", log.terminal_phase,
                            " blas_before=", blas_before,
                            " blas_after=", blas_after,
                            " policy=", policy,
                            " policy_hits=", count(==(policy), samples),
                            " nsamples=", length(samples),
                            " obj=", log.final_obj_primal)
                    """
                end

                function _concurrent_cmd(script, outfile)
                    env = copy(ENV)
                    env["JULIA_NUM_THREADS"] = string(nt_sub)
                    # Keep the subprocess off the network: it loads the
                    # already-resolved project environment only.
                    env["JULIA_PKG_SERVER"] = ""
                    cmd = Cmd(`$julia_bin --project=$proj_root --startup-file=no -e $script`;
                              env=env, dir=proj_root)
                    return pipeline(cmd; stdout=outfile)
                end

                # Spawn BOTH first, then wait on both: the solves must
                # overlap in wall-clock time — sequential run(cmd) calls
                # would defeat the point of this test.
                p_b = run(_concurrent_cmd(_concurrent_script("BigFloat", ready_b, ready_f),
                                     out_b), wait=false)
                p_f = run(_concurrent_cmd(_concurrent_script("Float64", ready_f, ready_b),
                                     out_f), wait=false)
                wait(p_b)
                wait(p_f)
                ok_b = success(p_b)
                ok_f = success(p_f)
                @test ok_b
                @test ok_f
                outb = read(out_b, String)
                outf = read(out_f, String)
                if !(ok_b && ok_f)
                    println("[concurrency] bigfloat subprocess stdout:\n", outb)
                    println("[concurrency] float64 subprocess stdout:\n", outf)
                end

                _concurrent_line(out, prefix) = begin
                    cands = [strip(l) for l in split(out, '\n') if startswith(strip(l), prefix)]
                    isempty(cands) ? nothing : cands[1]
                end
                # "RESULT k1=v1 k2=v2 ..." -> Dict("k1" => "v1", ...)
                _concurrent_kv(line) = begin
                    d = Dict{String,String}()
                    for tok in split(line)[2:end]
                        k, v = split(tok, '='; limit=2)
                        d[k] = v
                    end
                    d
                end

                if ok_b && ok_f
                    pid_b = parse(Int, split(_concurrent_line(outb, "PID"))[2])
                    pid_f = parse(Int, split(_concurrent_line(outf, "PID"))[2])
                    s_b = parse(Float64, split(_concurrent_line(outb, "TSTART"))[2])
                    e_b = parse(Float64, split(_concurrent_line(outb, "TEND"))[2])
                    s_f = parse(Float64, split(_concurrent_line(outf, "TSTART"))[2])
                    e_f = parse(Float64, split(_concurrent_line(outf, "TEND"))[2])
                    kv_b = _concurrent_kv(_concurrent_line(outb, "RESULT"))
                    kv_f = _concurrent_kv(_concurrent_line(outf, "RESULT"))

                    # Two distinct OS processes, both pdOPT, objectives
                    # matching the sequential expectations.
                    @test pid_b != pid_f
                    @test kv_b["phase"] == "pdOPT"
                    @test kv_f["phase"] == "pdOPT"
                    @test isapprox(parse(BigFloat, kv_b["obj"]),
                                   BigFloat("-4.19e+01"); rtol=BigFloat("1e-30"))
                    @test parse(Float64, kv_f["obj"]) ≈ -4.19e1 rtol=1e-7

                    # BLAS policy restored: each subprocess's post-solve
                    # count equals its OWN pre-solve count (the machine's
                    # OpenBLAS default), despite the
                    # concurrent sibling having installed a different
                    # policy in the same wall-clock window.
                    @test kv_b["blas_after"] == kv_b["blas_before"]
                    @test kv_f["blas_after"] == kv_f["blas_before"]

                    # The two policies are genuinely different (BigFloat
                    # pins 1 BLAS thread, Float64 takes the subprocess
                    # thread count) and each policy was observed ACTIVE
                    # during its own solve by the sampler thread.
                    @test parse(Int, kv_b["policy"]) == 1
                    @test parse(Int, kv_f["policy"]) == nt_sub
                    @test parse(Int, kv_b["policy_hits"]) >= 1
                    @test parse(Int, kv_f["policy_hits"]) >= 1

                    # Overlap proof: the [start, end] intervals of the two
                    # solves must intersect with positive duration.
                    ov = min(e_b, e_f) - max(s_b, s_f)
                    @test ov > 0
                    @info("concurrency proof", pid_b, pid_f,
                          bigfloat_interval=(s_b, e_b),
                          float64_interval=(s_f, e_f),
                          overlap_seconds=round(ov; digits=3),
                          policy_bigfloat=kv_b["policy"],
                          policy_float64=kv_f["policy"],
                          blas_before=kv_b["blas_before"])
                end
            end
        end
    end

    @testset "time_limit_sec marks timed_out and TIME_LIMIT exit reason" begin
        # solve! accepts a wall-clock time limit. Default (nothing) must
        # keep the existing behavior.
        problem = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), BigFloat
        )
        params = SDPA.Parameters{BigFloat}(precision=256, maxIteration=200)
        log_ok = SDPA.solve!(problem, params; verbose=false, io=devnull)
        @test log_ok.timed_out == false
        @test log_ok.terminal_phase == SDPA.pdOPT
        # Converged run → OPTIMAL_TERMINAL; the legacy flag and the
        # enum must agree on both branches.
        @test log_ok.exit_reason == SDPA.OPTIMAL_TERMINAL

        # limit=0.0 expires on the first wall-clock check (any positive
        # elapsed time exceeds 0.0) and marks the log as timed out.
        log_timed = SDPA.solve!(problem, params; verbose=false, io=devnull,
                                time_limit_sec=0.0)
        @test log_timed.timed_out == true
        @test log_timed.exit_reason == SDPA.TIME_LIMIT
    end

    @testset "exit_reason distinguishes terminal vs budget exhaustion" begin
        fixture = joinpath(@__DIR__, "..", "fixtures", "example1.dat-s")

        # (a) A converged solve exits via update_check! returning false
        # (terminal phase reached) → OPTIMAL_TERMINAL.
        problem = SDPA.read_dat_s(fixture, Float64)
        params = SDPA.Parameters{Float64}()
        log = SDPA.solve!(problem, params; verbose=false, io=devnull)
        @test log.terminal_phase == SDPA.pdOPT
        @test log.exit_reason == SDPA.OPTIMAL_TERMINAL

        # (b) maxIteration=1 ends the loop at the WHILE-condition's
        # budget leg (continue_iter still true) → MAX_ITERATION, while
        # terminal_phase holds whatever phase was frozen mid-solve (not
        # asserted — the point is that the enum, not the phase, carries
        # this information).
        p_budget = SDPA.read_dat_s(fixture, Float64)
        prm_budget = SDPA.Parameters{Float64}(maxIteration=1)
        log_budget = SDPA.solve!(p_budget, prm_budget; verbose=false, io=devnull)
        @test log_budget.iterations == 1
        @test log_budget.terminal_phase != SDPA.pdOPT
        @test log_budget.exit_reason == SDPA.MAX_ITERATION

        # (c) TIME_LIMIT is covered in the time-limit testset above (both
        # the timed-out branch and the converged branch).

        # (d) CHOLESKY_FAILURE / CANNOT_MOVE: no cheap deterministic
        # trigger exists in the fixture suite (they need a numerically
        # degenerate iterate mid-solve); the enum assignments at those
        # break sites are exercised only by code inspection. Not tested.
    end

    @testset "MOI TerminationStatus maps MAX_ITERATION to ITERATION_LIMIT" begin
        # Behavioral pin for the Optimizer wiring: a maxIteration=1 solve
        # through the MOI layer (TimeLimitSec NOT set) must report
        # ITERATION_LIMIT — not the frozen phase's ALMOST_OPTIMAL — and
        # RawStatusString must name the exit reason alongside the phase.
        # MOI-layer mirror of the raw-API test (b) above.
        MOI = SDPA.MOI
        # min x1 + x2  s.t.  x1 + x2 = 1,  x ≥ 0 — an infeasible-start
        # IPM that cannot converge in one iteration (example1-class
        # fixture shrunk to an in-memory MOI model).
        src = MOI.Utilities.Model{Float64}()
        x, _ = MOI.add_constrained_variables(src, MOI.Nonnegatives(2))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(
                MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0))
        MOI.add_constraint(src,
            MOI.ScalarAffineFunction(
                MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0),
            MOI.EqualTo(1.0))
        opt = SDPA.Optimizer{Float64}(silent=true)
        MOI.set(opt, MOI.RawOptimizerAttribute("maxIteration"), 1)
        @test MOI.get(opt, MOI.TimeLimitSec()) === nothing  # no limit set
        MOI.optimize!(opt, src)
        @test opt.solution_log.exit_reason == SDPA.MAX_ITERATION
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.ITERATION_LIMIT
        raw = MOI.get(opt, MOI.RawStatusString())
        @test occursin("MAX_ITERATION", raw)
        @test occursin(SDPA._raw_status_string(opt.solution_log.terminal_phase), raw)
    end

    @testset "empty problem takes deterministic early exit" begin
        # An empty problem (empty blockStruct — the shape an empty MOI
        # model produces — or all-zero blocks) must return a deterministic
        # empty pdOPT log via the early exit: no iteration work, zero
        # objectives/residuals, μ = 0 (NOT the stale central-path
        # lambdaStar the general path reported), exit_reason ==
        # EMPTY_PROBLEM. Both display conventions must tolerate it (the
        # REVERSE path negates/swaps over empty structures; -0.0 == 0).
        mk_empty = () -> SDPA.SDPAProblem{Float64}(
            0, 0, Int[],
            SDPA.SDPAVector{Float64}(0),
            SDPA.SDPASparseLinearSpace{Float64}(0, 0),
            SDPA.SDPASparseLinearSpace{Float64}[],
        )
        params = SDPA.Parameters{Float64}()
        for rev in (false, true)
            log = SDPA.solve!(mk_empty(), params; verbose=false, io=devnull,
                              reverse_primal_dual=rev)
            @test log.terminal_phase == SDPA.pdOPT
            @test log.iterations == 0
            @test log.exit_reason == SDPA.EMPTY_PROBLEM
            @test log.final_obj_primal == 0
            @test log.final_obj_dual == 0
            @test log.final_mu == 0
            @test log.final_primal_res == 0
            @test log.final_dual_res == 0
            @test isempty(log.trajectory)
            @test log.timed_out == false
        end

        # blockStruct = [0]: one zero-size block also gives nDim == 0
        # (Σ|bs| = 0) and must take the same early exit.
        p_zero = SDPA.SDPAProblem{Float64}(
            0, 1, [0],
            SDPA.SDPAVector{Float64}(0),
            SDPA.SDPASparseLinearSpace{Float64}(0, 0),
            SDPA.SDPASparseLinearSpace{Float64}[],
        )
        log0 = SDPA.solve!(p_zero, params; verbose=false, io=devnull)
        @test log0.terminal_phase == SDPA.pdOPT
        @test log0.iterations == 0
        @test log0.exit_reason == SDPA.EMPTY_PROBLEM
        @test log0.final_obj_primal == 0
        @test log0.final_mu == 0
    end

    @testset "constraint-free problems (m=0, nDim>0) are decided directly" begin
        # With no constraints the problem is min ⟨C, X⟩ over X ⪰ 0:
        # C semidefinite → optimum 0 at X = 0 (pdOPT, 0 iterations); C
        # with a negative direction → unbounded. Regression: the general
        # path used to report an unbounded-looking phase for BOTH cases
        # (the constraint-free iterate diverges with nothing to anchor it).
        mk_c_only = (diagvals, lpvals) -> begin
            n = length(diagvals)
            C = SDPA.SDPASparseLinearSpace{Float64}(1, length(lpvals))
            C.SDP_sp_index = [1]
            C.SDP_sp_block[1] = SDPA.SDPASparseMatrix{Float64}(n, n, :sparse, n)
            for (k, v) in enumerate(diagvals)
                SDPA.setelement_sdp!(C, 1, k, k, Float64(v))
            end
            for (k, v) in enumerate(lpvals)
                SDPA.setelement_lp!(C, k, Float64(v))
            end
            SDPA.SDPAProblem{Float64}(
                0, 1 + (length(lpvals) > 0 ? 1 : 0),
                length(lpvals) > 0 ? [n, -length(lpvals)] : [n],
                SDPA.SDPAVector{Float64}(0),
                C,
                SDPA.SDPASparseLinearSpace{Float64}[],
            )
        end
        params = SDPA.Parameters{Float64}()

        # All semantic assertions run with reverse_primal_dual=false (the
        # clean display convention). Under the default reverse=true the
        # solver negates C internally, so a user-C ⪰ 0 problem becomes an
        # internally UNBOUNDED one and takes the general path — that is the
        # documented REVERSE-display limitation for unbounded detection,
        # not something this early exit can or should paper over.
        log = SDPA.solve!(mk_c_only([1, 2, 3], Int[]), params;
                          verbose=false, io=devnull, reverse_primal_dual=false)
        @test log.terminal_phase == SDPA.pdOPT
        @test log.iterations == 0
        @test log.exit_reason == SDPA.OPTIMAL_TERMINAL
        @test log.final_obj_primal == 0
        @test log.final_obj_dual == 0
        @test log.final_mu == 0

        # Mixed SDP+LP, both semidefinite → same direct pdOPT.
        log_lp = SDPA.solve!(mk_c_only([2, 2], [5]), params;
                             verbose=false, io=devnull, reverse_primal_dual=false)
        @test log_lp.terminal_phase == SDPA.pdOPT
        @test log_lp.iterations == 0
        @test log_lp.final_obj_primal == 0

        # C = diag(-1, 2, 3) has a negative direction → truly unbounded.
        log_unb = SDPA.solve!(mk_c_only([-1, 2, 3], Int[]), params;
                              verbose=false, io=devnull, reverse_primal_dual=false)
        @test log_unb.terminal_phase in (SDPA.pUNBD, SDPA.dUNBD)
        @test log_unb.terminal_phase != SDPA.pdOPT

        # Negative LP scalar → also unbounded (PSD C on the SDP side).
        log_lp_unb = SDPA.solve!(mk_c_only([2, 2], [-3]), params;
                                 verbose=false, io=devnull,
                                 reverse_primal_dual=false)
        @test log_lp_unb.terminal_phase in (SDPA.pUNBD, SDPA.dUNBD)

        # force_sparse with m = 0 must fail loudly (not inside METIS).
        err = try
            SDPA.solve!(mk_c_only([1, 2, 3], Int[]), params;
                        verbose=false, io=devnull, force_sparse=true)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("force_sparse", err.msg)
    end

    @testset "warm-start init point is not mutated by solve!" begin
        # solve! iterates on current_pt in place; the solve must work on a
        # value copy so the caller's init object is unchanged after the
        # solve and can be reused or inspected.
        problem = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), Float64
        )
        params = SDPA.Parameters{Float64}()
        init = SDPA.SDPASolution{Float64}(
            problem.m, problem.blockStruct, params.lambdaStar)
        x_before = copy(init.xMat.SDP_block[1].de_ele)
        y_before = copy(init.yVec.ele)
        z_before = copy(init.zMat.SDP_block[1].de_ele)
        log = SDPA.solve!(problem, params; verbose=false, io=devnull, init=init)
        @test log.terminal_phase == SDPA.pdOPT
        @test init.xMat.SDP_block[1].de_ele == x_before
        @test init.yVec.ele == y_before
        @test init.zMat.SDP_block[1].de_ele == z_before
        # Warm start from the cold-start point must reproduce the cold run.
        log_cold = SDPA.solve!(problem, params; verbose=false, io=devnull)
        @test log.iterations == log_cold.iterations
        @test log.final_obj_primal == log_cold.final_obj_primal
    end

    @testset "warm start via ini-s round-trip reaches the same optimum" begin
        # Exercise the read_ini_s → solve!(init=…) hand-off end-to-end
        # with REAL solved values (the init-non-mutation testset above
        # locks non-mutation with a cold-start point; the io tests lock
        # parsing with hand-written fixtures). Here a converged example1
        # point is exported to the NSPS-exporter ini layout, read back,
        # and warm-started — it must reach the SAME optimum.
        T = Float64
        fixture = joinpath(@__DIR__, "..", "fixtures", "example1.dat-s")
        params = SDPA.Parameters{T}()
        # reverse_primal_dual=false: the log's X/y/Z ARE the internal
        # iterates (no display swap), so the ini round-trip is
        # value-exact and the warm start starts from the solved point.
        problem = SDPA.read_dat_s(fixture, T)
        log_cold = SDPA.solve!(problem, params; verbose=false, io=devnull,
                               reverse_primal_dual=false)
        @test log_cold.terminal_phase == SDPA.pdOPT

        # Mini exporter mirroring the NSPS layout read_ini_s
        # auto-detects: line 1 = y in SCS/MOSEK sign (the reader stores
        # it NEGATED, so write -y), then 4-tuples `l i j value` with l in
        # 1..nBlock = Z, nBlock+1..2nBlock = X; SDP blocks give the
        # upper triangle, LP blocks i == j.
        buf = IOBuffer()
        println(buf, join([-v for v in log_cold.yVec.ele], ", "))
        nBlock = length(problem.blockStruct)
        for (offset, dls) in ((0, log_cold.zMat), (nBlock, log_cold.xMat))
            sdp_k = 0
            for blk_idx in 1:nBlock
                bs = problem.blockStruct[blk_idx]
                if bs >= 2
                    sdp_k += 1
                    blk = dls.SDP_block[sdp_k].de_ele
                    for j in 1:bs, i in 1:j
                        println(buf, offset + blk_idx, " ", i, " ", j, " ",
                                string(blk[i, j]))
                    end
                else
                    lp_off = sum(abs(x) for x in problem.blockStruct[1:blk_idx-1]
                                 if x < 2; init=0)
                    for i in 1:abs(bs)
                        println(buf, offset + blk_idx, " ", i, " ", i, " ",
                                string(dls.LP_block[lp_off + i]))
                    end
                end
            end
        end
        ini = joinpath(mktempdir(), "example1_warm.ini-s")
        write(ini, take!(buf))

        # The read-back point must BE the solved point: string(Float64)
        # is the shortest round-tripping repr, so parse(T, s) == s's
        # value exactly and every entry compares with ==. (Blocks are
        # mutable structs — compare the de_ele matrices, not the block
        # objects, whose == is object identity.)
        init = SDPA.read_ini_s(ini, problem)
        @test init.yVec.ele == log_cold.yVec.ele
        @test [b.de_ele for b in init.xMat.SDP_block] ==
              [b.de_ele for b in log_cold.xMat.SDP_block]
        @test [b.de_ele for b in init.zMat.SDP_block] ==
              [b.de_ele for b in log_cold.zMat.SDP_block]
        @test init.xMat.LP_block == log_cold.xMat.LP_block
        @test init.zMat.LP_block == log_cold.zMat.LP_block

        # Warm start from the exported optimum: same phase, same optimum
        # (rtol 1e-6; iteration count is NOT pinned — a warm start may
        # converge in fewer or equal iterations, here typically 0 since
        # the point already satisfies the terminal tests).
        problem2 = SDPA.read_dat_s(fixture, T)
        log_warm = SDPA.solve!(problem2, params; verbose=false, io=devnull,
                               init=init, reverse_primal_dual=false)
        @test log_warm.terminal_phase == SDPA.pdOPT
        @test log_warm.final_obj_primal ≈ log_cold.final_obj_primal rtol=1e-6
        @test log_warm.final_obj_dual ≈ log_cold.final_obj_dual rtol=1e-6
    end

    @testset "solve! builds missing aggregate index tables before solving" begin
        # A manually constructed SDPAProblem never has initialize_index!
        # called; solving it with empty aggregate tables silently produces
        # garbage. solve! must build the tables when they were never
        # initialized.
        fixture = joinpath(@__DIR__, "..", "fixtures", "example1.dat-s")
        params = SDPA.Parameters{Float64}()
        ref = SDPA.solve!(SDPA.read_dat_s(fixture, Float64), params;
                          verbose=false, io=devnull, reverse_primal_dual=false)
        manual = SDPA.read_dat_s(fixture, Float64)
        # Simulate the manual-build state: derived tables never populated.
        fill!(manual.SDP_nConstraint, 0)
        fill!(manual.LP_nConstraint, 0)
        for k in eachindex(manual.SDP_constraint); empty!(manual.SDP_constraint[k]); end
        for k in eachindex(manual.LP_constraint); empty!(manual.LP_constraint[k]); end
        log = SDPA.solve!(manual, params; verbose=false, io=devnull,
                          reverse_primal_dual=false)
        @test log.terminal_phase == ref.terminal_phase
        @test log.final_obj_primal ≈ ref.final_obj_primal rtol=1e-9
        @test log.final_obj_dual ≈ ref.final_obj_dual rtol=1e-9
    end

    # Restore the BLAS thread count the testsets above mutated.
    BLAS.set_num_threads(_blas_threads_on_entry)
end
