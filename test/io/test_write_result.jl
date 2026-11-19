@testset "write_result" begin
    @testset "structural correctness on example1" begin
        for T in (Float64, BigFloat)
            problem = SDPA.read_dat_s(
                joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T
            )
            params = SDPA.Parameters{T}(precision=(T === BigFloat ? 256 : 53))

            # Synthetic iteration trajectory (1 row).
            trajectory = [
                SDPA.IterationRow{T}(
                    0,
                    T(10000),                          # mu
                    T(1),                              # thetaP
                    T(1),                              # thetaD
                    T(0),                              # objP
                    T(0),                              # objD
                    T(0),                              # alphaP
                    T(0),                              # alphaD
                    T(0.1),                            # beta
                ),
            ]

            # Capture output to IOBuffer.
            buf = IOBuffer()
            SDPA.write_result(
                buf, problem, params, trajectory,
                :pdOPT, 1,                        # phase, iteration
                T(1e-30), T(1e-30),               # mu, relgap
                T(2e-30), T(50),                  # gap, digits
                T(-4.19e1), T(-4.19e1),           # obj_p, obj_d
                T(1e-15), T(1e-15),               # p_feas, d_feas
                0.5;                              # compute_time
                data_filename="example1.dat-s",
                param_filename="param.sdpa",
                out_filename="example1.result",
            )
            output = String(take!(buf))

            # Structural assertions: every required section/line is present.
            @test occursin("SDPA start at", output)
            @test occursin("data      is example1.dat-s", output)
            @test occursin("parameter is param.sdpa", output)
            @test occursin("out       is example1.result", output)
            @test occursin("DENSE computation", output)
            @test occursin("mu      thetaP  thetaD  objP", output)
            @test occursin("phase.value = pdOPT", output)
            @test occursin("   Iteration = 1", output)
            @test occursin("          mu =", output)
            @test occursin("relative gap =", output)
            @test occursin("         gap =", output)
            @test occursin("      digits =", output)
            @test occursin("objValPrimal =", output)
            @test occursin("objValDual   =", output)
            @test occursin("p.feas.error =", output)
            @test occursin("d.feas.error =", output)
            @test occursin("relative eps =", output)
            @test occursin("total time   =", output)
            # The Parameters section uses the C++ display schema.
            @test occursin("Parameters are", output)
            # "Parameters are" appears EXACTLY once, as in the C++
            # reference output.
            @test length(collect(eachmatch(r"(?m)^Parameters are$", output))) == 1
            @test occursin("** Parameters **", output)
            @test occursin("maxIteration =", output)
            @test occursin("epsilonStar  =", output)
            @test occursin("precision    =", output)
            @test occursin("xPrint       = %+50.40Fe", output)
            @test !occursin("unsigned int maxIteration", output)
        end
    end

    @testset "relative eps reflects solve precision" begin
        # eps(BigFloat) depends on the GLOBAL precision at call time; a
        # 200-bit solve whose .result is written later (at the ambient
        # 256-bit default) would print 2^-255-ish instead of 2^-199. The
        # `precision` kwarg pins the printed eps to the solve precision;
        # the default (`nothing`) derives from `params.precision`.
        T = BigFloat
        problem = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T)
        params = SDPA.Parameters{T}()   # DEFAULT preset: precision = 200
        trajectory = SDPA.IterationRow{T}[]

        function printed_eps(; kw...)
            buf = IOBuffer()
            SDPA.write_result(
                buf, problem, params, trajectory, :noINFO, 0,
                T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0), 0.0; kw...)
            out = String(take!(buf))
            m = match(r"relative eps\s*=\s*([^\n]+)", out)
            @test m !== nothing
            # 40 significant digits are printed; parse back at ambient
            # precision (>= 200 bits holds ~60 digits, so exact enough).
            return parse(BigFloat, strip(m.captures[1]))
        end

        # The regression case: kwarg 200 at mismatched ambient 256 must
        # print eps(200) = 2^-199-ish, not the ambient 2^-255-ish.
        setprecision(BigFloat, 256) do
            v200 = printed_eps(precision=200)
            @test 0.5 <= v200 / BigFloat(2)^-(200 - 1) <= 2
            # Default kwarg now derives from params.precision (200), NOT
            # the ambient 256 — the same 2^-199-ish as the explicit kwarg.
            v_def = printed_eps()
            @test 0.5 <= v_def / BigFloat(2)^-(200 - 1) <= 2
        end
        # Explicit kwarg overrides params; default follows params.
        setprecision(BigFloat, 200) do
            @test 0.5 <= printed_eps(precision=200) / BigFloat(2)^-(200 - 1) <= 2
            @test 0.5 <= printed_eps() / BigFloat(2)^-(200 - 1) <= 2
            v256 = printed_eps(precision=256)
            @test 0.5 <= v256 / BigFloat(2)^-(256 - 1) <= 2
        end
    end

    @testset "exit reason line" begin
        # exit_reason (ExitReason enum, untyped kwarg — write_result.jl is
        # included before solve.jl defines the enum) prints as its
        # identifier under `phase.value`; the default omits the line
        # entirely (legacy output shape).
        problem = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), Float64)
        params = SDPA.Parameters{Float64}()
        trajectory = SDPA.IterationRow{Float64}[]
        buf = IOBuffer()
        SDPA.write_result(
            buf, problem, params, trajectory, :noINFO, 0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            exit_reason=SDPA.MAX_ITERATION,
        )
        out = String(take!(buf))
        @test occursin("exit reason = MAX_ITERATION", out)
        buf2 = IOBuffer()
        SDPA.write_result(
            buf2, problem, params, trajectory, :noINFO, 0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        @test !occursin("exit reason", String(take!(buf2)))
    end

    @testset "Parameters section matches C++ display schema byte-for-byte" begin
        # The C++ reference .result file was generated by Parameter::display
        # with the default GMP presets; our block must be byte-identical.
        problem = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), BigFloat)
        params = SDPA.Parameters{BigFloat}()  # DEFAULT GMP presets
        buf = IOBuffer()
        SDPA.write_result(
            buf, problem, params, SDPA.IterationRow{BigFloat}[],
            :noINFO, 0,
            BigFloat(0), BigFloat(0), BigFloat(0), BigFloat(0),
            BigFloat(0), BigFloat(0), BigFloat(0), BigFloat(0), 0.0,
        )
        our_lines = split(String(take!(buf)), '\n')
        ref_lines = split(read(
            joinpath(@__DIR__, "..", "reference", "example1.dat-s.result"), String), '\n')
        our_start = findfirst(l -> startswith(l, "** Parameters **"), our_lines)
        ref_start = findfirst(l -> startswith(l, "** Parameters **"), ref_lines)
        @test our_start !== nothing
        @test ref_start !== nothing
        # The C++ block is "** Parameters **" + 15 lines.
        @test join(our_lines[our_start:(our_start + 15)], '\n') ==
              join(ref_lines[ref_start:(ref_start + 15)], '\n')
    end

    @testset "reference sidecars agree with .result files" begin
        # The test/reference/*.json sidecars are
        # extracted from the adjacent C++ .result files by extract.jl, but
        # were read by no test — a regenerated sidecar could silently drift
        # from its .result. Wire every sidecar scalar into an equality
        # assertion against the .result summary it was extracted from.
        # Parsing mirrors _golden_cpp_primal (test_block_sparse_f1.jl):
        # regex over the flat file, NUMERIC compare — never string compare.
        # Float64 equality is exact here: extract.jl stores
        # repr(parse(Float64, <.result text>)), whose round-trip is exact
        # (the printed and repr'd digits may differ; the bits do not).
        REF_DIR = joinpath(@__DIR__, "..", "reference")
        # sidecar key => (value kind, .result summary regex); the regex
        # shapes mirror extract.jl's SCALAR_PATTERNS.
        result_patterns = (
            ("phase",        :string, r"phase\.value\s*=\s*(\S+)"),
            ("iteration",    :int,    r"\bIteration\s*=\s*(\d+)\s*$"m),
            ("mu",           :float,  r"\bmu\s*=\s*(-?\d+\.\d+(?:e[+-]?\d+)?)"),
            ("objValPrimal", :float,  r"objValPrimal\s*=\s*(-?\d+\.\d+(?:e[+-]?\d+)?)"),
            ("objValDual",   :float,  r"objValDual\s*=\s*(-?\d+\.\d+(?:e[+-]?\d+)?)"),
            ("pFeasError",   :float,  r"p\.feas\.error\s*=\s*(-?\d+\.\d+(?:e[+-]?\d+)?)"),
            ("dFeasError",   :float,  r"d\.feas\.error\s*=\s*(-?\d+\.\d+(?:e[+-]?\d+)?)"),
        )
        for (result_name, json_name) in (
            ("example1.dat-s.result", "example1.json"),
            ("arch0.dat-s.result",    "arch0.json"),
            ("control5.dat-s.result", "control5.json"),
        )
            @testset "$json_name vs $result_name" begin
                result_text = read(joinpath(REF_DIR, result_name), String)
                json_text = read(joinpath(REF_DIR, json_name), String)
                for (key, kind, pat) in result_patterns
                    # Sidecar values: extract.jl writes floats unquoted as
                    # repr(Float64), ints unquoted, strings quoted.
                    if kind === :string
                        mj = match(Regex("\"$(key)\"\\s*:\\s*\"([^\"]*)\""), json_text)
                    else
                        mj = match(Regex("\"$(key)\"\\s*:\\s*(-?[0-9.eE+-]+)"), json_text)
                    end
                    mr = match(pat, result_text)
                    if key in ("iteration", "objValPrimal")
                        # The two load-bearing fields must always be present
                        # in both files; optional fields are skipped below.
                        @test mj !== nothing
                        @test mr !== nothing
                    end
                    (mj === nothing || mr === nothing) && continue
                    jv, rv = mj.captures[1], mr.captures[1]
                    if kind === :string
                        @test jv == rv
                    elseif kind === :int
                        @test parse(Int, jv) == parse(Int, rv)
                    else
                        @test parse(Float64, jv) == parse(Float64, rv)
                    end
                end
                # The iteration count must also agree with the LAST row of
                # the iteration table (the summary counts those rows).
                mj = match(r"\"iteration\"\s*:\s*(\d+)", json_text)
                if mj !== nothing
                    rows = collect(eachmatch(
                        r"^\s*(\d+)\s+[-+]?\d+\.\d+e[+-]\d+"m, result_text))
                    @test !isempty(rows)
                    @test parse(Int, rows[end].captures[1]) ==
                          parse(Int, mj.captures[1])
                end
            end
        end
    end

    @testset "SPARSE mode flag" begin
        problem = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), Float64
        )
        params = SDPA.Parameters{Float64}()
        trajectory = SDPA.IterationRow{Float64}[]
        buf = IOBuffer()
        SDPA.write_result(
            buf, problem, params, trajectory,
            :noINFO, 0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            sparse_mode=true,
        )
        @test occursin("SPARSE computation", String(take!(buf)))
    end

    @testset "solution dump sections" begin
        # Build a small problem + final iterate, verify xMat/yVec/zMat
        # sections render.
        problem = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), Float64
        )
        params = SDPA.Parameters{Float64}()
        trajectory = SDPA.IterationRow{Float64}[]

        xMat = SDPA.SDPADenseLinearSpace{Float64}([2])
        xMat.SDP_block[1][1, 1] = 1.0
        xMat.SDP_block[1][2, 2] = 2.0
        yVec = SDPA.SDPAVector{Float64}([10.0, 20.0, 30.0])
        zMat = SDPA.SDPADenseLinearSpace{Float64}([2])
        zMat.SDP_block[1][1, 1] = 5.0

        buf = IOBuffer()
        SDPA.write_result(
            buf, problem, params, trajectory,
            :pdOPT, 0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            xMat=xMat, yVec=yVec, zMat=zMat,
        )
        out = String(take!(buf))
        @test occursin("xMat = ", out)
        @test occursin("yVec = ", out)
        @test occursin("zMat = ", out)
        # Solution values should appear in the dump.
        @test occursin("+1.0000000000000000e+00", out)   # xMat[1,1] = 1
        @test occursin("+2.0000000000000000e+00", out)   # xMat[2,2] = 2
        @test occursin("+2.0000000000000000e+01", out)   # yVec[2] = 20
    end

    @testset "write_result BigFloat precision" begin
        # T-aware _scalar_fmt helpers must
        # preserve enough digits at BigFloat(256). C++ uses %+.40Fe (40
        # significant digits); our BigFloat helper matches that. If the
        # T-aware format ever falls back to Float64's "%.16e" the digit
        # count would drop to ~17, well below 30.
        T = BigFloat
        problem = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T)
        params = SDPA.Parameters{T}(precision=256)
        log = SDPA.solve!(problem, params; verbose=false, io=devnull)
        buf = IOBuffer()
        SDPA.write_result(
            buf, problem, params, log.trajectory, Symbol(string(log.terminal_phase)),
            log.iterations, log.final_mu,
            BigFloat(0), BigFloat(0), BigFloat(0),   # relgap, gap, digits
            log.final_obj_primal, log.final_obj_dual,
            log.final_primal_res, log.final_dual_res, 0.0,
        )
        output = String(take!(buf))
        m = match(r"mu\s*=\s*([^\n]+)"i, output)
        @test m !== nothing
        mu_str = m.captures[1]
        digit_count = count(c -> isdigit(c), mu_str)
        @test digit_count >= 30
    end

    @testset "SolveInfo adapter signatures" begin
        # Both write_result and print_one_iteration must have >= 2 methods:
        # the detailed signature (canonical internal API) plus the
        # SolveInfo-based adapter.
        @test length(methods(SDPA.write_result)) >= 2
        @test length(methods(SDPA.print_one_iteration)) >= 2

        # Smoke-test the print_one_iteration adapter on example1 at Float64.
        T = Float64
        problem = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T)
        params = SDPA.Parameters{T}()
        # Build a trivial SolveInfo via the documented constructor.
        sol = SDPA.SDPASolution{T}(problem.m, problem.blockStruct, params.lambdaStar)
        solve_info = SDPA.SolveInfo{T}(problem, sol, params.lambdaStar, params.omegaStar)
        # Duck-typed alpha/beta with the fields the adapter reads.
        alpha_like = (primal=T(0.5), dual=T(0.5))
        beta_like  = (value=T(0.1),)
        buf = IOBuffer()
        SDPA.print_one_iteration(buf, 1, T(1), T(1e-3), T(1e-4), solve_info, alpha_like, beta_like)
        out = String(take!(buf))
        @test occursin(r"^\s*1 ", out)  # iteration row begins with iter number
        # thetaP/thetaD must print into their own columns (not both theta).
        fields = split(out)
        @test length(fields) == 9
        @test fields[3] == "1.0e-03"  # thetaP column
        @test fields[4] == "1.0e-04"  # thetaD column

        # Smoke-test the write_result adapter too: needs a Symbol phase.
        buf2 = IOBuffer()
        SDPA.write_result(
            buf2, problem, nothing, params, :pdOPT, solve_info, 0.0, 1)
        out2 = String(take!(buf2))
        @test occursin("phase.value = pdOPT", out2)
        @test occursin("   Iteration = 1", out2)
    end

    @testset "trajectory exponent formatting" begin
        # Regression for the %+7.2Fe specifier inherited from C++ SDPA-GMP
        # (mpf_t scientific notation). Julia's Printf does not support "F"
        # before "e": it renders fixed-point plus a stray literal "e" with no
        # exponent digits (e.g. "-41.90e"). Julia's %e must produce a
        # well-formed two-digit exponent (e.g. "-4.19e+01").
        for T in (Float64, BigFloat)
            row = SDPA.IterationRow{T}(
                1, T(1e-4), T(1e-3), T(1e-3),   # iter, mu, thetaP, thetaD
                T(-4.19e1), T(-4.19e1),         # objP, objD
                T(0.5), T(0.5), T(0.1),         # alphaP, alphaD, beta
            )
            buf = IOBuffer()
            SDPA.print_one_iteration(buf, row)
            line = String(take!(buf))
            fields = split(line)
            @test length(fields) == 9
            # Objective columns (fields 5, 6) need a scientific exponent with
            # a two-digit exponent after "e" — never a bare trailing "e".
            for col in (5, 6)
                @test occursin(r"[+-]?[0-9]+\.[0-9]{2}e[+-][0-9]{2}", fields[col])
            end
            # No stray "e" anywhere in the row (an "e" must be part of a
            # well-formed exponent, never followed by whitespace or line end).
            @test !occursin(r"e(\s|$)", line)
        end
    end

    @testset "multi-scalar LP block printing" begin
        # Regression: _print_dense_linear_space indexed LP_block by LP BLOCK
        # count, not per-scalar offset. blockStruct [-3,-3] holds 6 LP scalars;
        # all six must print in scalar order, not just one per block.
        dls = SDPA.SDPADenseLinearSpace{Float64}([-3, -3])
        @test length(dls.LP_block) == 6
        for k in 1:6
            dls.LP_block[k] = Float64(k)
        end
        buf = IOBuffer()
        SDPA._print_dense_linear_space(buf, dls, [-3, -3])
        out = String(take!(buf))
        expected = [
            "+1.0000000000000000e+00",
            "+2.0000000000000000e+00",
            "+3.0000000000000000e+00",
            "+4.0000000000000000e+00",
            "+5.0000000000000000e+00",
            "+6.0000000000000000e+00",
        ]
        for (k, exp) in enumerate(expected)
            @test occursin(exp, out)
        end
        pos = [findfirst(exp, out)[1] for exp in expected]
        @test issorted(pos)
    end
end
