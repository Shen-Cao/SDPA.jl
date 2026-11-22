# Fresh-process BigFloat adapter: output the C++ display convention at
# caller-selected decimal precision, without changing the solver or IO API.
using SDPA
using LinearAlgebra
using Printf

length(ARGS) == 7 || error("expected checkout input param output bits digits threads")
checkout, data_path, param_path, out_path = ARGS[1:4]
bits, digits, threads = parse.(Int, ARGS[5:7])
realpath(dirname(dirname(pathof(SDPA)))) == realpath(checkout) ||
    error("active Julia project resolved a different SDPA checkout")
Threads.nthreads() == threads || error("Julia thread count did not match the requested count")
BLAS.set_num_threads(1)

function print_vector(io, fmt, xs; negate=false)
    print(io, "{")
    for (i, x) in enumerate(xs)
        i != 1 && print(io, ",")
        Printf.format(io, fmt, negate ? -x : x)
    end
    print(io, "}")
end

function print_blocks(io, fmt, s, shapes)
    println(io, "{")
    sdp = lp = 1
    for bs in shapes
        if bs >= 2
            print(io, "{")
            mat = s.SDP_block[sdp].de_ele
            for i in 1:bs
                i != 1 && print(io, ",\n")
                print_vector(io, fmt, @view mat[i, :])
            end
            println(io, "}")
            sdp += 1
        else
            count = abs(bs)
            print_vector(io, fmt, @view s.LP_block[lp:lp + count - 1])
            println(io)
            lp += count
        end
    end
    println(io, "}")
end

setprecision(BigFloat, bits) do
    values = first.(split.(readlines(param_path)))
    # The upstream reader stores these controls in C++ double. Give Julia
    # the same represented values; this does NOT round any data coefficients.
    eps, lambda, omega, lb, ub, beta, betabar, gamma, epsdash =
        BigFloat.(parse.(Float64, values[2:10]))
    params = SDPA.Parameters{BigFloat}(;
        maxIteration=parse(Int, values[1]), precision=bits, epsilonStar=eps,
        lambdaStar=lambda, omegaStar=omega, lowerBound=lb, upperBound=ub,
        betaStar=beta, betaBar=betabar, gammaStar=gamma, epsilonDash=epsdash,
    )
    t0 = time_ns()
    p = SDPA.read_dat_s(data_path, BigFloat)
    t1 = time_ns()
    log = SDPA.solve!(p, params; reverse_primal_dual=true, verbose=true, io=stderr)
    t2 = time_ns()
    fmt = Printf.Format("%+.$(digits)e")
    open(out_path, "w") do io
        println(io, "phase.value = ", log.terminal_phase)
        println(io, "Iteration = ", log.iterations)
        println(io, "exit reason = ", log.exit_reason)
        println(io, "precision = ", bits)
        println(io, "threads = ", Threads.nthreads())
        println(io, "julia = ", VERSION)
        println(io, "active project = ", Base.active_project())
        println(io, "mpfr = ", Base.MPFR.version())
        println(io, "blas = ", BLAS.get_config())
        println(io, "read_seconds = ", (t1 - t0) / 1e9)
        println(io, "solve_seconds = ", (t2 - t1) / 1e9)
        println(io, "solve_core_seconds = ", log.timings.total)
        println(io, "xVec =")
        # _reverse_display swaps X/Z but preserves internal y. C++ prints -y.
        print_vector(io, fmt, log.yVec.ele; negate=true)
        println(io, "\nxMat =")
        print_blocks(io, fmt, log.xMat, p.blockStruct)
        println(io, "yMat =")
        print_blocks(io, fmt, log.zMat, p.blockStruct)
    end
end
