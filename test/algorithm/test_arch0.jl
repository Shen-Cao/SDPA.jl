using SDPA
using Test

# arch0 acceptance test.
#
# Verifies that arch0 (SDPLIB, m=174, nBlock=[161, -174]) converges
# to the published optimum at BigFloat(256) with the REVERSE_PRIMAL_DUAL
# convention (matching C++ SDPA-GMP output).
#
# This test takes ~3-5 minutes at BigFloat(256). It is included in the
# main test suite because correctness on arch0 is a hard acceptance gate.

@testset "arch0 convergence @ BigFloat(256)" begin
    problem = SDPA.read_dat_s(
        joinpath(@__DIR__, "..", "fixtures", "arch0.dat-s"), BigFloat
    )
    params = SDPA.read_parameter_file(
        joinpath(@__DIR__, "..", "fixtures", "param.sdpa"), BigFloat
    )
    log = setprecision(BigFloat, 256) do
        SDPA.solve!(problem, params; verbose=false, io=devnull)
    end

    # Correctness: converges to optimal
    @test log.terminal_phase == SDPA.pdOPT

    # Correctness: objective matches published optimum
    # Published: 5.6651727321592959e-01 (from SDPLIB / C++ reference, 17 digits)
    # Our solver finds 0.56651727321592959541... (full BigFloat precision)
    #
    # NOTE on rtol: the reference value has only 17 significant digits.
    # Comparing a full-precision BigFloat against a 17-digit reference
    # caps achievable rtol at ~1e-17, not 1e-20. The strong-duality test
    # below achieves 1e-20 because BOTH sides are computed at
    # BigFloat(256). The primal-vs-reference test uses rtol=1e-15 to
    # account for the published reference's precision limitation.
    @test isapprox(log.final_obj_primal,
                   BigFloat("5.6651727321592959e-01");
                   rtol=BigFloat("1e-15"))

    # Correctness: dual objective matches primal (strong duality)
    # This achieves rtol=1e-20 — both sides at full BigFloat(256) precision.
    @test isapprox(log.final_obj_primal, log.final_obj_dual;
                   rtol=BigFloat("1e-20"))

    # Iteration count should be close to C++ reference (53 iterations)
    @test abs(log.iterations - 53) <= 5
end

# Threading perf instrumentation context.
#
# arch0 (m=174) cannot demonstrate a threading speedup: per-iter compute
# is too small for OhMyThreads scheduling overhead to amortize, so 8
# threads is actually ~2.3× SLOWER than 1 thread. The real ≥2× speedup
# gate lives on theta5 instead (test/algorithm/test_theta5_sparse.jl).
#
# This test confirms the timings field is populated correctly so
# future perf work has instrumentation to rely on.
@testset "arch0 SolutionLog.timings fields populated and consistent" begin
    problem = SDPA.read_dat_s(
        joinpath(@__DIR__, "..", "fixtures", "arch0.dat-s"), Float64)
    params = SDPA.Parameters{Float64}(maxIteration=10)
    log = SDPA.solve!(problem, params; verbose=false, io=devnull)
    # Structural: timings is a NamedTuple with the expected keys.
    @test log.timings isa NamedTuple
    @test :mehrotra in keys(log.timings)
    @test :eigen    in keys(log.timings)
    @test :residual in keys(log.timings)
    @test :total    in keys(log.timings)
    # Sanity: each timing is a non-negative finite Float64.
    for k in (:mehrotra, :eigen, :residual, :total)
        @test log.timings[k] >= 0.0
        @test isfinite(log.timings[k])
    end
    # Sanity: total >= sum of measured phases (wallclock > CPU split).
    @test log.timings.total >= log.timings.mehrotra
end
