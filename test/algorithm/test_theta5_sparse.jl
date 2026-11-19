# theta5 sparse-path acceptance tests.
#
# theta5: m=3028, nBlock=[250] (one 250x250 SDP block).
# Density: ~0.0003 (very sparse).
#
# This is the canonical sparse-path benchmark. Verifies:
#   1. Sparse path produces same result as dense path at Float64.
#   2. Sparse path uses a real AMD ordering (a valid permutation;
#      identity is a legitimate AMD output).
#   3. (Optional, opt-in via SDPA_RUN_SLOW_TESTS=1 AND nthreads >= 8)
#      Sparse path is not materially slower than dense
#      (machine-qualified — see the perf testset below).

using SDPA
using Test

const _RUN_SLOW = get(ENV, "SDPA_RUN_SLOW_TESTS", "") in ("1", "true", "yes")  # grammar: keep in sync with the other slow-gated files

# Perf gates are machine-relative and therefore have their own opt-in,
# independent of the slow functional tests. See docs/performance.md
# ("Performance gates are machine-relative").
# for the calibration history and the rationale.
const _PERF_GATES = get(ENV, "SDPA_PERF_GATES", "") in ("1", "true", "yes")
const _GATE_RATIO = clamp(
    parse(Float64, get(ENV, "SDPA_PERF_GATE_RATIO", "0.8")), 0.0, 1.0)
const _GATE_MIN_THREADS = parse(Int, get(ENV, "SDPA_PERF_GATE_MIN_THREADS", "8"))
const _GATE_RUNS = max(1, parse(Int, get(ENV, "SDPA_PERF_GATE_RUNS", "3")))

@testset "theta5 aggregate graph yields valid AMD and METIS orderings" begin
    # The force_sparse path computes the real AMD ordering (which may
    # legitimately BE identity for some graphs — AMD's job is to reduce
    # fill, and identity may already be optimal).
    #
    # We verify: analyze_sparsity on theta5 selects sparse (since
    # force_sparse overrides) AND compute_ordering returns a valid
    # permutation.
    problem = SDPA.read_dat_s(
        joinpath(@__DIR__, "..", "fixtures", "theta5.dat-s"), Float64)
    adj = SDPA.build_aggregate_graph(problem)
    ordering = SDPA.compute_ordering(adj; method=:amd)
    # Valid permutation (not necessarily non-identity — AMD can return
    # identity when the graph is already well-ordered).
    @test length(ordering) == problem.m
    @test sort(ordering) == collect(1:problem.m)

    # METIS variant.
    ordering_metis = SDPA.compute_ordering(adj; method=:metis)
    @test length(ordering_metis) == problem.m
    @test sort(ordering_metis) == collect(1:problem.m)
end

if _RUN_SLOW
    @testset "theta5 sparse-path solve reaches feasibility at Float64 (slow)" begin
        problem = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "theta5.dat-s"), Float64)
        # C++-parity tolerances: the Parameters defaults (1e-30) are below
        # eps(Float64) ≈ 2.2e-16, so at Float64 the phase can never leave
        # noINFO (the convergence/feasibility tests are unreachable) and
        # this testset would always fail. 1e-7/1e-7 are the classic C++
        # SDPA Float64 defaults (upstream SDPA manual) — NOT this repo's
        # test/fixtures/param.sdpa, which was customized to 1e-30. Measured
        # control: pdOPT at 20 iterations with these tolerances.
        params = SDPA.Parameters{Float64}(
            maxIteration=50; epsilonStar=Float64(1e-7), epsilonDash=Float64(1e-7))
        log_sparse = SDPA.solve!(problem, params; verbose=false, io=devnull,
                                  force_sparse=true)
        @test log_sparse.terminal_phase == SDPA.pdOPT ||
              log_sparse.terminal_phase == SDPA.pdFEAS
        @test log_sparse.iterations > 5
    end
else
    @info "Set SDPA_RUN_SLOW_TESTS=1 to enable the slow theta5 solve test."
end

if _PERF_GATES
    @testset "theta5 perf gate: sparse not materially slower than dense (opt-in)" begin
        nt = Threads.nthreads()
        if nt < _GATE_MIN_THREADS
            @info "perf gate skipped" nthreads=nt required=_GATE_MIN_THREADS
            @test_skip false
        else
            params = SDPA.Parameters{Float64}(maxIteration=10)
            # Median-of-N for BOTH timings (single-shot @elapsed flakes on
            # a shared box). NOTE: the @elapsed results must be collected
            # in a vector comprehension — `sort((@elapsed f(), ...))`
            # misparses (the macro swallows the comma expression) into
            # sort(::Float64).
            _median_of_n(f, n) =
                (ts = Float64[@elapsed(f()) for _ in 1:n]; sort!(ts); ts[(n + 1) ÷ 2])
            problem_s = SDPA.read_dat_s(
                joinpath(@__DIR__, "..", "fixtures", "theta5.dat-s"), Float64)
            t_sparse = _median_of_n(() -> SDPA.solve!(problem_s, params;
                                                      verbose=false, io=devnull,
                                                      force_sparse=true), _GATE_RUNS)
            problem_d = SDPA.read_dat_s(
                joinpath(@__DIR__, "..", "fixtures", "theta5.dat-s"), Float64)
            t_dense = _median_of_n(() -> SDPA.solve!(problem_d, params;
                                                     verbose=false, io=devnull,
                                                     force_sparse=false), _GATE_RUNS)
            @info "theta5 perf gate" nthreads=nt runs=_GATE_RUNS t_sparse=t_sparse t_dense=t_dense observed_ratio=t_dense / t_sparse gate_ratio=_GATE_RATIO
            # Machine-qualified one-sided tripwire: the sparse path must
            # not be materially slower than the dense path (not a speedup
            # assertion). Machine-dependent, hence opt-in; absolute
            # performance comparisons live in benchmark/nsps/.
            # Calibration history and gate rationale:
            # docs/performance.md ("Performance gates are machine-relative").
            @test t_dense >= _GATE_RATIO * t_sparse
        end
    end
else
    @debug "perf gates disabled; set SDPA_PERF_GATES=1 to enable"
end

