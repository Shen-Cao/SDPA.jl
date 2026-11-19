# SDPA.jl benchmark suite
#
# Wall-clock timing and allocation benchmarks for example1, arch0, theta5
# at Float64 and BigFloat(256), dense and sparse paths.
#
# Usage:
#   JULIA_PKG_SERVER="" julia --project=. benchmark/benchmarks.jl
#
# Optional environment variables:
#   JULIA_NUM_THREADS    governs Julia threading (Julia default: 1)
#   SDPA_BENCH_OUT       output file (default: tmp/benchmarks.md)
#   SDPA_BENCH_SKIP_BIGFLOAT=1   skip BigFloat runs (for quick smoke)
#   SDPA_BENCH_SKIP_SPARSE=1     skip sparse runs (theta5 sparse is slow)
#   SDPA_BENCH_SKIP_THETA5=1     skip theta5 (large, structurally dense)
#
# Output is appended to SDPA_BENCH_OUT as a Markdown table; each run is
# prefixed by a timestamp separator line.

using SDPA
using Printf
using Dates

# (BenchmarkTools not in [deps]; we use stdlib `@timed` for allocation stats.)

struct BenchResult
    name::String                     # e.g. "example1 / Float64 / dense"
    elapsed_sec::Float64             # wall-clock seconds
    alloc_bytes::Int                 # total bytes allocated (from @timed)
    phase::String                    # terminal phase
    obj_primal::String               # final primal objective
    iterations::Int                  # iteration count
    error::Union{Nothing,String}     # error message, if any
end

function run_one_bench(name::String, fixture::String, ::Type{T};
        maxIteration::Int=100, force_sparse::Bool=false,
        precision::Int=256) where {T<:Real}
    println()
    println("="^70)
    println("BENCH: ", name)
    println("="^70)
    path = joinpath(@__DIR__, "..", "test", "fixtures", fixture)
    p = SDPA.read_dat_s(path, T)
    println("  loaded: m=$(p.m), nBlock=$(p.nBlock)")

    # Solve once for timing (don't combine with @timed — it perturbs GC).
    if T <: BigFloat
        t0 = time_ns()
        log_result = setprecision(BigFloat, precision) do
            SDPA.solve!(p, SDPA.Parameters{T}(maxIteration=maxIteration,
                                             precision=precision);
                        verbose=false, io=devnull,
                        force_sparse=force_sparse)
        end
        t1 = time_ns()
    else
        t0 = time_ns()
        log_result = SDPA.solve!(p, SDPA.Parameters{T}(maxIteration=maxIteration);
                                  verbose=false, io=devnull,
                                  force_sparse=force_sparse)
        t1 = time_ns()
    end
    elapsed_sec = (t1 - t0) / 1e9

    # Separate @timed run for allocation stats (bytes only — Julia's
    # @timed does not expose alloc count).
    alloc_bytes = if T <: BigFloat
        stats = @timed setprecision(BigFloat, precision) do
            SDPA.solve!(p, SDPA.Parameters{T}(maxIteration=maxIteration,
                                             precision=precision);
                        verbose=false, io=devnull,
                        force_sparse=force_sparse)
        end
        # @timed returns (value, elapsed, bytes, gctime, gcstats, ...)
        stats[3]
    else
        stats = @timed SDPA.solve!(p, SDPA.Parameters{T}(maxIteration=maxIteration);
                                    verbose=false, io=devnull,
                                    force_sparse=force_sparse)
        stats[3]
    end

    phase = string(log_result.terminal_phase)
    obj_str = T <: BigFloat ? string(log_result.final_obj_primal) :
                               @sprintf("%.6f", log_result.final_obj_primal)
    iters = log_result.iterations

    println(@sprintf("  elapsed: %.3fs", elapsed_sec))
    println(@sprintf("  allocs:  %.2f GB", alloc_bytes / 1e9))
    println("  phase:   ", phase)
    println("  obj:     ", obj_str)
    println("  iters:   ", iters)
    return BenchResult(name, elapsed_sec, alloc_bytes,
                       phase, obj_str, iters, nothing)
end

function run_benchmarks(; skip_bigfloat::Bool=false,
                          skip_sparse::Bool=false,
                          skip_theta5::Bool=false)
    results = BenchResult[]
    # example1 — small (m=27), should converge in <1s Float64
    push!(results, run_one_bench(
        "example1 / Float64 / dense",
        "example1.dat-s", Float64; maxIteration=100))
    if !skip_sparse
        push!(results, run_one_bench(
            "example1 / Float64 / sparse",
            "example1.dat-s", Float64; maxIteration=100, force_sparse=true))
    end
    if !skip_bigfloat
        push!(results, run_one_bench(
            "example1 / BigFloat(256) / dense",
            "example1.dat-s", BigFloat; maxIteration=100, precision=256))
        if !skip_sparse
            push!(results, run_one_bench(
                "example1 / BigFloat(256) / sparse",
                "example1.dat-s", BigFloat; maxIteration=100, precision=256,
                force_sparse=true))
        end
    end

    # arch0 — medium (m=141)
    push!(results, run_one_bench(
        "arch0 / Float64 / dense",
        "arch0.dat-s", Float64; maxIteration=100))
    if !skip_bigfloat
        push!(results, run_one_bench(
            "arch0 / BigFloat(256) / dense",
            "arch0.dat-s", BigFloat; maxIteration=100, precision=256))
    end

    # theta5 — large (m=3028), structurally dense. The sparse path is NOT
    # a benchmark target (theta5's skyline ratio is 1.0). Skip sparse by
    # default.
    if !skip_theta5
        push!(results, run_one_bench(
            "theta5 / Float64 / dense",
            "theta5.dat-s", Float64; maxIteration=100))
    end

    return results
end

function write_results(results::Vector{BenchResult}, outfile::String,
                       label::String)
    mkpath(dirname(outfile))
    open(outfile, "a") do io
        println(io)
        println(io, "--- appended ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), " ---")
        println(io)
        println(io, "# SDPA.jl benchmark — ", label)
        println(io)
        println(io, "Date: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, "Julia threads: ", Threads.nthreads())
        println(io, "Machine: ", gethostname())
        println(io)
        println(io, "| Benchmark | Elapsed (s) | Mem (GB) | Phase | Obj | Iters |")
        println(io, "|---|---:|---:|---|---|---:|")
        for r in results
            println(io, "| ", r.name, " | ",
                    @sprintf("%.3f", r.elapsed_sec), " | ",
                    @sprintf("%.2f", r.alloc_bytes / 1e9), " | ",
                    r.phase, " | ", r.obj_primal, " | ",
                    r.iterations, " |")
        end
    end
    println("\nWrote: ", outfile)
end

# Entry point when run as a script.
if abspath(PROGRAM_FILE) == @__FILE__
    skip_bigfloat = haskey(ENV, "SDPA_BENCH_SKIP_BIGFLOAT")
    skip_sparse = haskey(ENV, "SDPA_BENCH_SKIP_SPARSE")
    skip_theta5 = haskey(ENV, "SDPA_BENCH_SKIP_THETA5")
    outfile = get(ENV, "SDPA_BENCH_OUT", joinpath("tmp", "benchmarks.md"))
    label = get(ENV, "SDPA_BENCH_LABEL", "run")

    println("SDPA.jl benchmark runner")
    println("  Julia threads: ", Threads.nthreads())
    println("  Skip BigFloat: ", skip_bigfloat)
    println("  Skip sparse:   ", skip_sparse)
    println("  Skip theta5:   ", skip_theta5)
    println("  Out file:      ", outfile)
    println("  Label:         ", label)

    results = run_benchmarks(;
        skip_bigfloat=skip_bigfloat,
        skip_sparse=skip_sparse,
        skip_theta5=skip_theta5)
    write_results(results, outfile, label)
    println("\nDONE.")
end
