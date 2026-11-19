# benchmark/bench_bigfloat.jl
#
# Wall-clock and allocation benchmark for BigFloat solves (example1, arch0,
# NSPS) plus kernel-level @allocated probes for Rgemm!/Rpotf2!/Rtrsm!.
#
# Usage:
#   JULIA_PKG_SERVER="" JULIA_NUM_THREADS=4 \
#   julia --project=. benchmark/bench_bigfloat.jl
#
# Optional env vars:
#   SDPA_BENCH_OUT    output markdown file (default: tmp/bigfloat_bench.md)
#   SDPA_BENCH_SKIP_NSPS=1   skip NSPS run (slow — tens of minutes)

using SDPA
using Printf
using Dates
using LinearAlgebra

const OUT = get(ENV, "SDPA_BENCH_OUT", joinpath("tmp", "bigfloat_bench.md"))
const SKIP_NSPS = haskey(ENV, "SDPA_BENCH_SKIP_NSPS")

struct Result
    name::String
    elapsed_sec::Float64
    alloc_bytes::Int
    alloc_count::Int
    phase::String
    obj_primal::String
    iterations::Int
    precision::Int
end

function run_solve(name::String, fixture::String, ::Type{T};
        maxIteration::Int=100, precision::Int=256,
        force_sparse::Bool=false) where {T<:Real}
    println("\n=== $name ===")
    problem = SDPA.read_dat_s(fixture, T)
    params = SDPA.Parameters{T}(precision=precision, maxIteration=maxIteration)
    t0 = time()
    # Capture allocation stats.
    stats = @timed begin
        log = SDPA.solve!(problem, params; verbose=true, force_sparse=force_sparse)
    end
    elapsed = time() - t0
    r = Result(name, elapsed, stats.bytes, stats.gcstats.allocd,
               string(log.terminal_phase), string(log.final_obj_primal),
               log.iterations, precision)
    println("Phase: ", r.phase, "  Iter: ", r.iterations,
            "  ObjP: ", r.obj_primal)
    println("Wall: ", round(elapsed, digits=3), "s",
            "  Alloc bytes: ", r.alloc_bytes,
            "  Alloc count: ", r.alloc_count)
    return r
end

# Measure Rgemm! allocation for 10×10 BigFloat matrix multiply.
function bench_rgemm_alloc()
    println("\n=== Rgemm! @allocated (10×10 BigFloat(256)) ===")
    setprecision(BigFloat, 256) do
        n = 10
        A = [BigFloat(rand()) for _ in 1:n, _ in 1:n]
        B = [BigFloat(rand()) for _ in 1:n, _ in 1:n]
        C = [BigFloat(rand()) for _ in 1:n, _ in 1:n]
        alpha = one(BigFloat)
        beta = one(BigFloat)
        # Warmup
        SDPA.Rgemm!("N", "N", n, n, n, alpha, A, B, beta, C)
        # Measure
        alloc_bytes = @allocated SDPA.Rgemm!("N", "N", n, n, n, alpha, A, B, beta, C)
        println("Rgemm! NN 10×10 @allocated = ", alloc_bytes, " bytes")
        return alloc_bytes
    end
end

function bench_rpotf2_alloc()
    println("\n=== Rpotf2! @allocated (10×10 BigFloat(256)) ===")
    setprecision(BigFloat, 256) do
        n = 10
        # Build an SPD matrix A = B'B + n*I
        B = [BigFloat(rand()) for _ in 1:n, _ in 1:n]
        A = B' * B
        for i in 1:n; A[i, i] += n; end
        # Warmup
        Ac = copy(A)
        SDPA.Rpotf2!("U", n, Ac)
        # Measure
        Ac2 = copy(A)
        alloc_bytes = @allocated SDPA.Rpotf2!("U", n, Ac2)
        println("Rpotf2! 10×10 @allocated = ", alloc_bytes, " bytes")
        return alloc_bytes
    end
end

function bench_rtrsm_alloc()
    println("\n=== Rtrsm! @allocated (10×10 BigFloat(256)) ===")
    setprecision(BigFloat, 256) do
        n = 10
        B = [BigFloat(rand()) for _ in 1:n, _ in 1:n]
        A = B' * B
        for i in 1:n; A[i, i] += n; end
        # Cholesky factor
        U = copy(A)
        SDPA.Rpotf2!("U", n, U)
        X = [BigFloat(rand()) for _ in 1:n, _ in 1:n]
        alpha = one(BigFloat)
        # Warmup
        Xc = copy(X)
        SDPA.Rtrsm!("L", "U", "N", "N", n, n, alpha, U, Xc)
        # Measure
        Xc2 = copy(X)
        alloc_bytes = @allocated SDPA.Rtrsm!("L", "U", "N", "N", n, n, alpha, U, Xc2)
        println("Rtrsm! 10×10 @allocated = ", alloc_bytes, " bytes")
        return alloc_bytes
    end
end

function main()
    mkpath(dirname(OUT))
    results = Result[]
    println("=" ^ 60)
    println("SDPA.jl BigFloat benchmark")
    println("Date: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("Julia threads: ", Threads.nthreads())
    println("Output: ", OUT)
    println("=" ^ 60)

    # Kernel allocation baselines
    rgemm_bytes = bench_rgemm_alloc()
    rpotf2_bytes = bench_rpotf2_alloc()
    rtrsm_bytes = bench_rtrsm_alloc()

    # example1 Float64 dense (reference)
    push!(results, run_solve("example1 / Float64 / dense",
        joinpath(@__DIR__, "..", "test", "fixtures", "example1.dat-s"),
        Float64; maxIteration=100))

    # example1 BigFloat(256) dense
    push!(results, run_solve("example1 / BigFloat(256) / dense",
        joinpath(@__DIR__, "..", "test", "fixtures", "example1.dat-s"),
        BigFloat; maxIteration=100, precision=256))

    # arch0 BigFloat(256) dense
    push!(results, run_solve("arch0 / BigFloat(256) / dense",
        joinpath(@__DIR__, "..", "test", "fixtures", "arch0.dat-s"),
        BigFloat; maxIteration=100, precision=256))

    # NSPS BigFloat(256) — slow, gated
    nsps_fixture = joinpath(@__DIR__, "nsps", "nsps_aug_nu1p0.dat-s")
    if !SKIP_NSPS
        if isfile(nsps_fixture)
            push!(results, run_solve("NSPS / BigFloat(256) / dense",
                nsps_fixture, BigFloat;
                maxIteration=5, precision=256))  # cap at 5 iterations
        else
            @info "skipping NSPS (fixture not found)"
        end
    end

    # Write markdown report
    open(OUT, "a") do io
        println(io)
        println(io, "--- appended ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), " ---")
        println(io)
        println(io, "# BigFloat benchmark")
        println(io)
        println(io, "Date: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, "Julia threads: ", Threads.nthreads())
        println(io, "Julia version: ", VERSION)
        println(io)
        println(io, "## Kernel allocation baseline (10×10 BigFloat(256))")
        println(io)
        println(io, "| Kernel | @allocated (bytes) |")
        println(io, "|--------|-------------------:|")
        println(io, "| Rgemm! NN | ", rgemm_bytes, " |")
        println(io, "| Rpotf2!   | ", rpotf2_bytes, " |")
        println(io, "| Rtrsm!    | ", rtrsm_bytes, " |")
        println(io)
        println(io, "## Solve benchmark")
        println(io)
        println(io, "| Problem | Precision | Wall (s) | Alloc bytes | Alloc count | Phase | Iter |")
        println(io, "|---------|----------:|---------:|------------:|-------------:|-------|-----:|")
        for r in results
            println(io, "| ", r.name, " | ", r.precision, " | ",
                    round(r.elapsed_sec, digits=3), " | ",
                    r.alloc_bytes, " | ", r.alloc_count, " | ",
                    r.phase, " | ", r.iterations, " |")
        end
        println(io)
        for r in results
            println(io, "## ", r.name)
            println(io)
            println(io, "- Phase: `", r.phase, "`")
            println(io, "- Iterations: ", r.iterations)
            println(io, "- Wall-clock: ", round(r.elapsed_sec, digits=3), " s")
            println(io, "- Allocated: ", r.alloc_bytes, " bytes")
            println(io, "- ObjPrimal: `", r.obj_primal, "`")
            println(io)
        end
    end
    println("\nReport written to ", OUT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
