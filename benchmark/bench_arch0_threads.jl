# Threading speedup benchmark for arch0.
#
# Usage (run once per thread count):
#   JULIA_NUM_THREADS=1 julia --project benchmark/bench_arch0_threads.jl
#   JULIA_NUM_THREADS=8 julia --project benchmark/bench_arch0_threads.jl
#
# This script measures wall-clock time per iteration at each thread count.
# Note: Julia thread count is fixed at startup via JULIA_NUM_THREADS.

using SDPA
using Printf

problem = SDPA.read_dat_s(
    joinpath(@__DIR__, "..", "test", "fixtures", "arch0.dat-s"), BigFloat)
params = SDPA.Parameters{BigFloat}(precision=256, maxIteration=50)

t0 = time()
log = SDPA.solve!(problem, params; verbose=false, reverse_primal_dual=true)
elapsed = time() - t0

nthreads = Threads.nthreads()
@printf("threads=%d iters=%d time=%.1fs per_iter=%.3fs objP=%.10g\n",
    nthreads, log.iterations, elapsed, elapsed / log.iterations,
    Float64(log.final_obj_primal))
