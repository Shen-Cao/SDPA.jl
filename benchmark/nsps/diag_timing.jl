# benchmark/nsps/diag_timing.jl
#
# Timing probe driver for the NSPS benchmark set (256-bit). Runs a small
# number of iterations (default 3) and prints, as parseable TIMING_* lines:
#   - TIMING_PROBE  header (name, mode, env flags, threads, maxiter, precision)
#   - TIMING_PROBLEM  m / nBlock / blockStruct
#   - TIMING_ROW    full trajectory rows (iter mu objP objD thetaP thetaD
#                   alphaP alphaD beta) — for cross-run bit-identity checks
#   - TIMING_DONE   phase, iterations, wall, per-iter, log.timings breakdown
#
# The solve! verbose @info lines (stderr) additionally report the
# block-level skyline entries/density for the chosen ordering.
#
# Usage (mode=sparse pairs with SDPA_FORCE_SPARSE_PROFILE=1; NSPS's
# block-level density exceeds the 25% gate, so the dense path would
# otherwise be taken):
#   JULIA_NUM_THREADS=8 SDPA_FORCE_SPARSE_PROFILE=1 \
#     julia --project=. benchmark/nsps/diag_timing.jl nsps_aug_nu1p0 sparse
#   # block-level ordering variant:
#   JULIA_NUM_THREADS=8 SDPA_FORCE_SPARSE_PROFILE=1 SDPA_BLOCK_ORDERING=1 \
#     julia --project=. benchmark/nsps/diag_timing.jl nsps_aug_nu1p0 sparse
#
# Env overrides:
#   SDPA_DIAG_MAXITER    max iterations (default 3)
#   SDPA_DIAG_PRECISION  BigFloat bits (default 256)
#
# Note: this driver relies only on the committed public logging interface
# (per-iteration wall time and the coarse log.timings breakdown).
using SDPA
using Printf

name = ARGS[1]
mode = ARGS[2]
path = isfile(name) ? name : joinpath(@__DIR__, "$name.dat-s")
maxiter = parse(Int, get(ENV, "SDPA_DIAG_MAXITER", "3"))
precision_bits = parse(Int, get(ENV, "SDPA_DIAG_PRECISION", "256"))
println("TIMING_PROBE name=", name, " path=", path, " mode=", mode,
        " maxiter=", maxiter, " precision=", precision_bits,
        " threads=", Threads.nthreads(),
        " block_ordering=", get(ENV, "SDPA_BLOCK_ORDERING", ""),
        " ordering_env=", get(ENV, "SDPA_ORDERING", ""),
        " force_sparse_profile=", get(ENV, "SDPA_FORCE_SPARSE_PROFILE", ""))
flush(stdout)

setprecision(BigFloat, precision_bits) do
    problem = SDPA.read_dat_s(path, BigFloat)
    params = SDPA.Parameters{BigFloat}(precision=precision_bits, maxIteration=maxiter)
    println("TIMING_PROBLEM m=", problem.m, " nBlock=", problem.nBlock,
            " blockStruct=", problem.blockStruct, " mode=", mode)
    flush(stdout)
    t0 = time()
    log = SDPA.solve!(problem, params; verbose=true, io=stderr,
                      force_sparse=(mode == "sparse"))
    elapsed = time() - t0

    for row in log.trajectory
        println("TIMING_ROW iter=", row.iter, " mu=", row.mu,
                " objP=", row.objP, " objD=", row.objD,
                " thetaP=", row.thetaP, " thetaD=", row.thetaD,
                " alphaP=", row.alphaP, " alphaD=", row.alphaD,
                " beta=", row.beta)
    end

    println("TIMING_DONE name=", name, " mode=", mode,
            " iterations=", log.iterations, " phase=", log.terminal_phase,
            " wall=", round(elapsed; digits=1), "s",
            " per_iter=", round(elapsed / max(log.iterations, 1); digits=1), "s",
            " final_mu=", log.final_mu, " final_objP=", log.final_obj_primal,
            " final_objD=", log.final_obj_dual)
    println("LOG_TIMINGS mehrotra=", round(log.timings.mehrotra; digits=1),
            " eigen=", round(log.timings.eigen; digits=1),
            " residual=", round(log.timings.residual; digits=1),
            " total=", round(log.timings.total; digits=1))
    flush(stdout)
end
