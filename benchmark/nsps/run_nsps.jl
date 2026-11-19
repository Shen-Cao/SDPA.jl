# Cold-start NSPS benchmark run (256-bit). Args: problem-name, path-mode (dense|sparse)
#
# Usage:
#   JULIA_NUM_THREADS=8 SDPA_FORCE_SPARSE_PROFILE=1 \
#     julia --project=. benchmark/nsps/run_nsps.jl nsps_aug_nu1p0 sparse
#
# Env overrides:
#   SDPA_NSPS_MAXITER   max iterations (default 200); use 1 for a smoke run
#   SDPA_NSPS_OUT       output file (default: tmp/nsps_bench_<name>_<mode>.txt);
#                       results include the full trajectory
using SDPA
if length(ARGS) != 2
    println(stderr, "usage: julia run_nsps.jl <name> <mode>  (mode ∈ dense|sparse)")
    exit(1)
end
name = ARGS[1]
mode = ARGS[2]
maxiter = parse(Int, get(ENV, "SDPA_NSPS_MAXITER", "200"))
out = get(ENV, "SDPA_NSPS_OUT", joinpath("tmp", "nsps_bench_$(name)_$(mode).txt"))
mkpath(dirname(out))
setprecision(BigFloat, 256) do
    problem = SDPA.read_dat_s(joinpath(@__DIR__, "$name.dat-s"), BigFloat)
    params = SDPA.Parameters{BigFloat}(precision=256, maxIteration=maxiter)
    t0 = time()
    log = SDPA.solve!(problem, params; verbose=true, io=stderr, force_sparse=(mode == "sparse"))
    elapsed = time() - t0
    open(out, "w") do f
        println(f, "name: ", name, " mode: ", mode)
        println(f, "Phase: ", log.terminal_phase)
        println(f, "Iterations: ", log.iterations)
        println(f, "Wall: ", round(elapsed; digits=1), "s")
        println(f, "Per-iter: ", round(elapsed / max(log.iterations, 1); digits=1), "s")
        println(f, "final_obj_primal: ", log.final_obj_primal)
        println(f, "final_obj_dual:   ", log.final_obj_dual)
        println(f, "final_mu:         ", log.final_mu)
        println(f, "final_primal_res: ", log.final_primal_res)
        println(f, "final_dual_res:   ", log.final_dual_res)
        println(f, "--- trajectory (reversed log) ---")
        for row in log.trajectory
            println(f, "iter=", row.iter, " mu=", row.mu, " objP=", row.objP, " objD=", row.objD,
                    " thetaP=", row.thetaP, " thetaD=", row.thetaD,
                    " alphaP=", row.alphaP, " alphaD=", row.alphaD)
        end
    end
    println("BENCH-DONE ", name, " ", mode)
end
