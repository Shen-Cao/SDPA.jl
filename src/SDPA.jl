module SDPA

# SDPA.jl — pure-Julia arbitrary-precision semidefinite programming solver.
# Copyright (C) 2026 SDPA.jl contributors.
# Licensed under GPL-2.0-or-later. See LICENSE.

using LinearAlgebra
using SparseArrays
using Printf
using Dates
using MathOptInterface
using OhMyThreads
using MutableArithmetics

# Explicit public API surface (Julia ≥ 1.11 `public`): names that are
# part of the supported API without being exported into Main. Internal
# underscore names and everything else stay accessible-but-unsupported.
public solve!, read_dat, read_dat_s, read_ini_s, write_result,
       Parameters, DEFAULT, UNSTABLE_BUT_FAST, STABLE_BUT_SLOW,
       SDPAProblem, SDPASolution, SolutionLog, PhaseValue,
       ExitReason, Optimizer

include("mpfr_inplace.jl")
include("mplapack/Rblas.jl")
include("mplapack/Rgemm.jl")
include("mplapack/Rlapack.jl")
include("mplapack/dispatch.jl")
include("types/SDPAVector.jl")
include("types/SDPABlockVector.jl")
include("types/SDPADenseMatrix.jl")
include("types/SDPASparseMatrix.jl")
include("types/SDPASparseLinearSpace.jl")
include("types/SDPADenseLinearSpace.jl")
include("types/SDPAProblem.jl")
include("types/SDPASolution.jl")
include("types/SDPAResiduals.jl")
include("types/SDPAWorkVariables.jl")
include("Parameters.jl")
include("algorithm/PhaseMachine.jl")
include("algorithm/SolveInfo.jl")
include("algorithm/DirectionParameter.jl")
include("algorithm/JordanAlgebra.jl")
include("sparse/SchurSkyline.jl")
include("sparse/BorderedPlan.jl")
include("algorithm/NewtonState.jl")
include("algorithm/SchurComplement.jl")
include("algorithm/SchurSolve.jl")
include("algorithm/Mehrotra.jl")
include("algorithm/StepLength.jl")
include("algorithm/IterateUpdate.jl")
include("linear_algebra/dispatch.jl")
include("linear_algebra/min_eigen.jl")
include("threading/BigFloatPool.jl")
include("threading/ThreadedOps.jl")
include("io/smoke_parser.jl")
include("io/_dat_common.jl")
include("io/read_dat_s.jl")
include("io/read_dat.jl")
include("io/read_ini_s.jl")
include("io/write_result.jl")
include("solve.jl")

# MOI/JuMP integration (optional — loaded after core solver)
include("moi/Optimizer.jl")
include("moi/ingest.jl")
include("sparse/AggregateSparsity.jl")
include("sparse/SparseCholesky.jl")

# BLAS thread policy: `_solve_inner!` sets BLAS threads based on the
# numeric type:
#
#   * BigFloat: BLAS is irrelevant (BigFloat dispatches to
#     `GenericLinearAlgebra`, pure Julia), so we pin BLAS to 1 thread
#     to avoid it competing with Julia threads.
#   * Float64: LAPACK is the hot path (`cholesky!`, `mul!`, `ldiv!`), and
#     BLAS multi-threading is the primary speedup lever there. We set
#     BLAS threads to `Threads.nthreads()` so the user's
#     `JULIA_NUM_THREADS` setting governs LAPACK parallelism.
#
# Oversubscription is avoided by the BigFloat path pinning BLAS to 1
# (Julia threads are the only parallelism for BigFloat) and by the
# Float64 path matching BLAS threads to Julia threads (1:1).
# References: ThreadPinning.jl BLAS docs; JuliaLang/julia#44201.
function __init__()
    # Runtime MPFR thread-safety assertion.
    # MPFR must be built with --enable-thread-safe for our per-thread
    # BigFloat arithmetic to be safe. Julia's bundled libmpfr has this,
    # but a custom-built MPFR might not. We assert at module load.
    # See docs/architecture.md for the aliasing analysis and
    # test/threading/test_cross_thread_bit_identity.jl for the empirical
    # cross-thread verification.
    tls_ok = ccall((:mpfr_buildopt_tls_p, Base.MPFR.libmpfr), Cint, ())
    if tls_ok == 0
        @warn "MPFR was NOT built thread-safe (mpfr_buildopt_tls_p=0). " *
              "Concurrent BigFloat operations across Julia threads may " *
              "produce wrong results. Install a thread-safe MPFR " *
              "(configured with --enable-thread-safe)."
    end
    return nothing
end

end # module SDPA
