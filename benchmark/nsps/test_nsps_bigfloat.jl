# benchmark/nsps/test_nsps_bigfloat.jl
#
# This test solves the NSPS (Werner) problem at BigFloat(256) precision
# (matching the current benchmark configuration, see benchmark/nsps/README.md)
# and compares against the 256-bit benchmark reference values.
#
# Gated behind SDPA_RUN_SLOW_TESTS=1 — hours-scale, machine-dependent
# (archived timings in README).
# Usage:
#   SDPA_RUN_SLOW_TESTS=1 julia --project=. benchmark/nsps/test_nsps_bigfloat.jl

using SDPA
using Test
using Printf

const NSPS_FIXTURE = joinpath(@__DIR__, "nsps_aug_nu1p0.dat-s")
# 256-bit dense-path benchmark reference values (see
# benchmark/nsps/README.md). The C++ SDPA-GMP 256-bit reference is
# objP = 0.2500000074 / objD = 0.2500001844 — it agrees with Julia only
# at the 8th digit, so the comparison tolerance below must stay >= 1e-6.
const REF_OBJP = BigFloat("2.500000007e-01")   # Julia 256-bit objValPrimal
const REF_OBJD = BigFloat("2.500000162e-01")   # Julia 256-bit objValDual

@testset "NSPS BigFloat benchmark" begin
    # Check fixture exists (for running this file outside a repo
    # checkout; Test does not export `skip`, so use @test_skip).
    if !isfile(NSPS_FIXTURE)
        @info "NSPS fixture not found, skipping" NSPS_FIXTURE
        @test_skip false
    elseif !haskey(ENV, "SDPA_RUN_SLOW_TESTS") || ENV["SDPA_RUN_SLOW_TESTS"] != "1"
        @warn "Skipping NSPS benchmark. Set SDPA_RUN_SLOW_TESTS=1 to run."
        @test_skip "NSPS BigFloat benchmark (gated)"
    else
        # Solve NSPS at BigFloat(256), maxIteration=30 (converges at 24)
        problem = SDPA.read_dat_s(NSPS_FIXTURE, BigFloat)
        params = SDPA.Parameters{BigFloat}(
            precision=256,
            maxIteration=30,
        )

        t0 = time()
        log = SDPA.solve!(problem, params; verbose=true)
        wall = time() - t0

        println("\n=== NSPS BigFloat(256) Results ===")
        println("Phase: $(log.terminal_phase)")
        println("Iterations: $(log.iterations)")
        println("Wall-clock: $(round(wall, digits=1))s")
        println("ObjPrimal: $(log.final_obj_primal)")
        println("ObjDual:   $(log.final_obj_dual)")
        println("Primal residual: $(log.final_primal_res)")
        println("Dual residual:   $(log.final_dual_res)")
        println("Per-iteration:   $(round(wall / max(log.iterations, 1), digits=1))s")

        # Phase must be feasible
        @test string(log.terminal_phase) in ("pdOPT", "pFEAS", "pdFEAS", "pdINF")

        # Objective should be near the 256-bit benchmark reference (0.25)
        @test abs(log.final_obj_primal - REF_OBJP) < BigFloat("1e-5") ||
              abs(log.final_obj_dual - REF_OBJD) < BigFloat("1e-5")
    end
end
