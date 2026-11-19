using SDPA
using Test

# control5 convergence (dense path).
#
# control5 (m=351, nBlock=2, blocks=[50,25]) has aggregate density 1.0,
# so every density gate (aggregate and block-level skyline) selects
# DENSE. The profile-form/sparse path unit tests are in
# test/sparse/test_sparse_cholesky.jl.
#
# Float64 test: At Float64, the solver reaches noINFO (precision limit)
# but the objective converges to within 1e-3 of the published value.
# Formal pdOPT convergence requires BigFloat precision — see the
# BigFloat(256) test below.
#
# BigFloat(256) test: very slow (hours-scale), opt-in. Set
# SDPA_RUN_SLOW_TESTS=1 to run.
# C++ reference outputs captured under test/reference/.

@testset "control5 convergence (dense path, Float64)" begin
    problem = SDPA.read_dat_s(
        joinpath(@__DIR__, "..", "fixtures", "control5.dat-s"), Float64)
    params = SDPA.Parameters{Float64}(maxIteration=100)
    log = SDPA.solve!(problem, params; verbose=false, reverse_primal_dual=true)
    # At Float64, noINFO is expected (precision prevents full convergence).
    # The objective value is the primary correctness criterion.
    @test Float64(log.final_obj_primal) ≈ 16.883599050955793 atol=1e-3
    @test log.iterations > 10  # solver made progress
end

@testset "control5 convergence (dense path, BigFloat(256))" begin
    # This test is very slow (hours-scale). Set SDPA_RUN_SLOW_TESTS=1 to
    # enable.
    # C++ reference outputs captured under test/reference/.
    if get(ENV, "SDPA_RUN_SLOW_TESTS", "") in ("1", "true", "yes")  # grammar: keep in sync with the other slow-gated files
        problem = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "control5.dat-s"), BigFloat)
        params = SDPA.Parameters{BigFloat}(precision=256, maxIteration=100)
        log = SDPA.solve!(problem, params; verbose=false, reverse_primal_dual=true)
        @test log.terminal_phase == SDPA.pdOPT
        @test abs(Float64(log.final_obj_primal) - 16.883599050955793) < 1e-10
    else
        @test_skip false  # Set SDPA_RUN_SLOW_TESTS=1 to run (slow, hours-scale)
    end
end


