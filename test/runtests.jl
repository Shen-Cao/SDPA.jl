# SDPA.jl test suite.
#
# Slow end-to-end tests (the BigFloat(256) control5 and
# use_formula_per_block cases; ~2 h each) are skipped by default; set
# SDPA_RUN_SLOW_TESTS=1 to enable them.

using SDPA
using Test
using LinearAlgebra
using GenericLinearAlgebra  # CRITICAL: must be loaded for BigFloat cholesky!/eigen

@testset "SDPA.jl" begin
    @testset "smoke" begin
        @test SDPA !== nothing  # module loads
    end
    include("env.jl")
    include("smoke_parser.jl")
    include("types/test_core_types.jl")
    include("types/test_linear_space.jl")
    include("types/test_SDPAProblem.jl")
    include("types/test_solution_residuals_workvars.jl")
    include("test_Parameters.jl")
    include("io/test_read_dat_s.jl")
    include("io/test_read_dat.jl")
    include("io/test_read_ini_s.jl")
    include("io/test_write_result.jl")
    include("algorithm/test_PhaseMachine.jl")
    include("algorithm/test_SolveInfo.jl")
    include("algorithm/test_DirectionParameter.jl")
    include("algorithm/test_JordanAlgebra.jl")
    include("algorithm/test_NewtonState.jl")
    include("algorithm/test_SchurComplement.jl")
    include("algorithm/test_bmat_inplace_alloc.jl")
    include("algorithm/test_SchurSolve.jl")
    include("algorithm/test_Mehrotra.jl")
    include("algorithm/test_StepLength.jl")
    include("algorithm/test_IterateUpdate.jl")
    include("algorithm/test_solve.jl")
    include("algorithm/test_result_invariants.jl")
    include("algorithm/test_iterate_acceptance.jl")
    include("algorithm/test_reverse_precision.jl")
    include("algorithm/test_step_recovery.jl")
    include("algorithm/test_inverse_finiteness.jl")
    include("algorithm/test_constraint_free_finiteness.jl")
    include("algorithm/test_arch0.jl")
    include("moi/test_moi_basic.jl")
    include("moi/test_objective_sense.jl")
    include("moi/test_jump_f64.jl")
    include("moi/test_convex_bf.jl")
    include("moi/test_moi_full.jl")
    include("moi/test_moi_conformance.jl")
    include("linear_algebra/test_dispatch.jl")
    include("linear_algebra/test_min_eigen.jl")
    include("linear_algebra/test_triangular_fast_path.jl")
    include("threading/test_BigFloatPool.jl")
    include("threading/test_ThreadedOps.jl")
    include("threading/test_cross_thread_bit_identity.jl")
    include("algorithm/test_nested_tasks_bit_identity.jl")
    include("sparse/test_sparse_cholesky.jl")
    include("sparse/test_sparse_lp_profile.jl")
    include("sparse/test_sparse_schur_divergence.jl")
    include("sparse/test_ordering_routing.jl")
    include("sparse/test_routing_freeze.jl")
    include("algorithm/test_sparse_e2e.jl")
    include("algorithm/test_theta5_sparse.jl")
    include("algorithm/test_block_sparse_f1.jl")
    include("algorithm/test_sparse_block_index.jl")
    include("algorithm/test_use_formula_per_block.jl")
    include("algorithm/test_cpp_parity_minors.jl")
    include("algorithm/test_control5.jl")
    include("algorithm/test_empty_block_lists.jl")
    include("mpfr/test_mpfr_inplace.jl")
    include("mpfr/test_Rblas.jl")
    include("mpfr/test_Rgemm.jl")
    include("mpfr/test_Rlapack.jl")
    include("mpfr/test_aliasing.jl")
    include("repo_hygiene.jl")
    include("fixtures_checksums.jl")
end
