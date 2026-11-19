using SDPA
using Test
using MathOptInterface
const MOI = MathOptInterface

# Curated MOI.Test conformance subset over the cones this solver
# supports (Nonnegatives, PSD triangle, Reals). MOI.Test's @requires
# machinery auto-skips features the optimizer does not advertise
# (incremental interface, GreaterThan/LessThan, primal starts, ...), so
# the include list below runs the model/objective/variable/attribute
# batteries plus the linear and PSD conic tests end-to-end through a
# caching optimizer (copy_to path — the same pipeline JuMP uses).
@testset "MOI.Test conformance subset" begin
    # NOTE: MOI.instantiate returns the RAW optimizer for this factory
    # (non-incremental), so wrap explicitly: the caching layer stores all
    # model data (variables, objective, constraints) and copy_to feeds the
    # solver — the same pipeline JuMP uses.
    opt = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.Model{Float64}(),          # cache (first argument!)
        SDPA.Optimizer{Float64}(silent=true),   # optimizer
    )
    config = MOI.Test.Config(atol=1e-5, rtol=1e-5)
    MOI.Test.runtests(
        opt,
        config;
        include = [
            "test_conic_linear",
            "test_conic_PositiveSemidefiniteConeTriangle",
            "test_conic_Reals",
            "test_model_",
            "test_objective_",
            "test_variable_",
            "test_basic_constraint_",
            "test_constraint_",
            "test_attribute_",
            "test_linear_",
        ],
        exclude = [
            # The solver does not support warm starts.
            "VariablePrimalStart",
            "ConstraintPrimalStart",
            "ConstraintDualStart",
            # Both build a model whose ONLY variable is a bare
            # MOI.add_variable scalar (no cone) and then optimize — a model
            # this solver rejects at copy time by contract
            # (UnsupportedConstraint{VariableIndex, Reals}). Without the
            # explicit exclusion the harness silently drops them ("Test
            # errored", Total 0).
            "test_attribute_RawStatusString",
            "test_attribute_SolveTimeSec",
        ],
        warn_unsupported = false,
        verbose = false,
    )
end
