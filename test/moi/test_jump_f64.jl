using SDPA
using JuMP
using MathOptInterface
using Test

# Tests that JuMP can use SDPA.jl as a solver backend via the
# MOI Optimizer wrapper. This exercises the full JuMP → MOI →
# SDPA conversion pipeline at Float64.

@testset "JuMP Float64 SDP via SDPA" begin
    # Problem: min X[1,1] + X[2,2]
    #   s.t.  X[1,1] + X[2,2] == 1
    #         X ⪰ 0 (2×2)
    # Optimal: objP = 1.0
    model = Model(() -> SDPA.Optimizer{Float64}(silent=true))
    set_silent(model)
    @variable(model, X[1:2, 1:2], PSD)
    @constraint(model, X[1,1] + X[2,2] == 1)
    @objective(model, Min, X[1,1] + X[2,2])
    optimize!(model)

    @test termination_status(model) == MathOptInterface.OPTIMAL
    @test objective_value(model) ≈ 1.0 atol=1e-5
    # Check primal solution
    X_val = value.(X)
    @test isapprox(X_val, X_val', atol=1e-5)       # symmetric
    @test (X_val[1,1] + X_val[2,2]) ≈ 1.0 atol=1e-5  # constraint satisfied
end
