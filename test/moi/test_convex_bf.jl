using SDPA
using Convex
using MathOptInterface
using Test

# Tests that Convex.jl can use SDPA.jl as a solver backend.
# Convex creates auxiliary PSD variables and links them to
# objective variables through equality constraints. The variable
# linking pass in ingest.jl resolves these connections.

@testset "Convex Float64 SDP via SDPA" begin
    # Problem: min x + y
    #   s.t.  [x 0.4; 0.4 y] ⪰ 0  (2×2 PSD constraint)
    #         x + y == 1
    # Optimal: objP = 1.0 at x = y = 0.5 (X = [[0.5,0.4],[0.4,0.5]] is
    # strictly positive definite — an INTERIOR optimum). The off-diagonal
    # is 0.4 (not 0.5) because 0.5 places the optimum exactly on the PSD
    # boundary (rank-1 X) where the IPM stalls at pdFEAS; with the
    # MOI 0.5 symmetric-fill convention, a boundary-optimum fixture
    # would report ALMOST_OPTIMAL.
    x = Variable()
    y = Variable()
    problem = minimize(x + y, [x + y == 1, [x 0.4; 0.4 y] ⪰ 0])
    solve!(problem, () -> SDPA.Optimizer{Float64}(silent=true))

    @test problem.status == MathOptInterface.OPTIMAL
    @test evaluate(x + y) ≈ 1.0 atol=1e-5
end

# BigFloat Convex integration: pass numeric_type=BigFloat so Convex
# creates BigFloat constraints that match the Optimizer{BigFloat} type
# parameter.
@testset "Convex BigFloat SDP via SDPA" begin
    # Same interior-optimum fixture as the Float64 variant above.
    x = Variable()
    y = Variable()
    problem = minimize(x + y, [x + y == 1, [x 0.4; 0.4 y] ⪰ 0];
                       numeric_type=BigFloat)
    solve!(problem, () -> SDPA.Optimizer{BigFloat}(silent=true, precision=128))

    @test problem.status == MathOptInterface.OPTIMAL
    @test abs(evaluate(x + y) - 1) < BigFloat("1e-30")
end
