@testset "environment" begin
    @test parse(Int, string(VERSION.major)) == 1
    @test parse(Int, string(VERSION.minor)) >= 12  # HARD requirement: thread-safe setprecision do-block
    @test Threads.nthreads() >= 1

    # BLAS thread policy: the policy is set conditionally inside
    # _solve_inner! based on numeric type (BigFloat → 1, Float64 → nthreads);
    # __init__ does not pin BLAS to any count.
    # Module load must NOT touch the process BLAS policy: __init__ leaves
    # BLAS at the system default, and the policy is only installed/restored
    # inside solve!. Re-running __init__ must not change it either.
    n_blas_before = BLAS.get_num_threads()
    SDPA.__init__()
    @test BLAS.get_num_threads() == n_blas_before

    # GenericLinearAlgebra loaded? (load-bearing for BigFloat cholesky!/eigen)
    @test isdefined(Main, :GenericLinearAlgebra)

    # Thread-safe setprecision do-block works? (ScopedValues-based; Julia >= 1.12)
    @test setprecision(BigFloat, 100) do
        precision(BigFloat("1.0"))
    end == 100
end
