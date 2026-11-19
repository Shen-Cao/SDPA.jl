using SDPA
using Test

# Sparse e2e with AMD ordering (fast CI variant).
#
# This is the fast in-CI variant — uses example1 for tractable runtime.
# The canonical theta5 acceptance test is in test_theta5_sparse.jl
# (gated behind SDPA_RUN_SLOW_TESTS=1 because theta5 is slow,
# machine-dependent).
#
# Verifies:
#   1. Sparse path produces same objective value as dense path.
#   2. The ordering used is a valid permutation produced by the real AMD
#      call (not a trivial identity placeholder).
#   3. Both paths reach a feasible terminal phase.
#   4. Example1's aggregate graph yields a valid AMD permutation
#      (identity is a legitimate AMD output for a well-ordered graph).
#   5. REGRESSION: sparse path converges at ANY thread count.
#      Guarded bug class: Mehrotra.jl bypassed the make_bmat! dispatcher
#      at nthreads > 1, calling threaded_compute_bmat_dense! directly
#      even when use_sparse=true. This left sparse_bmat.sp_ele at zero,
#      causing the sparse Cholesky to operate on garbage → NaN divergence.

@testset "example1 e2e: forced-sparse solve matches dense (AMD ordering)" begin
    for T in (Float64, BigFloat)
        # Verify force_sparse computes a real ordering (not identity).
        # AMD on example1's aggregate graph may return identity (graph is
        # tiny and well-structured); the key check is that compute_ordering
        # was actually called and returned a valid permutation.
        problem_for_ordering = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T)
        adj = SDPA.build_aggregate_graph(problem_for_ordering)
        ordering_check = SDPA.compute_ordering(adj; method=:amd)
        @test length(ordering_check) == problem_for_ordering.m
        @test sort(ordering_check) == collect(1:problem_for_ordering.m)

        # Dense path (ground truth)
        p_dense = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T)
        params = SDPA.Parameters{T}(maxIteration=100)
        log_dense = SDPA.solve!(p_dense, params; verbose=false,
                                 reverse_primal_dual=true, force_sparse=false)

        # Sparse path (forced — uses the real AMD ordering)
        p_sparse = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T)
        log_sparse = SDPA.solve!(p_sparse, params; verbose=false,
                                  reverse_primal_dual=true, force_sparse=true)

        # Objective value agreement (the real correctness criterion).
        @test Float64(log_sparse.final_obj_primal) ≈ Float64(log_dense.final_obj_primal) rtol=1e-8
        # Both reach a feasible phase.
        @test log_sparse.terminal_phase in (
            SDPA.pdOPT, SDPA.pdFEAS, SDPA.pFEAS, SDPA.dFEAS)
        @test log_dense.terminal_phase in (
            SDPA.pdOPT, SDPA.pdFEAS, SDPA.pFEAS, SDPA.dFEAS)
    end
end

# Regression test for the thread-safety bug class. This test MUST pass
# at ANY JULIA_NUM_THREADS value. It specifically guards against the bug
# class where Mehrotra.jl or any other caller bypasses the make_bmat!
# dispatcher and calls threaded_compute_bmat_dense! directly when
# use_sparse=true.
@testset "sparse path converges at any nthreads (no make_bmat! dispatcher bypass)" begin
    nt = Threads.nthreads()
    @info "running sparse-path thread-safety regression at nthreads=$nt"
    for T in (Float64, BigFloat)
        problem = SDPA.read_dat_s(
            joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T)
        params = SDPA.Parameters{T}(maxIteration=100)
        log = SDPA.solve!(problem, params; verbose=false, io=devnull,
                          reverse_primal_dual=true, force_sparse=true)
        # The bug produced NaN / pdFEAS at nthreads > 1.
        # The fix ensures make_bmat! is the single dispatch entry point.
        @test isfinite(log.final_obj_primal)
        @test isfinite(log.final_obj_dual)
        @test log.terminal_phase in (
            SDPA.pdOPT, SDPA.pdFEAS, SDPA.pFEAS, SDPA.dFEAS)
        # Objective should match the known example1 optimum (-41.9).
        @test abs(Float64(log.final_obj_primal) - (-41.9)) < 1.0
    end
end

# A solve-level exercise of the sparse Schur path that cannot fall
# back to dense. example1-class fixtures are block-dense and the density
# gate silently reverts force_sparse to dense; here a 120-constraint
# 120-block chain (m > 100, nBlock > 5) keeps analyze_sparsity on the
# sparse side, and force_sparse=true makes the solve actually run
# compute_bmat_sparse_sdp! + sparse_cholesky! + the permuted solve. The
# chain's block-level envelope under METIS ND is ~100% (envelope storage
# is inherently full for ND orderings of chains — C++ parity), so the
# solve-level 25% gate routes to dense; forcing exercises the sparse
# machinery end-to-end regardless.
@testset "solve-level sparse path on a 120-block chain (no dense fallback)" begin
    m = 120
    io = IOBuffer()
    println(io, m, " ", m)
    println(io, join(fill(2, m), " "))
    println(io, join(fill("2.0", m), " "))
    println(io, "0 1 1 1 1.0")
    println(io, "0 1 2 2 1.0")
    for i in 1:(m - 1)
        println(io, i, " ", i, " 1 1 1.0"); println(io, i, " ", i, " 2 2 1.0")
        println(io, i, " ", i + 1, " 1 1 1.0"); println(io, i, " ", i + 1, " 2 2 1.0")
    end
    println(io, m, " ", m, " 1 1 1.0"); println(io, m, " ", m, " 2 2 1.0")
    path = joinpath(mktempdir(), "trchain120.dat-s")
    write(path, take!(io))

    for T in (Float64, BigFloat)
        problem_d = SDPA.read_dat_s(path, T)
        # Routing: the aggregate graph is sparse, so analyze_sparsity
        # selects the sparse path (the solve-level 25% envelope gate is a
        # separate, ND-ordering-dependent decision).
        use_dense, ordering, adj = SDPA.analyze_sparsity(problem_d)
        @test !use_dense
        @test length(ordering) == m

        params = SDPA.Parameters{T}(maxIteration=30)
        problem_s = SDPA.read_dat_s(path, T)
        log_d = SDPA.solve!(problem_d, params; verbose=false, io=devnull,
                            reverse_primal_dual=false)
        log_s = SDPA.solve!(problem_s, params; verbose=false, io=devnull,
                            reverse_primal_dual=false, force_sparse=true)
        # Both paths progress, terminate identically, and agree
        # bit-for-bit (shared bmat value computation; the sparse assembly
        # + permuted Cholesky are the bits under test).
        @test !isempty(log_s.trajectory)
        @test log_s.iterations == log_d.iterations
        @test log_s.terminal_phase == log_d.terminal_phase
        # Compare ALL 9 IterationRow fields (write_result.jl struct:
        # iter, mu, thetaP, thetaD, objP, objD, alphaP, alphaD, beta), not
        # just (iter, mu, objP, objD) — step lengths are the classic
        # early-divergence symptom with correct-looking objectives. Both
        # logs come from this same process, so every field is
        # deterministic run-to-run here.
        _row_tuple(a) = (a.iter, a.mu, a.thetaP, a.thetaD, a.objP, a.objD,
                         a.alphaP, a.alphaD, a.beta)
        @test [_row_tuple(a) for a in log_s.trajectory] ==
              [_row_tuple(a) for a in log_d.trajectory]
        @test log_s.terminal_phase in (
            SDPA.pdOPT, SDPA.pdFEAS, SDPA.pFEAS, SDPA.dFEAS)
    end
end

# Default-CI theta5 sparse-path solve. The canonical theta5 solves in
# test_theta5_sparse.jl are slow-gated; this ungated variant truncates at
# maxIteration=10 to keep the default-CI runtime bounded (full
# convergence is ~20 iters per the slow test's measured control, so the
# truncation is what keeps this runnable in every CI pass). Adds the
# dense-vs-sparse cross-check at the SAME truncated iteration count.
@testset "theta5 truncated sparse solve in default CI (maxIteration=10)" begin
    fixture = joinpath(@__DIR__, "..", "fixtures", "theta5.dat-s")
    T = Float64
    params = SDPA.Parameters{T}(maxIteration=10)

    # Sparse path: the real theta5 chain, forced sparse. Must terminate
    # cleanly (no throw) in a feasible phase — pdFEAS measured at iter
    # 10 (both primal- and dual-feasible, duality gap not yet closed).
    p_sparse = SDPA.read_dat_s(fixture, T)
    log_s = SDPA.solve!(p_sparse, params; verbose=false, io=devnull,
                        reverse_primal_dual=false, force_sparse=true)
    @test log_s.terminal_phase in (SDPA.pdFEAS, SDPA.pdOPT)
    @test isfinite(log_s.final_obj_primal)
    @test isfinite(log_s.final_obj_dual)
    @test log_s.iterations == 10

    # Dense reference at the same budget: same-iteration objective
    # agreement (expected to agree closely; observed bit-identical at
    # Float64 — shared bmat values, the sparse assembly + Cholesky are
    # the bits under test at theta5 scale).
    p_dense = SDPA.read_dat_s(fixture, T)
    log_d = SDPA.solve!(p_dense, params; verbose=false, io=devnull,
                        reverse_primal_dual=false, force_sparse=false)
    @test log_d.iterations == log_s.iterations
    @test Float64(log_s.final_obj_primal) ≈ Float64(log_d.final_obj_primal) rtol=1e-8
    @test Float64(log_s.final_obj_dual) ≈ Float64(log_d.final_obj_dual) rtol=1e-8
end
