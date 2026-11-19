# test/sparse/test_ordering_routing.jl
#
# Unit tests for the SDPA_ORDERING env-gated ordering routing (block/auto)
# on a SYNTHETIC fixture constructed in-test.
#
# No existing fixture can reach the block-ordering branch:
#   - theta5 is nBlock=1 -> use_dense=true (the `SDPA_BLOCK_ORDERING && !use_dense`
#     gate at src/solve.jl never fires);
#   - sdp6_blocks / f4 / block_sparse all trip the m<100 dense gate.
#
# Fixture (m=100, nBlock=5, five SDP blocks of 20): constraint i has a single
# diagonal entry in block b_i = ((i-1)%5)+1 at position p_i = ((i-1)÷5)+1.
# Measured properties:
#   - aggregate density = 1%  <= 0.25  -> analyze_sparsity selects sparse
#   - position-level closed-profile density = 99.84% > 0.4 -> auto routes to
#     block per the pinned rule (block iff position-level closed-profile
#     density > 0.4; 0.4 = C++ extend_threshold parity)
#   - block-level envelope density = 20.79% <= 25% -> the sparse path
#     survives with the block ordering (the unset run's 99.84% envelope
#     exceeds 25% and takes the dense path).
#
# NOTE on the (b) assertion below: on this fixture the dense and sparse
# paths produce BIT-IDENTICAL trajectories (Float64 and BigFloat(256)
# across block sizes — the shared bmat value
# computation and matching factorization accumulation order make the
# two paths numerically identical on this well-structured problem), so
# a trajectory-difference assertion would fail vacuously. The ordering
# effect of the branch is instead asserted directly via the selected
# PATH (dense for the unset run, sparse profile for the block runs) —
# the envelope density 99.84% vs 20.79% is the ordering's observable
# consequence, and the routing signal asserts the auto-mode decision.

using SDPA
using Test

function _write_routing_fixture(dir::String)
    m = 100
    nblock = 5
    bs = 20
    io = IOBuffer()
    println(io, "$m $nblock")
    println(io, join(fill(bs, nblock), " "))
    println(io, join(fill("1.0", m), " "))
    println(io, "0 1 1 1 1.0")          # C: single diagonal entry
    for i in 1:m
        b = (i - 1) % nblock + 1
        p = (i - 1) ÷ nblock + 1
        println(io, "$i $b $p $p 1.0")  # A_i: single diagonal entry
    end
    path = joinpath(dir, "routing_fixture.dat-s")
    write(path, take!(io))
    return path
end

# Full trajectory rows (iter mu objP objD thetaP thetaD alphaP alphaD beta).
_trajectory(log) = [(r.iter, r.mu, r.objP, r.objD, r.thetaP, r.thetaD,
                     r.alphaP, r.alphaD, r.beta) for r in log.trajectory]

function _solve_with_env(path, params, env_pairs; verbose=false)
    saved = Dict(k => get(ENV, k, nothing) for (k, _) in env_pairs)
    try
        for (k, v) in env_pairs
            if v === nothing
                delete!(ENV, k)
            else
                ENV[k] = v
            end
        end
        problem = SDPA.read_dat_s(path, Float64)
        return SDPA.solve!(problem, params; verbose=verbose, io=devnull)
    finally
        for (k, v) in saved
            if v === nothing
                delete!(ENV, k)
            else
                ENV[k] = v
            end
        end
    end
end

@testset "SDPA_ORDERING env-gated routing (synthetic fixture)" begin
    fixture_dir = mktempdir()
    try
        path = _write_routing_fixture(fixture_dir)
        problem = SDPA.read_dat_s(path, Float64)
        params = SDPA.Parameters{Float64}(maxIteration=6)

        # --- Fixture property sanity (inputs to the pinned routing rule) ---
        adj = SDPA.build_aggregate_graph(problem)
        @test SDPA.compute_aggregate_density(adj) <= 0.25   # sparse decision
        ordering = SDPA.compute_ordering(adj; method=:metis)
        rev = Vector{Int}(undef, problem.m)
        for (pos, orig) in enumerate(ordering)
            rev[orig] = pos
        end
        sky = SDPA.build_skyline_from_problem!(problem, ordering, rev)
        full = problem.m * (problem.m + 1) ÷ 2
        pos_density = sum(sky[j] - j + 1 for j in 1:problem.m) / full
        @test pos_density > 0.4   # auto mode must route this fixture to block

        log_unset  = _solve_with_env(path, params,
            [("SDPA_ORDERING", nothing), ("SDPA_BLOCK_ORDERING", nothing)])
        log_block  = _solve_with_env(path, params,
            [("SDPA_ORDERING", "block"), ("SDPA_BLOCK_ORDERING", nothing)])
        log_legacy = _solve_with_env(path, params,
            [("SDPA_ORDERING", nothing), ("SDPA_BLOCK_ORDERING", "1")])

        rows_unset  = _trajectory(log_unset)
        rows_block  = _trajectory(log_block)
        rows_legacy = _trajectory(log_legacy)

        # (a) SDPA_ORDERING=block trajectory == SDPA_BLOCK_ORDERING=1 trajectory
        @test rows_block == rows_legacy
        # (b) The branch fired and changed the ordering (non-vacuity): the
        #     position envelope (99.84%) exceeds the 25% gate so the UNSET
        #     run takes the DENSE path, while the block-level envelope
        #     (20.79%) keeps the block runs on the SPARSE path. (The
        #     trajectories themselves are bit-identical across the two
        #     paths on this fixture — see the note above — so the path
        #     selection, not the trajectory, proves the ordering change.)
        logs_u, _ = Test.collect_test_logs() do
            _solve_with_env(path, params,
                [("SDPA_ORDERING", nothing), ("SDPA_BLOCK_ORDERING", nothing)];
                verbose=true)
        end
        logs_b, _ = Test.collect_test_logs() do
            _solve_with_env(path, params,
                [("SDPA_ORDERING", "block"), ("SDPA_BLOCK_ORDERING", nothing)];
                verbose=true)
        end
        logs_l, _ = Test.collect_test_logs() do
            _solve_with_env(path, params,
                [("SDPA_ORDERING", nothing), ("SDPA_BLOCK_ORDERING", "1")];
                verbose=true)
        end
        has_info(logs, needle) = any(l -> l.level == Base.CoreLogging.Info &&
                                          occursin(needle, l.message), logs)
        @test has_info(logs_u, "using dense path")
        @test has_info(logs_b, "Block-level profile")
        @test has_info(logs_l, "Block-level profile")
        # Non-vacuous: the solves progressed (non-empty trajectories)
        @test !isempty(rows_unset)
        @test !isempty(rows_block)

        # (c) SDPA_ORDERING=auto picks block per the routing rule (asserted
        # via the routing log signal)
        logs, log_auto = Test.collect_test_logs() do
            _solve_with_env(path, params,
                [("SDPA_ORDERING", "auto"), ("SDPA_BLOCK_ORDERING", nothing)];
                verbose=true)
        end
        @test has_info(logs, "ordering routing: block")
        @test _trajectory(log_auto) == rows_block

        # (d) env unset == SDPA_BLOCK_ORDERING empty (bit-identical)
        log_empty = _solve_with_env(path, params,
            [("SDPA_ORDERING", nothing), ("SDPA_BLOCK_ORDERING", "")])
        @test _trajectory(log_empty) == rows_unset
    finally
        rm(fixture_dir; recursive=true, force=true)
    end
end

@testset "SDPA_* env var validation (fail-fast before analyze_sparsity)" begin
    fixture_dir = mktempdir()
    try
        path = _write_routing_fixture(fixture_dir)
        params = SDPA.Parameters{Float64}(maxIteration=6)

        # SDPA_ORDERING with an invalid value must throw an ErrorException whose
        # message names the variable (also proves validation runs up front,
        # before analyze_sparsity's aggregate-graph + METIS work is wasted).
        err = try
            _solve_with_env(path, params, [("SDPA_ORDERING", "blok")])
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("SDPA_ORDERING", sprint(showerror, err))

        # SDPA_BLOCK_ORDERING with an invalid value must throw.
        err = try
            _solve_with_env(path, params, [("SDPA_BLOCK_ORDERING", "yes")])
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("SDPA_BLOCK_ORDERING", sprint(showerror, err))

        # SDPA_FORCE_SPARSE_PROFILE is a bench-only knob: an invalid value
        # warns (rather than erroring), and the solve still runs.
        logs, _ = Test.collect_test_logs() do
            _solve_with_env(path, params,
                [("SDPA_FORCE_SPARSE_PROFILE", "banana")])
        end
        has_warn(logs, needle) = any(l -> l.level == Base.CoreLogging.Warn &&
                                          occursin(needle, l.message), logs)
        @test has_warn(logs, "SDPA_FORCE_SPARSE_PROFILE ignored")
    finally
        rm(fixture_dir; recursive=true, force=true)
    end
end
