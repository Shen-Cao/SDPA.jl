# test/sparse/test_routing_freeze.jl
#
# Routing freeze for the aggregate-graph early gate + pair dedup.
#
# The dense/sparse routing outcome of every committed problem fixture is
# pinned here. The gate (a) skips building the position-level aggregate
# graph when the m/nBlock pre-rules of `decide_dense_or_sparse` already
# route dense (mirroring C++ sdpa_chordal.cpp:349 `m <= 100 || nBlock <= 5`),
# and (b) dedups clique pairs inside `build_aggregate_graph`. Both are
# required to leave the adjacency pattern — and therefore the routing
# outcome — bit-identical: dedup only removes multiplicities that
# `sparse`'s combine already erased, and the gate only skips work whose
# result the routing ignored. This test hardcodes the recorded outcomes
# and fails if any fixture's routing flips (the recorded outcomes are
# thread-count independent).
#
# The table records, per fixture: (m, nBlock, aggregate density,
# decide_dense_or_sparse, analyze_sparsity use_dense). Every committed
# fixture currently routes DENSE (the sparse-path fixtures are synthetic,
# built in-test by test_ordering_routing.jl / test_sparse_e2e.jl).

using SDPA
using Test

const _ROUTING_FREEZE_FIXTURES = joinpath(@__DIR__, "..", "fixtures")

# (fixture, m, nBlock, density, decide_dense, analyze_use_dense)
const _ROUTING_FREEZE = [
    ("arch0.dat-s",                174,   2, 2190 / 30276,     true, true),
    ("block_sparse.dat-s",          10,   3, 0.1,              true, true),
    ("control5.dat-s",             351,   2, 1.0,              true, true),
    ("cycle4_nonchordal.dat-s",      4,   4, 0.75,             true, true),
    ("example1.dat-s",               3,   1, 7 / 9,            true, true),
    ("f4_mixed_dense_sparse.dat-s", 18,   7, 18 / 324,         true, true),
    ("lp_first_mixed.dat-s",        18,   7, 18 / 324,         true, true),
    ("minimal_sdp_lp.dat-s",         2,   2, 1.0,              true, true),
    ("sdp6_blocks.dat-s",           18,   6, 18 / 324,         true, true),
    ("sparse_lp_profile.dat-s",      2,   3, 1.0,              true, true),
    ("theta5.dat-s",              3028,   1, 3028 / 3028^2,    true, true),
    ("example1.dat",                 3,   1, 7 / 9,            true, true),
]

@testset "routing freeze: dense/sparse outcome per fixture" begin
    for (f, m, nBlock, density_pin, decide_pin, use_dense_pin) in _ROUTING_FREEZE
        problem = endswith(f, ".dat-s") ?
            SDPA.read_dat_s(joinpath(_ROUTING_FREEZE_FIXTURES, f), Float64) :
            SDPA.read_dat(joinpath(_ROUTING_FREEZE_FIXTURES, f), Float64)
        @test problem.m == m
        @test problem.nBlock == nBlock

        # The position-level graph pattern must be unchanged:
        # same nnz, same density.
        adj = SDPA.build_aggregate_graph(problem)
        density = SDPA.compute_aggregate_density(adj)
        @test density ≈ density_pin rtol=1e-6

        # The decision rule on the (m, nBlock, density) triple — the
        # freeze target. The early gate can skip the graph but must never
        # change this answer.
        @test SDPA.decide_dense_or_sparse(m, nBlock, density) == decide_pin

        # The routing outcome observed by solve.jl (gate may return early).
        use_dense, _, _ = SDPA.analyze_sparsity(problem)
        @test use_dense == use_dense_pin
    end
end
