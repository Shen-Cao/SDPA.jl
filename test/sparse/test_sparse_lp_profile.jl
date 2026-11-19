# LP-pair coverage regression test: the block-level sparse profile must
# include constraint pairs that share an LP scalar but NO SDP block.
#
# build_skyline_from_problem! must merge the per-SDP-block constraint
# lists (C++ Chordal::makeGraph SDP merge, sdpa_chordal.cpp:189-195) AND
# the LP constraint lists, while
# _compute_bmat_sparse_lp! silently `continue`s past any (i,j) pair
# outside the profile. On a mixed problem an LP pair with disjoint SDP
# blocks must have its LP Schur contribution assembled, or the Schur
# matrix is truncated. C++ merges the LP lists in makeGraph
# (sdpa_chordal.cpp:204-211) and rErrors "program bug" on a missing pair
# (sdpa_newton.cpp:677).
#
# Fixture: test/fixtures/sparse_lp_profile.dat-s (m=2, blockStruct
# [2, 2, -1]). A_1 touches SDP block 1 + the LP scalar; A_2 touches SDP
# block 2 + the LP scalar: the (1,2) pair is LP-only.
#
# All sparse-vs-dense comparisons build the profile from the ACTUAL
# block-level skyline (profile_from_skyline!, the profile solve.jl
# installs) — NOT the full-profile idiom used by
# test/algorithm/test_sparse_block_index.jl, which would
# mask the truncation locked in here — and materialize the sparse bmat
# into a full m×m matrix (zero outside the profile) for the comparison,
# so a pair missing from the profile shows up as a VALUE mismatch, not a
# silently shortened vector.

using SDPA
using Test

const _FIXTURE_DIR = joinpath(dirname(@__DIR__), "fixtures")

# Assemble sparsely with the block-level-skyline profile (as solve.jl
# does), assemble densely, and return (sparse full m×m matrix, dense
# bmat de_ele, skyline). Identity ordering.
function _sparse_vs_dense_skyline(T, problem)
    ordering = collect(1:problem.m)
    reverse_ordering = collect(1:problem.m)
    skyline = SDPA.build_skyline_from_problem!(
        problem, ordering, reverse_ordering)

    params = SDPA.Parameters{T}()
    current_pt = SDPA.SDPASolution{T}(
        problem.m, problem.blockStruct, params.lambdaStar)
    work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)

    ns_s = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
    SDPA.compute_formula!(ns_s, problem)
    ns_s.use_sparse = true
    ns_s.ordering = ordering
    ns_s.reverse_ordering = reverse_ordering
    sb, di = SDPA.profile_from_skyline!(skyline, T)
    ns_s.sparse_bmat = sb
    ns_s.diagonalIndex = di
    SDPA.make_aggrigate_index_sdp!(ns_s, problem, ordering)
    SDPA.make_bmat!(ns_s, problem, current_pt, work)

    # Materialize the profile-form sparse bmat into a full symmetric
    # matrix (identity ordering: slot (row_index=column j, column_index=
    # row i) holds B[i,j]).
    full = zeros(T, problem.m, problem.m)
    for idx in eachindex(sb.sp_ele)
        i = sb.column_index[idx]
        j = sb.row_index[idx]
        full[i, j] = sb.sp_ele[idx]
        full[j, i] = sb.sp_ele[idx]
    end

    ns_d = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
    SDPA.compute_formula!(ns_d, problem)
    SDPA.make_bmat!(ns_d, problem, current_pt, work)  # dense path
    return full, ns_d.bmat.de_ele, skyline
end

# ---------------------------------------------------------------------------
# The LP-only (1,2) pair must be covered by the block-level skyline and
# its LP contribution assembled.
# ---------------------------------------------------------------------------
@testset "LP-only pair covered by the block-level skyline" begin
    T = BigFloat
    problem = SDPA.read_dat_s(
        joinpath(_FIXTURE_DIR, "sparse_lp_profile.dat-s"), T)
    @test problem.blockStruct == [2, 2, -1]
    @test problem.LP_nConstraint == [2]           # both share the LP scalar
    @test problem.SDP_nConstraint[1:2] == [1, 1]  # disjoint SDP blocks

    ordering = collect(1:problem.m)
    reverse_ordering = collect(1:problem.m)
    skyline = SDPA.build_skyline_from_problem!(
        problem, ordering, reverse_ordering)

    # The LP-only pair (1,2) must extend the skyline
    # (the SDP blocks contribute no pairs).
    @test skyline[1] == 2

    # Sparse assembly with the block-level profile == dense bmat at
    # rtol 1e-30 (a skipped LP contribution at (1,2) would leave
    # B[1,2] = 0 in the materialized sparse bmat → mismatch).
    sp_full, dense, _ = _sparse_vs_dense_skyline(T, problem)
    @test isapprox(sp_full, dense; rtol=BigFloat("1e-30"))
end

# ---------------------------------------------------------------------------
# Guard: existing mixed and SDP-only fixtures — sparse == dense
# with the block-level-skyline profile (no behavior change: none of them
# has an LP pair).
# ---------------------------------------------------------------------------
@testset "No-regression: mixed and SDP-only fixtures sparse == dense" begin
    for T in (BigFloat, Float64)
        for name in ("minimal_sdp_lp", "lp_first_mixed", "example1",
                     "sdp6_blocks")
            problem = SDPA.read_dat_s(
                joinpath(_FIXTURE_DIR, "$name.dat-s"), T)
            sp_full, dense, _ = _sparse_vs_dense_skyline(T, problem)
            rtol = T === BigFloat ? BigFloat("1e-30") : 1e-12
            @test isapprox(sp_full, dense; rtol=rtol)
        end
    end
end
