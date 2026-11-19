# Regression test: the sparse path must assemble the FULL block-level
# Schur profile (matching C++ Chordal::makeGraph), never silently dropping
# block-level (i,j) pairs outside a position-level skyline.
#
# Fixture: test/fixtures/block_sparse.dat-s (m=10, 3 SDP blocks of 10×10,
# optimum 165). Every constraint touches every block at ONE scattered
# diagonal position, so:
#   * position-level graph (build_aggregate_graph) is DISJOINT → the old
#     position-level profile is diagonal-only (10 entries);
#   * block-level graph is COMPLETE → 135 off-diagonal block-level pairs
#     were SILENTLY DROPPED by the old make_aggrigate_index_sdp! `continue`,
#     truncating B → NSPS-class divergence.
#
# Design choices:
# ---------------------------------------------------------------------------
# (a) Path assertion uses @test_logs with match_mode=:any. The "Using sparse
#     Schur complement path (m=...)" @info in solve.jl goes to the LOGGER
#     (not io) and proves the solver entered the sparse branch of the
#     dispatch under force_sparse=true. match_mode=:any is REQUIRED because
#     block_sparse's block-level graph is complete by construction (skyline
#     density 100% > 25%), so the UNCHANGED density gate in solve.jl emits
#     additional @info records ("Block-level skyline: ...",
#     "Block-level density >25% ...") and legitimately falls back to the
#     dense path — exactly as C++ SDPA-GMP does (sdpa_chordal.cpp:
#     349: m=10 <= m_threshold=100 → dense). With the default match_mode=:all
#     the extra gate records make @test_logs fail regardless, so it cannot
#     serve as the discriminator.
# (b) The contract check below: calling make_aggrigate_index_sdp! with a
#     POSITION-level profile (the profile the OLD solve.jl built) must be
#     REJECTED LOUDLY (error(), C++ parity with rError "program bug" at
#     sdpa_newton.cpp:481) instead of silently dropping the 135
#     off-diagonal block pairs. The block-level profile then covers every
#     block pair (the error never fires) — verified directly.
# ---------------------------------------------------------------------------

using SDPA
using Test

# Parse the C++ 200-bit golden objective from test/fixtures/golden/block_sparse.json
# (flat file; no JSON dependency in the project). Compare NUMERICALLY at
# rtol=1e-30 — never as strings.
function _golden_cpp_primal(path::String)
    raw = read(path, String)
    m = match(r"\"cpp_obj_primal_full\"\s*:\s*\"([^\"]+)\"", raw)
    m === nothing && error("golden file missing cpp_obj_primal_full: $path")
    return BigFloat(m.captures[1])
end

@testset "block-level sparse profile covers every block pair (block_sparse)" begin
    fixture = joinpath(@__DIR__, "..", "fixtures", "block_sparse.dat-s")
    golden_path = joinpath(@__DIR__, "..", "fixtures", "golden", "block_sparse.json")
    @test isfile(fixture)
    @test isfile(golden_path)

    params = SDPA.Parameters{BigFloat}(precision=200)
    rtol = BigFloat("1e-30")

    # (a) Sparse-branch entry: force_sparse=true must enter the sparse
    # branch of the dispatch (the "Using sparse..." @info in solve.jl).
    # match_mode=:any — see the design note in the file header.
    p_entry = SDPA.read_dat_s(fixture, BigFloat)
    @test_logs (:info, r"Using sparse") match_mode=:any SDPA.solve!(
        p_entry, params; verbose=true, io=devnull, force_sparse=true)

    # (c) Sparse-invocation objective == dense-invocation objective at
    # rtol 1e-30 (200-bit). (Block-level density gate falls back to dense
    # for this fixture in BOTH invocations — see header note; the
    # correctness teeth are the contract checks below.)
    p_sparse = SDPA.read_dat_s(fixture, BigFloat)
    log_sparse = SDPA.solve!(p_sparse, params; verbose=false, io=devnull,
                             force_sparse=true)
    p_dense = SDPA.read_dat_s(fixture, BigFloat)
    log_dense = SDPA.solve!(p_dense, params; verbose=false, io=devnull,
                            force_sparse=false)
    @test log_sparse.terminal_phase == SDPA.pdOPT
    @test log_dense.terminal_phase == SDPA.pdOPT
    @test isapprox(log_sparse.final_obj_primal, log_dense.final_obj_primal; rtol=rtol)
    @test isapprox(log_sparse.final_obj_dual, log_dense.final_obj_dual; rtol=rtol)

    # (d) Objective matches the C++ golden at rtol 1e-30 (numeric compare).
    golden_obj = _golden_cpp_primal(golden_path)
    @test isapprox(log_sparse.final_obj_primal, golden_obj; rtol=rtol)

    # ------------------------------------------------------------------
    # CONTRACT: a position-level profile must be rejected loudly
    # ------------------------------------------------------------------
    # Mirror the force_sparse path in solve.jl: Metis ordering on the
    # position-level aggregate graph.
    adj = SDPA.build_aggregate_graph(p_sparse)
    ordering = SDPA.compute_ordering(adj; method=:metis)
    reverse_ordering = Vector{Int}(undef, p_sparse.m)
    for (pos, orig) in enumerate(ordering)
        reverse_ordering[orig] = pos
    end

    # Position-level skyline — the profile the OLD solve.jl built.
    pos_skyline = SDPA.build_skyline_from_graph!(adj, ordering, reverse_ordering)
    # Block-level skyline — the density-gate input in solve.jl and the
    # profile input.
    blk_skyline = SDPA.build_skyline_from_problem!(
        p_sparse, ordering, reverse_ordering)

    full_nnz = p_sparse.m * (p_sparse.m + 1) ÷ 2
    # Fixture structure: block-level skyline is COMPLETE (55 = m(m+1)/2),
    # position-level skyline is diagonal-only (10) → 135 off-diagonal
    # block-level pairs lie outside the old profile.
    @test sum(blk_skyline[j] - j + 1 for j in 1:p_sparse.m) == full_nnz
    @test sum(pos_skyline[j] - j + 1 for j in 1:p_sparse.m) == p_sparse.m

    # NEGATIVE side: a position-level profile (old solve.jl behavior) must
    # FAIL LOUDLY — error() (C++ parity: "program bug" rError at
    # sdpa_newton.cpp:481) — not silently drop block-level pairs.
    ns_pos = SDPA.NewtonState{BigFloat}(p_sparse.m, p_sparse.blockStruct)
    ns_pos.ordering = ordering
    ns_pos.reverse_ordering = reverse_ordering
    sp_pos, di_pos = SDPA.profile_from_skyline!(pos_skyline, BigFloat)
    ns_pos.sparse_bmat = sp_pos
    ns_pos.diagonalIndex = di_pos
    @test_throws ErrorException SDPA.make_aggrigate_index_sdp!(
        ns_pos, p_sparse, ordering)

    # POSITIVE side: the block-level profile covers EVERY block-level pair —
    # the profile-bound error never fires. Upper-triangle skip semantics
    # unchanged
    # (inz<jnz / tie i<j, mirroring C++ sdpa_newton.cpp:441-442): with
    # nonZeroEffect == 1 for all (constraint, block) entries, the kept
    # pairs per block are those with i >= j → 10·11/2 = 55, × 3 blocks = 165.
    ns_blk = SDPA.NewtonState{BigFloat}(p_sparse.m, p_sparse.blockStruct)
    ns_blk.ordering = ordering
    ns_blk.reverse_ordering = reverse_ordering
    sp_blk, di_blk = SDPA.profile_from_skyline!(blk_skyline, BigFloat)
    ns_blk.sparse_bmat = sp_blk
    ns_blk.diagonalIndex = di_blk
    SDPA.make_aggrigate_index_sdp!(ns_blk, p_sparse, ordering)
    @test length(ns_blk.SDP_constraint1) == 3 * full_nnz
    @test sum(ns_blk.SDP_block_table_len) == length(ns_blk.SDP_constraint1)
    @test all(1 .<= ns_blk.SDP_location_sparse_bMat .<= (di_blk[end] - 1))
end
