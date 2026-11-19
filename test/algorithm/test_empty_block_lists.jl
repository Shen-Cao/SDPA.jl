# Regression test: `initialize_index!` must skip zero-nonzero
# (empty) blocks when building the per-block constraint lists.
#
# C++ parity: SparseLinearSpace keeps only blocks with
# `SDP_NonZeroNumber[l] > 0` / `LP_NonZeroNumber[l] == true`
# (sdpa_struct.cpp:885-912, 948-972), and the reader records a block
# only when it carries a value != 0 (sdpa_io.cpp:539, 550). The Julia
# reader allocates EVERY block as an empty shell (read_dat_s.jl
# make_empty_space); without the skip, `initialize_index!` would list
# every constraint in every block:
#   * block-level skyline / density overestimated → spurious dense-path
#     fallback (the density gate always saw a complete skyline);
#   * constraint lists larger than C++ (block-level (i,j) pairs that
#     C++ never emits).

using SDPA
using Test

const _FIXTURE_DIR = joinpath(dirname(@__DIR__), "fixtures")

# ---------------------------------------------------------------------------
# Synthetic for the density-decision case (see the testset below).
#
# m=100, 6 SDP blocks of size 20. Constraint i touches block
# (i-1)÷20+1 at TWO diagonal positions: the DISTINCT position
# (i-1)%20+1 (independent A_i → nonsingular Schur matrix) and the shared
# position (1,1) (each block's 20 constraints form a clique in the
# position-level graph → METIS keeps the blocks contiguous, so the
# block-level skyline density is the TRUE ~20.8%, not an artifact of a
# disconnected graph). Block 6 is EMPTY in every A_i AND in C
# (completely unused). C carries E11 in blocks 1..5, b = 2.
#
# True structure: position-level density 20%, block-level skyline density
# ~20.8% (≤ 25% → sparse). With inflated lists: every block lists all 100
# constraints → complete skyline → 100% → dense path.
#
# NOTE: the dual optimum lies on the boundary (Z* has a zero
# eigenvalue), so sparse-vs-dense trajectories may split near
# termination — the agreement assertions below use a tolerance
# appropriate for that degeneracy (the decision flip is the point).
# ---------------------------------------------------------------------------
function _make_empty_block_synth(T)
    m, nBlock, nSDP = 100, 6, 6
    blockStruct = fill(20, nBlock)
    b = SDPA.SDPAVector{T}(fill(T(2), m))
    C = SDPA.SDPASparseLinearSpace{T}(nSDP, 0)
    A = [SDPA.SDPASparseLinearSpace{T}(nSDP, 0) for _ in 1:m]
    # Allocate every block (mirror the reader's make_empty_space).
    for S in (C, A...)
        for k in 1:nSDP
            S.SDP_sp_block[k] = SDPA.SDPASparseMatrix{T}(20, 20, :sparse, 0)
        end
    end
    for blk in 1:5   # C touches only the used blocks
        SDPA.setelement_sdp!(C, blk, 1, 1, one(T))
    end
    for i in 1:m
        blk = (i - 1) ÷ 20 + 1
        r = (i - 1) % 20 + 1
        SDPA.setelement_sdp!(A[i], blk, r, r, one(T))
        SDPA.setelement_sdp!(A[i], blk, 1, 1, one(T))
    end
    return SDPA.SDPAProblem{T}(m, nBlock, blockStruct, b, C, A)
end

# Inflate the constraint lists to the fully-connected semantics (every
# constraint in every block). Used to force the dense path on the SAME
# problem.
function _inflate_lists!(problem)
    for l in 1:problem.nBlock
        problem.SDP_nConstraint[l] = problem.m
        problem.SDP_constraint[l] = collect(1:problem.m)
        problem.SDP_blockIndex[l] = fill(l, problem.m)
    end
    return problem
end

# ---------------------------------------------------------------------------
# Case 1: unused SDP block — excluded from SDP_constraint and the skyline.
# ---------------------------------------------------------------------------
@testset "unused SDP block excluded from constraint lists and skyline" begin
    # sdp6_blocks: 6 SDP blocks (3×3), each genuinely touched by 3
    # constraints.
    problem = SDPA.read_dat_s(joinpath(_FIXTURE_DIR, "sdp6_blocks.dat-s"), BigFloat)
    @test problem.SDP_nConstraint == [3, 3, 3, 3, 3, 3]

    # Empty block 3 in every A_i; keep it only in C (objective-only block).
    for i in 1:problem.m
        Ai = problem.A[i]
        for sdp_k in 1:Ai.SDP_sp_nBlock
            if Ai.SDP_sp_index[sdp_k] == 3
                SDPA.setzero!(Ai.SDP_sp_block[sdp_k])
            end
        end
    end
    SDPA.setelement_sdp!(problem.C, 3, 1, 1, one(BigFloat))
    SDPA.initialize_index!(problem)

    @test problem.SDP_nConstraint[3] == 0
    @test isempty(problem.SDP_constraint[3])
    @test isempty(problem.SDP_blockIndex[3])
    @test problem.SDP_nConstraint[1] == 3          # used blocks unaffected
    @test problem.SDP_nConstraint[6] == 3

    # Block-level skyline (build_skyline_from_problem!): block 3 couples
    # NO constraint pairs, so the skyline is no longer complete.
    adj = SDPA.build_aggregate_graph(problem)
    ordering = SDPA.compute_ordering(adj; method=:metis)
    reverse_ordering = Vector{Int}(undef, problem.m)
    for (pos, orig) in enumerate(ordering)
        reverse_ordering[orig] = pos
    end
    sky = SDPA.build_skyline_from_problem!(problem, ordering, reverse_ordering)
    full_nnz = problem.m * (problem.m + 1) ÷ 2
    @test sum(sky[j] - j + 1 for j in 1:problem.m) < full_nnz
end

# ---------------------------------------------------------------------------
# Case 2: unused LP scalar — value-based skip (LP entries are scalars).
# ---------------------------------------------------------------------------
@testset "zero-valued LP scalar excluded from constraint lists" begin
    # minimal_sdp_lp: m=2, blockStruct=[2, -1] (one 2×2 SDP block + one
    # LP scalar). The LP scalar is nonzero only in A_1.
    problem = SDPA.read_dat_s(joinpath(_FIXTURE_DIR, "minimal_sdp_lp.dat-s"), BigFloat)
    @test problem.LP_nConstraint == [1]

    # Zero the only nonzero LP coefficient → empty LP block.
    # (setelement_lp! with a zero value removes the compact entry.)
    SDPA.setelement_lp!(problem.A[1], 1, BigFloat(0))
    SDPA.initialize_index!(problem)

    @test problem.LP_nConstraint[1] == 0
    @test isempty(problem.LP_constraint[1])
    @test isempty(problem.LP_blockIndex[1])
end

# ---------------------------------------------------------------------------
# Case 3: density decision — the gate must see the true (non-inflated)
# block-level density and take the sparse path; the sparse-path result
# must equal the dense-path result on the same problem.
# ---------------------------------------------------------------------------
@testset "density gate sees true block density (empty blocks excluded)" begin
    m = 100
    p_sparse = _make_empty_block_synth(Float64)
    SDPA.initialize_index!(p_sparse)
    @test p_sparse.SDP_nConstraint == [20, 20, 20, 20, 20, 0]

    # Gate math, exactly as the density gate in solve.jl (METIS ordering
    # from the position-level graph, then the block-level skyline
    # density).
    adj = SDPA.build_aggregate_graph(p_sparse)
    ordering = SDPA.compute_ordering(adj; method=:metis)
    reverse_ordering = Vector{Int}(undef, m)
    for (pos, orig) in enumerate(ordering)
        reverse_ordering[orig] = pos
    end
    block_skyline = SDPA.build_skyline_from_problem!(
        p_sparse, ordering, reverse_ordering)
    full_nnz = m * (m + 1) ÷ 2
    block_density = sum(block_skyline[j] - j + 1 for j in 1:m) / full_nnz
    @test block_density <= 0.25

    # Solve-level decision: the sparse branch is actually taken (the
    # "Block-level profile" @info in solve.jl; the dense branch would log
    # "Block-level density >25%" instead).
    params = SDPA.Parameters{Float64}(lambdaStar=1.0)
    log_sparse = @test_logs (:info, r"Block-level profile") match_mode=:any SDPA.solve!(
        p_sparse, params; verbose=true, io=devnull)

    # No-regression: the inflated lists force the dense path on the SAME
    # problem; the outcome agrees at solver tolerance (the dual
    # optimum is on the boundary, so trajectories may split near
    # termination — cf. the degenerate-note above).
    p_dense = _make_empty_block_synth(Float64)
    SDPA.initialize_index!(p_dense)
    _inflate_lists!(p_dense)
    log_dense = SDPA.solve!(p_dense, params; verbose=false, io=devnull)
    @test log_sparse.terminal_phase == log_dense.terminal_phase
    @test isapprox(log_sparse.final_obj_primal, log_dense.final_obj_primal; rtol=1e-4)
    @test isapprox(log_sparse.final_obj_dual, log_dense.final_obj_dual; rtol=1e-4)
end
