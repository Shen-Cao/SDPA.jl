# Shared helpers for the sparse block-index regression tests
# (test/algorithm/test_sparse_block_index.jl and
# test/threading/test_cross_thread_bit_identity.jl).
#
# Each A_i structurally carries an empty shell for every block position
# and `change_to_dense!` may leave zero entries, so the per-block
# constraint tables must be derived from the ACTUAL nonzeros of A, not
# from the shells. `initialize_index!` does exactly that (zero-nonzero
# blocks are skipped); `_rebuild_true_tables!` re-derives the per-block
# constraint tables the same way, so the density gate and the location
# table see the true structure regardless of how the tables were last
# populated.
#
# This file defines plain functions only; subprocess scripts load it via
# `include(<abs path>)`.

# _profile_from_dense lives in its own canonical file (test/_profile_from_dense.jl)
# so every consumer shares ONE definition per process — no redefinition
# warnings. Subprocess scripts include this helper file directly (fresh
# Main), so the guarded include resolves to the canonical definition.
isdefined(@__MODULE__, :_profile_from_dense) ||
    include(joinpath(@__DIR__, "..", "_profile_from_dense.jl"))

"""
    _rebuild_true_tables!(problem)

Rebuild `SDP_nConstraint`/`SDP_constraint`/`SDP_blockIndex` (and the LP
counterparts) from the stored nonzeros of `problem.A`, counting only
blocks with actual nonzeros (empty shells skipped). For freshly read
fixtures this is idempotent with `initialize_index!`; it re-derives the
tables for problems whose tables were hand-mutated or populated
otherwise. Returns the problem.
"""
function _rebuild_true_tables!(problem)
    fill!(problem.SDP_nConstraint, 0)
    for l in 1:problem.nBlock
        empty!(problem.SDP_constraint[l])
        empty!(problem.SDP_blockIndex[l])
    end
    for l in 1:length(problem.LP_constraint)
        empty!(problem.LP_constraint[l])
        empty!(problem.LP_blockIndex[l])
    end
    fill!(problem.LP_nConstraint, 0)
    for i in 1:problem.m
        Ai = problem.A[i]
        for sdp_k in 1:Ai.SDP_sp_nBlock
            blk = Ai.SDP_sp_block[sdp_k]
            nz = if blk.type === :sparse
                blk.nonZeroEffect
            else
                count(!iszero, blk.de_ele)
            end
            nz == 0 && continue
            global_k = Ai.SDP_sp_index[sdp_k]
            push!(problem.SDP_constraint[global_k], i)
            push!(problem.SDP_blockIndex[global_k], sdp_k)
            problem.SDP_nConstraint[global_k] += 1
        end
        # LP blocks: compact storage keeps only nonzeros, so iterating
        # the compact list needs no iszero filter. LP_sp_index[lp_pos]
        # is the problem-level LP scalar position (identity values).
        for lp_pos in 1:Ai.LP_nnz
            lp_k = Ai.LP_sp_index[lp_pos]
            push!(problem.LP_constraint[lp_k], i)
            push!(problem.LP_blockIndex[lp_k], lp_k)
            problem.LP_nConstraint[lp_k] += 1
        end
    end
    return problem
end
