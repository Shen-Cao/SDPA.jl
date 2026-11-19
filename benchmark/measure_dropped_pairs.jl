# benchmark/measure_dropped_pairs.jl — informational measurement (NOT a gate).
#
# Quantifies the block-level (i,j) pairs that the old position-level profile
# silently dropped for test/fixtures/block_sparse.dat-s — i.e. the magnitude
# of the Schur-matrix truncation behind the NSPS sparse divergence
# (see docs/sparse-path.md).
#
# Run (informational only):
#   JULIA_NUM_THREADS=1 JULIA_PKG_SERVER="" $JULIA --project=. benchmark/measure_dropped_pairs.jl

using SDPA
using LinearAlgebra

function main()
    fixture = joinpath(@__DIR__, "..", "test", "fixtures", "block_sparse.dat-s")
problem = SDPA.read_dat_s(fixture, BigFloat)
m = problem.m

# Mirror the solve.jl force_sparse path: Metis ordering on the position-level
# aggregate graph.
adj = SDPA.build_aggregate_graph(problem)
ordering = SDPA.compute_ordering(adj; method=:metis)
reverse_ordering = Vector{Int}(undef, m)
for (pos, orig) in enumerate(ordering)
    reverse_ordering[orig] = pos
end

blk_sky = SDPA.build_skyline_from_problem!(problem, ordering, reverse_ordering)
pos_sky = SDPA.build_skyline_from_graph!(adj, ordering, reverse_ordering)

full_nnz = m * (m + 1) ÷ 2
blk_nnz = sum(blk_sky[j] - j + 1 for j in 1:m)
pos_nnz = sum(pos_sky[j] - j + 1 for j in 1:m)

println("== block_sparse.dat-s profile counts (m=$m, full upper triangle=$full_nnz) ==")
println("block-level skyline (C++ Chordal::makeGraph connectivity): " *
        "$blk_nnz entries (density=$(round(100 * blk_nnz / full_nnz; digits=1))%)")
println("position-level skyline (OLD profile):                     " *
        "$pos_nnz entries (density=$(round(100 * pos_nnz / full_nnz; digits=1))%)")

# Dropped pairs: block-level pairs outside the position-level skyline.
dropped = 0
for l in 1:length(problem.SDP_nConstraint)
    nC = problem.SDP_nConstraint[l]
    for k1 in 1:nC, k2 in 1:nC
        i = problem.SDP_constraint[l][k1]
        j = problem.SDP_constraint[l][k2]
        ii, jj = minmax(reverse_ordering[i], reverse_ordering[j])
        if jj > pos_sky[ii]
            dropped += 1
        end
    end
end
println("block-level pairs OUTSIDE the position-level skyline " *
        "(silently dropped by the OLD make_aggrigate_index_sdp! `continue`): $dropped")

# Max |B[i,j]| over dropped pairs — F3 formula for two diagonal-position
# constraints in the same block l:
#     B[i,j] = A_i[p,p] · A_j[q,q] · invZ[p,q] · X[q,p]
# where A_i = e_p e_pᵀ, A_j = e_q e_qᵀ (fixture: every constraint has one
# diagonal nonzero per block, value 1.0).
#   * At IPM iteration 0: X = Z = lambdaStar·I are DIAGONAL → invZ[p,q] =
#     X[q,p] = 0 for p ≠ q → max dropped-pair |B| = 0. This is the audit's
#     "validation gate trap": an iteration-0 measurement would falsely pass.
#   * From iteration 1 onward X and Z⁻¹ are dense PD → the term is
#     generically nonzero: B is block-level dense.
lambdaStar = SDPA.Parameters{BigFloat}(precision=200).lambdaStar
println("\n== max |B[i,j]| over dropped pairs (F3 formula, informational) ==")
println("iteration 0 (X = Z = lambdaStar·I = $(lambdaStar)·I): max |B| = 0 " *
        "<- audit 'validation gate trap' (diagonal slacks)")
# Illustration with a generic dense PD iterate: X = Z⁻¹ = Hilbert(10).
H = [inv(BigFloat(i + j - 1)) for i in 1:10, j in 1:10]
maxB = zero(BigFloat)
maxpair = (0, 0)
for l in 1:length(problem.SDP_nConstraint)
    nC = problem.SDP_nConstraint[l]
    for k1 in 1:nC, k2 in 1:nC
        i = problem.SDP_constraint[l][k1]
        j = problem.SDP_constraint[l][k2]
        ii, jj = minmax(reverse_ordering[i], reverse_ordering[j])
        jj <= pos_sky[ii] && continue  # only DROPPED pairs
        ib = problem.SDP_blockIndex[l][k1]
        jb = problem.SDP_blockIndex[l][k2]
        p = problem.A[i].SDP_sp_block[ib].row_index[1]
        q = problem.A[j].SDP_sp_block[jb].row_index[1]
        av = problem.A[i].SDP_sp_block[ib].sp_ele[1]
        bv = problem.A[j].SDP_sp_block[jb].sp_ele[1]
        val = abs(av * bv * H[p, q] * H[q, p])
        if val > maxB
            maxB = val
            maxpair = (i, j)
        end
    end
end
println("generic dense iterate (X = Z⁻¹ = Hilbert(10), dense PD): " *
        "max |B| = $(Float64(maxB)) (constraint pair $maxpair)")
println("\nInterpretation: the old position-level profile dropped $dropped pairs " *
        "whose F3 contributions are structurally ZERO at iteration 0 but " *
        "generically O(1) once X, Z⁻¹ densify — the truncated Schur system " *
        "behind the NSPS sparse divergence (see docs/sparse-path.md).")
end

main()
