# Canonical test-local construction of a full upper-triangle profile-form
# sparse matrix from a dense `A` + `ordering`, via the LIVE sparse-path
# pipeline (profile_from_skyline! + _fill_sparse_values!).
#
# Shared by test/algorithm/test_SchurComplement.jl,
# test/sparse/test_sparse_cholesky.jl, and (via
# test/algorithm/_sparse_block_index_helpers.jl) the sparse block-index
# subprocess scripts. Kept in its own file so the function is defined
# exactly ONCE per process — the three consumers include this file behind
# an `isdefined` guard, avoiding method-redefinition warnings.
function _profile_from_dense(A::AbstractMatrix{T}, ordering) where {T<:Real}
    m = size(A, 1)
    sb, di = SDPA.profile_from_skyline!(fill(m, m), T)
    SDPA._fill_sparse_values!(sb, A, ordering, di)
    return sb, di
end
