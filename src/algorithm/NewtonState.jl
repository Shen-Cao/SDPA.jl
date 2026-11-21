"""
    NewtonState{T<:Real}

Cumulative state for one Newton iteration of the IPM. Defined here,
before the Schur-complement functions that write into its fields.
Mirrors C++ `sdpa::Newton` at
`sdpa_newton.h:49-106`.

Fields
-----
- `bmat::Union{Nothing,SDPADenseMatrix{T}}`: Schur complement matrix
  (dense; the sparse path uses the separate `sparse_bmat` field on the
  same struct). `nothing` until the dense
  path actually runs — the forced-sparse path never needs it, so it is
  allocated lazily via `ensure_dense_bmat!` (the default constructor
  still allocates it eagerly; construct with `allocate_dense_bmat=false`
  to defer).
- `gvec::SDPAVector{T}`:                      Schur RHS vector
- `DxMat::SDPADenseLinearSpace{T}`:           predictor/corrector Δx
- `DyVec::SDPAVector{T}`:                     Δy
- `DzMat::SDPADenseLinearSpace{T}`:           Δz
- `r_zinvMat::SDPADenseLinearSpace{T}`:       `R · Z⁻¹` where
  R = βμI - X·Z (predictor) or βμI - X·Z - Δx·Δz (corrector)
- `x_rd_zinvMat::SDPADenseLinearSpace{T}`:    cached `X · dual_residual · Z⁻¹`
  (computed once in the predictor, reused in the corrector; see `make_gvec!`)
- `useFormula::Matrix{Symbol}`:               per-(constraint, SDP block)
  formula selection: `:F1`, `:F2`, or `:F3` (selected in `compute_formula!`).
  Indexed `useFormula[i, l]` with `l` the PROBLEM-block index (the index
  in scope at every consumer: compute_bmat_dense!, compute_bmat_sparse_sdp!,
  threaded_compute_bmat_sparse_sdp!, threaded_compute_bmat_dense!).
  Mirrors C++ `Newton::useFormula[i * SDP_nBlock + l]` (sdpa_newton.h:65,
  sdpa_newton.cpp:75) — one formula per (constraint, SDP block), so a
  dense block can never inherit a sparse block's F3 choice.
- `cholesky_factorization::Union{Cholesky{T, Matrix{T}}, Nothing}`:
  cached Schur Cholesky, reused from predictor to corrector. `nothing`
  until the predictor factorizes it (and on Cholesky failure).

SOCP-related fields are skipped (C++ has them, we don't need them).
Sparse-related fields (`sparse_bmat`, `ordering`,
`reverse_ordering`, `use_sparse`) ARE present in the struct as
placeholder fields (initialized to `nothing` / `Int[]` / `false`).
They are populated only when `analyze_sparsity` selects the sparse
path; on the dense path they remain at their default values.
"""
mutable struct NewtonState{T<:Real}
    bmat::Union{Nothing,SDPADenseMatrix{T}}
    gvec::SDPAVector{T}
    DxMat::SDPADenseLinearSpace{T}
    DyVec::SDPAVector{T}
    DzMat::SDPADenseLinearSpace{T}
    r_zinvMat::SDPADenseLinearSpace{T}
    x_rd_zinvMat::SDPADenseLinearSpace{T}
    useFormula::Matrix{Symbol}
    cholesky_factorization::Union{Cholesky{T,Matrix{T}},Nothing}
    bordered::Union{Nothing,_BorderedSchur{T}}
    # Sparse Schur complement fields
    sparse_bmat::Union{Nothing,SDPASparseMatrix{T},_SchurSkyline{T}}
    # Only used by explicit-coordinate callers. Production _SchurSkyline
    # owns its validated offsets and does not duplicate them here.
    diagonalIndex::Vector{Int}
    ordering::Vector{Int}
    reverse_ordering::Vector{Int}
    use_sparse::Bool
    # Sparse location-table fields. Parallel arrays
    # indexed by t in 1:total_entries. C++ counterpart at
    # sdpa_newton.h:80-84.
    SDP_constraint1::Vector{Int}
    SDP_constraint2::Vector{Int}
    blockIndex1::Vector{Int}
    blockIndex2::Vector{Int}
    SDP_location_sparse_bMat::Vector{Int}
    # Per-SDP-block lengths (so compute_bmat_sparse_sdp! can iterate
    # block-by-block). Length = problem.nBlock. C++ has this as the
    # `int* SDP_number` field on the Newton class.
    SDP_block_table_len::Vector{Int}
    # Permutation scratch buffers: permute gvec → sparse solve
    # → reverse-permute DyVec. Sized to m once per problem.
    _g_perm_scratch::Vector{T}
    _dy_perm_scratch::Vector{T}
    # Reusable Int work vector for compute_formula!'s upNonZeroCount
    # accumulation. Sized to m once per problem.
    _up_counts_scratch::Vector{Int}
    # Threaded sparse-assembly scratch: per-entry value buffers for
    # threaded_compute_bmat_sparse_sdp! (sized to the location-table
    # total_entries) and threaded_compute_bmat_sparse_lp! (sized to the
    # LP pair count). BigFloat slots must be independent AND valid — the
    # compute phase overwrites every slot on every call, so each buffer is
    # allocated once per problem and reused; reallocated lazily if the
    # location table / LP pair count resizes. Same idiom as the
    # _g_perm_scratch/_dy_perm_scratch caches above.
    _sdp_all_values::Vector{T}
    _lp_all_values::Vector{T}
    # Cache: per-LP-scalar constraint lists (see
    # `_lp_constraint_lists!` in SchurComplement.jl). Value-independent
    # for the whole solve (built from `A[i].LP_sp_index`, which never
    # changes), so computed once on first use and reused across every
    # make_bmat! call. Empty until first use — the consumers only call
    # the accessor when LP_nBlock > 0, so `isempty` is a safe
    # "not yet computed" sentinel.
    _lp_clists::Vector{Vector{Int}}
end

"""
    NewtonState{T}(m::Int, blockStruct::AbstractVector{Int};
                   allocate_dense_bmat::Bool=true) where T

Allocate the Newton state for an `m`-constraint problem with the given
signed `blockStruct`. All matrix/vector fields are zero-initialized;
`cholesky_factorization = nothing` until the predictor step computes it.

The dense Schur-complement matrix `bmat` is allocated eagerly by
default. The forced-sparse path never uses it, so pass
`allocate_dense_bmat=false` to defer the allocation (an m×m 256-bit
BigFloat matrix can reach hundreds of GB for large m);
`ensure_dense_bmat!` then allocates it on demand if the solver later
switches to the dense path.
"""
function NewtonState{T}(
    m::Int, blockStruct::AbstractVector{Int};
    allocate_dense_bmat::Bool=true,
) where {T<:Real}
    m >= 0 || error("m must be non-negative, got $m")
    return NewtonState{T}(
        allocate_dense_bmat ? SDPADenseMatrix{T}(m, m) : nothing, # bmat (lazy)
        SDPAVector{T}(m),                                       # gvec
        SDPADenseLinearSpace{T}(blockStruct),                  # DxMat
        SDPAVector{T}(m),                                       # DyVec
        SDPADenseLinearSpace{T}(blockStruct),                  # DzMat
        SDPADenseLinearSpace{T}(blockStruct),                  # r_zinvMat
        SDPADenseLinearSpace{T}(blockStruct),                  # x_rd_zinvMat
        fill(:F1, m, length(blockStruct)),                   # useFormula default (per (i, l))
        nothing,                                                # cholesky_factorization
        nothing,                                                # bordered
        nothing,                                                # sparse_bmat
        Int[],                                                  # diagonalIndex
        Int[],                                                  # ordering
        Int[],                                                  # reverse_ordering
        false,                                                  # use_sparse
        Int[],                                                  # SDP_constraint1
        Int[],                                                  # SDP_constraint2
        Int[],                                                  # blockIndex1
        Int[],                                                  # blockIndex2
        Int[],                                                  # SDP_location_sparse_bMat
        Int[],                                                  # SDP_block_table_len
        Vector{T}(undef, m),                                    # _g_perm_scratch
        Vector{T}(undef, m),                                    # _dy_perm_scratch
        zeros(Int, m),                                          # _up_counts_scratch
        T[],                                                    # _sdp_all_values (lazy)
        T[],                                                    # _lp_all_values (lazy)
        Vector{Vector{Int}}(),                                  # _lp_clists (lazy)
    )
end

"""
    ensure_dense_bmat!(newton_state::NewtonState{T}) -> SDPADenseMatrix{T}

Return `newton_state.bmat`, allocating it on first use if the state was
constructed with `allocate_dense_bmat=false`. The allocation matches the
default constructor exactly: `SDPADenseMatrix{T}(m, m)` with `m` the
state's constraint count (the same `m` the constructor sized `gvec` to).
No-op (allocation-free) when the bmat is already present — the dense
Schur-complement consumers call this before accessing `bmat.de_ele`, so
the forced-sparse path (which never calls them) keeps `bmat === nothing`.
"""
function ensure_dense_bmat!(newton_state::NewtonState{T}) where {T<:Real}
    bmat = newton_state.bmat
    if bmat === nothing
        m = length(newton_state.gvec.ele)
        bmat = SDPADenseMatrix{T}(m, m)
        newton_state.bmat = bmat
    end
    return bmat
end
