"""
    SDPASparseLinearSpace{T<:Real}

The "constant matrix side" of an SDP, partitioned into SDP/SOCP/LP blocks
and stored in sparse form (per-block `SDPASparseMatrix{T}`).

Mirrors C++ `sdpa::SparseLinearSpace` at
`sdpa_struct.h:135-172`. The C++ class is used as the
canonical form for problem data (constant matrix C and constraint
matrices {A_i}); the dense variant is used for solution iterates where
sparsity has been collapsed.

The SDPA-GMP source has SOCP support but it is dead code (`#if 0` in
`sdpa_io.cpp`). SDPA.jl does not support SOCP, but the SOCP fields are
preserved as zero-length vectors for type-layout parallelism with the
C++ so a side-by-side read stays readable.

Fields
------
- `SDP_sp_nBlock::Int`:                          SDP block count
- `SOCP_sp_nBlock::Int`:                         always 0
- `LP_sp_nBlock::Int`:                           LP scalar count (the
  problem-level LP dimension; equals the number of scalar LP variables,
  `nLP_total`). Never shrinks — solutions and work arrays index
  `LP_block` / `LP_sp_index` against it.
- `SDP_sp_index::Vector{Int}`:                   SDP block indices (into
  the blockStruct vector of the original problem; same as 1:SDP_sp_nBlock
  for typical contiguous layouts)
- `SOCP_sp_index::Vector{Int}`:                  empty
- `LP_sp_index::Vector{Int}`:                    compact LP block indices:
  the problem-level LP scalar positions of the stored nonzeros, kept
  sorted ascending (matching C++ `SparseLinearSpace::LP_sp_index` at
  sdpa_struct.cpp:959-972)
- `SDP_sp_block::Vector{SDPASparseMatrix{T}}`:   one sparse matrix per
  SDP block
- `SOCP_sp_block::Vector{SDPASparseMatrix{T}}`:  empty (SOCP unused)
- `LP_nnz::Int`:                                 number of stored nonzero
  LP entries (compact LP storage: ONLY nonzeros are kept)
- `LP_sp_val::Vector{T}`:                        values parallel to
  `LP_sp_index` (kept sorted by index, so a lookup is a binary search)

The LP storage is compact per-space (mirrors C++ `SparseLinearSpace`:
`LP_NonZeroNumber` + compact `LP_sp_index`/`LP_sp_block` arrays,
sdpa_struct.cpp:948-972 and sdpa_io.cpp:540-560), instead of a dense
`Vector{T}` of length `LP_sp_nBlock` per constraint. At scale (many
thousands of constraints, tens of thousands of LP scalars, only a few
LP nonzeros per constraint) the dense array would allocate hundreds of
millions of BigFloat objects — tens of GB — while the compact form
stores only the actual entries. `LP_sp_index` is compact here for the
same reason: a full-length identity index vector per space would still
cost GBs.

The "index" arrays exist because the C++ solver can permute block
ordering (sparse Schur complement reorders for fill reduction). For
the canonical problem-data case (`InputData::C` and `A[i]`), they are
`1:nBlock` in order.
"""
mutable struct SDPASparseLinearSpace{T<:Real}
    SDP_sp_nBlock::Int
    SOCP_sp_nBlock::Int
    LP_sp_nBlock::Int
    SDP_sp_index::Vector{Int}
    SOCP_sp_index::Vector{Int}
    LP_sp_index::Vector{Int}
    SDP_sp_block::Vector{SDPASparseMatrix{T}}
    SOCP_sp_block::Vector{SDPASparseMatrix{T}}
    LP_nnz::Int
    LP_sp_val::Vector{T}

    function SDPASparseLinearSpace{T}(
        SDP_sp_nBlock::Int, LP_sp_nBlock::Int
    ) where {T<:Real}
        SDP_sp_nBlock >= 0 || error("SDP_sp_nBlock must be non-negative")
        LP_sp_nBlock >= 0 || error("LP_sp_nBlock must be non-negative")
        return new{T}(
            SDP_sp_nBlock,
            0,  # SOCP_sp_nBlock: always 0 (SOCP unsupported)
            LP_sp_nBlock,
            collect(1:SDP_sp_nBlock),       # SDP_sp_index
            Int[],                          # SOCP_sp_index: empty
            Int[],                          # LP_sp_index: compact nonzero positions, empty
            Vector{SDPASparseMatrix{T}}(undef, SDP_sp_nBlock),  # SDP_sp_block
            SDPASparseMatrix{T}[],          # SOCP_sp_block: empty
            0,          # LP_nnz: no nonzeros yet
            T[],        # LP_sp_val: empty
        )
    end
end

"""
    setelement_sdp!(S, block, i, j, value)

Set the `(i, j)` entry of the SDP block `block` in `S` to `value`.
The block must already be assigned with its real dimensions — every
construction path sizes its blocks up front (the readers'
`make_empty_space` loops and the MOI ingest pre-assign
`SDPASparseMatrix{T}(bs, bs, :sparse, 0)`, matching the C++
SparseLinearSpace::setElement_SDP call sites at sdpa_struct.cpp:780-815).
Writing to an unassigned block errors loudly: without real dimensions
there is nothing to bounds-check against, and the failure would
otherwise surface later as an opaque BoundsError in `change_to_dense!`.
"""
function setelement_sdp!(
    S::SDPASparseLinearSpace{T}, block::Int, i::Int, j::Int, value::T
) where {T<:Real}
    1 <= block <= S.SDP_sp_nBlock ||
        error("SDP block index $block out of [1, $(S.SDP_sp_nBlock)]")
    # Positivity is validated unconditionally: indices are 1-based, so
    # i=0 / j=0 (or negative) are never legitimate.
    1 <= i ||
        error("SDP entry row index $i must be >= 1 (block $block)")
    1 <= j ||
        error("SDP entry column index $j must be >= 1 (block $block)")
    if !isassigned(S.SDP_sp_block, block)
        error("SDPASparseLinearSpace: SDP block $block has no dimensions (never sized) — pre-assign SDP_sp_block[$block] = SDPASparseMatrix{T}(bs, bs, :sparse, 0); readers and MOI ingest do this automatically")
    end
    mat = S.SDP_sp_block[block]
    # Bounds are validated against the block's real dimensions. Every
    # assigned block is sized by its construction path, so the checks
    # run unconditionally — a (0,0)-sized block fails loudly here on
    # the first write instead of passing silently. `i <= j` is
    # intentionally NOT enforced: the dense reader relies on
    # lower-triangle writes.
    i <= mat.nRow ||
        error("SDP entry row index $i out of [1, $(mat.nRow)] (block $block)")
    j <= mat.nCol ||
        error("SDP entry column index $j out of [1, $(mat.nCol)] (block $block)")
    # Index-mirror of C++ SparseLinearSpace::setElement_SDP at sdpa_struct.cpp:780-815.
    if mat.type === :dense
        mat.de_ele[i, j] = value
    else
        # Sparse path: append (row, col, value) to the index/value arrays.
        # The C++ uses a NonZeroCount capacity-doubling scheme; we use
        # Julia's push! which has the same amortized O(1) property.
        if mat.nonZeroEffect == mat.nonZeroCount
            new_cap = max(8, mat.nonZeroCount * 2)
            resize!(mat.row_index, new_cap)
            resize!(mat.column_index, new_cap)
            resize!(mat.sp_ele, new_cap)
            mat.nonZeroCount = new_cap
        end
        mat.nonZeroEffect += 1
        idx = mat.nonZeroEffect
        mat.row_index[idx] = i
        mat.column_index[idx] = j
        mat.sp_ele[idx] = value
    end
    return S
end

"""
    setelement_lp!(S, block, value)

Set the `block`-th LP scalar to `value`. LP "blocks" are scalars (when
`blockStruct[k] == 1`) or short packed vectors (when `blockStruct[k] < 0`);
the `SparseLinearSpace` stores them as individual scalars in the compact
`LP_sp_index`/`LP_sp_val` arrays (mirrors C++ SparseLinearSpace::setElement_LP
at `sdpa_struct.cpp:817-835`).

Multi-element LP blocks would normally need an inner offset within the LP
array, but SDPLIB problems only use scalar LP blocks; we follow C++ and
treat LP blocks as scalars here.

Compact storage semantics: only nonzero values are stored (C++
sdpa_io.cpp:540-560 reads LP entries only when `tmp != 0.0`). Setting a
zero value removes the entry (a missing entry reads back as 0).
`LP_sp_index` is kept sorted ascending, so lookups are binary searches and
iteration is in LP-scalar order.
"""
function setelement_lp!(
    S::SDPASparseLinearSpace{T}, block::Int, value::T
) where {T<:Real}
    1 <= block <= S.LP_sp_nBlock ||
        error("LP block index $block out of [1, $(S.LP_sp_nBlock)]")
    idx = searchsortedfirst(S.LP_sp_index, block)
    if idx <= S.LP_nnz && S.LP_sp_index[idx] == block
        if iszero(value)
            # Removing the entry: a missing entry reads back as 0.
            deleteat!(S.LP_sp_index, idx)
            deleteat!(S.LP_sp_val, idx)
            S.LP_nnz -= 1
        else
            S.LP_sp_val[idx] = value
        end
    else
        iszero(value) && return S  # never store zeros
        insert!(S.LP_sp_index, idx, block)
        insert!(S.LP_sp_val, idx, value)
        S.LP_nnz += 1
    end
    return S
end

"""
    addelement_lp!(S, block, value)

Accumulate `value` into the `block`-th LP scalar (`LP_sp_val += value`).
Used by the MOI ingest path where multiple terms can map to the same LP
position. Keeps the compact arrays nonzero-only: if the accumulated sum
is exactly zero the entry is removed.
"""
function addelement_lp!(
    S::SDPASparseLinearSpace{T}, block::Int, value::T
) where {T<:Real}
    1 <= block <= S.LP_sp_nBlock ||
        error("LP block index $block out of [1, $(S.LP_sp_nBlock)]")
    idx = searchsortedfirst(S.LP_sp_index, block)
    if idx <= S.LP_nnz && S.LP_sp_index[idx] == block
        newval = S.LP_sp_val[idx] + value
        if iszero(newval)
            deleteat!(S.LP_sp_index, idx)
            deleteat!(S.LP_sp_val, idx)
            S.LP_nnz -= 1
        else
            S.LP_sp_val[idx] = newval
        end
    else
        iszero(value) && return S
        insert!(S.LP_sp_index, idx, block)
        insert!(S.LP_sp_val, idx, value)
        S.LP_nnz += 1
    end
    return S
end

"""
    get_lp_value(S, lp_k) -> T

Return the stored LP value at LP scalar `lp_k`, or `zero(T)` if the
entry is absent (compact storage keeps only nonzeros). Binary search
over the sorted `LP_sp_index`.
"""
@inline function get_lp_value(S::SDPASparseLinearSpace{T}, lp_k::Int) where {T<:Real}
    idx = searchsortedfirst(S.LP_sp_index, lp_k)
    if idx <= S.LP_nnz && S.LP_sp_index[idx] == lp_k
        return S.LP_sp_val[idx]
    end
    return zero(T)
end

function setzero!(S::SDPASparseLinearSpace{T}) where {T<:Real}
    for k in 1:S.SDP_sp_nBlock
        if isassigned(S.SDP_sp_block, k)
            setzero!(S.SDP_sp_block[k])
        end
    end
    # Compact LP storage: "all zeros" == "no stored entries".
    S.LP_nnz = 0
    empty!(S.LP_sp_index)
    empty!(S.LP_sp_val)
    return S
end

function setidentity!(S::SDPASparseLinearSpace{T}, scalar::Real=one(T)) where {T<:Real}
    s = T(scalar)
    for k in 1:S.SDP_sp_nBlock
        if !isassigned(S.SDP_sp_block, k)
            # Block never allocated; skip (identity undefined for empty block).
            continue
        end
        setidentity!(S.SDP_sp_block[k], s)
    end
    # Every LP scalar becomes `s` (dense in the compact form: all
    # LP_sp_nBlock entries stored). Independent BigFloat copies per slot.
    S.LP_nnz = S.LP_sp_nBlock
    resize!(S.LP_sp_index, S.LP_sp_nBlock)
    for k in 1:S.LP_sp_nBlock
        S.LP_sp_index[k] = k
    end
    resize!(S.LP_sp_val, S.LP_sp_nBlock)
    _fill_independent!(S.LP_sp_val, s)
    return S
end

"""
    change_to_dense!(S; force=false)

Convert each SDP block from sparse to dense if its density exceeds 0.20
(delegates to `change_to_dense!` on each `SDPASparseMatrix`). Mirrors
C++ SparseLinearSpace::changeToDense at `sdpa_struct.cpp:940-960`.
"""
function change_to_dense!(S::SDPASparseLinearSpace{T}; force::Bool=false) where {T<:Real}
    for k in 1:S.SDP_sp_nBlock
        if isassigned(S.SDP_sp_block, k)
            change_to_dense!(S.SDP_sp_block[k]; force=force)
        end
    end
    return S
end

function Base.copyto!(
    dst::SDPASparseLinearSpace{T}, src::SDPASparseLinearSpace{T}
) where {T<:Real}
    dst.SDP_sp_nBlock == src.SDP_sp_nBlock ||
        throw(DimensionMismatch("SDP_sp_nBlock mismatch"))
    dst.LP_sp_nBlock == src.LP_sp_nBlock ||
        throw(DimensionMismatch("LP_sp_nBlock mismatch"))
    copyto!(dst.SDP_sp_index, src.SDP_sp_index)
    for k in 1:dst.SDP_sp_nBlock
        if isassigned(src.SDP_sp_block, k)
            if !isassigned(dst.SDP_sp_block, k)
                dst.SDP_sp_block[k] = copy(src.SDP_sp_block[k])
            else
                copyto!(dst.SDP_sp_block[k], src.SDP_sp_block[k])
            end
        elseif isassigned(dst.SDP_sp_block, k)
            # src block k is unassigned — dst must not keep stale data
            # from an earlier copy (`copyto!` must leave dst == src). Zero
            # the dst block via the established setzero! idiom (:sparse →
            # nonZeroEffect = 0; :dense → independently zeroed de_ele).
            setzero!(dst.SDP_sp_block[k])
        end
    end
    # Compact LP storage: index vector is a plain copy (Int); the value
    # vector must be a value-copy so each BigFloat slot is independent
    # (`copyto!` on Vector{BigFloat} would share objects).
    dst.LP_nnz = src.LP_nnz
    resize!(dst.LP_sp_index, src.LP_nnz)
    copyto!(dst.LP_sp_index, src.LP_sp_index)
    dst.LP_sp_val = _copy_independent(src.LP_sp_val)
    return dst
end
