"""
    SDPASparseMatrix{T<:Real}

A sparse-or-dense matrix with element type `T`. Mirrors C++
`sdpa::SparseMatrix` at `sdpa_struct.h:72-109`.

The C++ class carries both layouts simultaneously and converts between
them via `changeToDense`. We mirror that exact structure: a `:sparse`
matrix has row/column index arrays plus values; a `:dense` matrix has a
flat `de_ele::Vector{T}` (NOT a `Matrix{T}` — the C++ storage is a flat
pointer, and `changeToDense!` reallocates it as `nRow*nCol`).

Three "non-zero counters" track different things in the C++ F1/F2/F3
cost model:

  - `nonZeroNumber`:      declared capacity (allocator hint)
  - `nonZeroCount`:       memory actually allocated (current capacity)
  - `nonZeroEffect`:      currently-stored nonzeros (in use)

The three are kept distinct because `changeToDense!` triggers when the
"density" = `nonZeroEffect / (nRow*nCol)` exceeds ~20% (C++
`sdpa_struct.cpp:491`), and we need to know the *current usage*
distinct from the *allocated capacity* to avoid reallocating on every
insertion.

Fields
------
- `nRow::Int`, `nCol::Int`
- `type::Symbol`:                  `:sparse` or `:dense`
- `nonZeroNumber::Int`:           declared nonzero capacity
- `nonZeroCount::Int`:            allocated capacity
- `nonZeroEffect::Int`:           currently-stored nonzeros
- `de_ele::Matrix{T}`:            dense branch storage (size
                                  `(nRow, nCol)` when `type == :dense`;
                                  `(0, 0)` when `:sparse`)
- `row_index::Vector{Int}`:       sparse branch row indices
- `column_index::Vector{Int}`:    sparse branch column indices
- `sp_ele::Vector{T}`:            sparse branch values

For a `:dense` matrix, the index arrays are empty and `de_ele` is the
authoritative storage. For a `:sparse` matrix, `de_ele` is `(0, 0)` and
the index+value arrays are authoritative.

The dense-branch storage uses `Matrix{T}` (column-major, GC-managed,
immediately compatible with `cholesky!(Hermitian(M, :U))`, `mul!`,
`eigen`, etc.) rather than the C++'s flat `mpf_class *de_ele` pointer.
The column-major layout matches both Julia and MPLAPACK conventions.
"""
mutable struct SDPASparseMatrix{T<:Real}
    nRow::Int
    nCol::Int
    type::Symbol
    nonZeroNumber::Int
    nonZeroCount::Int
    nonZeroEffect::Int
    de_ele::Matrix{T}
    row_index::Vector{Int}
    column_index::Vector{Int}
    sp_ele::Vector{T}

    function SDPASparseMatrix{T}(
        nRow::Int, nCol::Int, type::Symbol, nonZeroNumber::Int=0
    ) where {T<:Real}
        nRow >= 0 || error("nRow must be non-negative, got $nRow")
        nCol >= 0 || error("nCol must be non-negative, got $nCol")
        nonZeroNumber >= 0 || error("nonZeroNumber must be non-negative, got $nonZeroNumber")
        (type === :sparse || type === :dense) ||
            error("type must be :sparse or :dense, got $type")
        de_ele = type === :dense ? _independent_zeros(T, nRow, nCol) : Matrix{T}(undef, 0, 0)
        row_idx = type === :sparse ? zeros(Int, nonZeroNumber) : Int[]
        col_idx = type === :sparse ? zeros(Int, nonZeroNumber) : Int[]
        sp_ele = type === :sparse ? _independent_zeros(T, nonZeroNumber) : T[]
        return new{T}(
            nRow, nCol, type, nonZeroNumber, nonZeroNumber, 0,
            de_ele, row_idx, col_idx, sp_ele,
        )
    end
end

function setzero!(m::SDPASparseMatrix{T}) where {T<:Real}
    if m.type === :dense
        _fill_independent!(m.de_ele, zero(T))
    else
        m.nonZeroEffect = 0
    end
    return m
end

"""
    setidentity!(m, scalar=1)

Set `m` to `scalar * I`. For a `:dense` matrix, fill the diagonal with
`scalar` and zero elsewhere. For a `:sparse` matrix, allocate exactly
`min(nRow, nCol)` entries on the diagonal.

The sparse path mirrors C++ `SparseMatrix::setIdentity` at
`sdpa_struct.cpp:470-490`.
"""
function setidentity!(m::SDPASparseMatrix{T}, scalar::Real=one(T)) where {T<:Real}
    s = T(scalar)
    n = min(m.nRow, m.nCol)
    if m.type === :dense
        _fill_independent!(m.de_ele, zero(T))
        for i in 1:n
            # Independent diagonal elements (see SDPADenseMatrix
            # setidentity! — in-place MPFR kernels must not share objects).
            m.de_ele[i, i] = deepcopy(s)
        end
    else
        # Sparse: reset to n diagonal entries.
        if length(m.row_index) < n
            resize!(m.row_index, n)
            resize!(m.column_index, n)
            resize!(m.sp_ele, n)
        end
        for i in 1:n
            m.row_index[i] = i
            m.column_index[i] = i
            m.sp_ele[i] = deepcopy(s)
        end
        m.nonZeroNumber = n
        m.nonZeroCount = n
        m.nonZeroEffect = n
    end
    return m
end

Base.size(m::SDPASparseMatrix) = (m.nRow, m.nCol)
Base.size(m::SDPASparseMatrix, d::Int) = (d == 1) ? m.nRow : (d == 2 ? m.nCol : 1)
Base.eltype(::Type{SDPASparseMatrix{T}}) where {T<:Real} = T
Base.eltype(::SDPASparseMatrix{T}) where {T<:Real} = T

function Base.copyto!(dst::SDPASparseMatrix{T}, src::SDPASparseMatrix{T}) where {T<:Real}
    dst.nRow == src.nRow || throw(DimensionMismatch("nRow mismatch"))
    dst.nCol == src.nCol || throw(DimensionMismatch("nCol mismatch"))
    dst.type === src.type ||
        error("copyto! type mismatch: $(dst.type) vs $(src.type); call change_to_dense! first")
    dst.nonZeroNumber = src.nonZeroNumber
    dst.nonZeroCount = src.nonZeroCount
    dst.nonZeroEffect = src.nonZeroEffect
    _copyto_independent!(dst.de_ele, src.de_ele)
    # src's sparse arrays may be capacity-grown (resize!-doubled in
    # setelement_sdp!) past dst's lengths. Base.copyto! on mismatched
    # vectors throws a raw BoundsError, and dst's counters were just set
    # to src's — the next append would write past dst's shorter arrays.
    # Bring dst to src's lengths first (resize! to the same length is a
    # no-op on the happy path).
    resize!(dst.row_index, length(src.row_index))
    resize!(dst.column_index, length(src.column_index))
    resize!(dst.sp_ele, length(src.sp_ele))
    copyto!(dst.row_index, src.row_index)
    copyto!(dst.column_index, src.column_index)
    # sp_ele must be TOTAL: every slot 1:length(dst.sp_ele) ends up
    # defined. resize! growth leaves #undef BigFloat slots that mp_set!
    # cannot even reference, so first (re)define each slot with an
    # independent zero (plain assignment — the _fill_independent! idiom),
    # then value-copy the in-use prefix 1:nonZeroEffect. The prefix is
    # never read beyond nonZeroEffect: src's capacity tail may hold #undef
    # BigFloat slots, which a full-vector _copyto_independent! would read
    # (UndefRefError).
    _fill_independent!(dst.sp_ele, zero(T))
    _copy_prefix_independent!(dst.sp_ele, src.sp_ele, src.nonZeroEffect)
    return dst
end

# Copy the in-use prefix `src[1:n]` into `dst[1:n]` as independent values
# (BigFloat slots must not alias, matching `_copyto_independent!`). Only the
# prefix is read: the tail of a capacity-grown (`resize!`-doubled) BigFloat
# vector holds undefined references, so the full-vector `_copy_independent`
# would throw UndefRefError.
@inline function _copy_prefix_independent!(
    dst::Vector{BigFloat}, src::Vector{BigFloat}, n::Int
)
    @inbounds for k in 1:n
        mp_set!(dst[k], src[k])
    end
    return dst
end
@inline function _copy_prefix_independent!(
    dst::Vector{T}, src::Vector{T}, n::Int
) where {T<:Real}
    @inbounds for k in 1:n
        dst[k] = src[k]
    end
    return dst
end

function Base.copy(m::SDPASparseMatrix{T}) where {T<:Real}
    out = SDPASparseMatrix{T}(m.nRow, m.nCol, m.type, m.nonZeroCount)
    out.nonZeroCount = m.nonZeroCount
    out.nonZeroEffect = m.nonZeroEffect
    out.de_ele = _copy_independent(m.de_ele)
    copyto!(out.row_index, m.row_index)
    copyto!(out.column_index, m.column_index)
    # `out`'s constructor zero-initialized its full capacity slots; copy the
    # in-use prefix into them. Dense matrices carry no sparse value array.
    if m.type === :sparse
        _copy_prefix_independent!(out.sp_ele, m.sp_ele, m.nonZeroEffect)
    else
        out.sp_ele = _copy_independent(m.sp_ele)
    end
    return out
end

"""
    change_to_dense!(m; force=false)

Convert `m` from `:sparse` to `:dense` in place. Conversion is triggered
automatically when the density (`nonZeroEffect / (nRow*nCol)`) exceeds
0.20, matching C++ `SparseMatrix::changeToDense(bool forceChange=false)`
at `sdpa_struct.cpp:491`.

After conversion: `de_ele` is a `Matrix{T}` of size `(nRow, nCol)`
(column-major). Index arrays are released. `nonZeroNumber/Count/Effect`
retain their meaning.

If `force=true`, converts regardless of density.
"""
function change_to_dense!(m::SDPASparseMatrix{T}; force::Bool=false) where {T<:Real}
    m.type === :dense && return m
    total = m.nRow * m.nCol
    density = total == 0 ? 1.0 : m.nonZeroEffect / total
    if !force && density <= 0.20
        return m
    end
    de_ele = _independent_zeros(T, m.nRow, m.nCol)
    for k in 1:m.nonZeroEffect
        i = m.row_index[k]
        j = m.column_index[k]
        v = m.sp_ele[k]
        de_ele[i, j] = v
        # Mirror upper-triangle entries to lower triangle so the dense
        # matrix is fully symmetric. The SDPA sparse format stores only
        # upper-triangle entries (i <= j); without this mirroring, the
        # Frobenius inner product in cal_f1! undercounts off-diagonal
        # contributions by 2x, corrupting the Schur complement matrix.
        if i != j
            # Independent mirror: `de_ele[j, i] = v` would store the SAME
            # BigFloat object in both transposed slots, so an in-place MPFR
            # op on one would corrupt the mirror. `copy(v)` would also
            # alias on Julia 1.12 (copy returns the same object for
            # BigFloat), so value-copy via mp_set! into the pre-allocated
            # independent slot.
            if T <: BigFloat
                mp_set!(de_ele[j, i], v)
            else
                de_ele[j, i] = v
            end
        end
    end
    m.de_ele = de_ele
    empty!(m.row_index)
    empty!(m.column_index)
    empty!(m.sp_ele)
    m.type = :dense
    m.nonZeroNumber = total
    m.nonZeroCount = total
    m.nonZeroEffect = total
    return m
end
