"""
    SDPASparseMatrix{T<:Real}

A sparse-or-dense matrix with element type `T`. Mirrors C++
`sdpa::SparseMatrix` at `sdpa_struct.h:72-109`.

The C++ class carries both layouts and converts between them via
`changeToDense`. A Julia `:sparse` matrix has row/column index arrays plus
values; a `:dense` matrix has a column-major `de_ele::Matrix{T}`.

Three "non-zero counters" track different things in the C++ F1/F2/F3
cost model:

  - `nonZeroNumber`:      declared capacity (allocator hint)
  - `nonZeroCount`:       memory actually allocated (current capacity)
  - `nonZeroEffect`:      currently-stored nonzeros (in use)

For sparse storage the three are kept distinct because `changeToDense!` triggers when the
"density" = `nonZeroEffect / (nRow*nCol)` exceeds ~20% (C++
`sdpa_struct.cpp:491`), and we need to know the *current usage*
distinct from the *allocated capacity* to avoid reallocating on every
insertion. For dense storage all three are the number of dense slots, as
used by the Schur formula cost model. They don't indicate whether all
of the stored values happen to be zero.

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
        if type === :dense
            n = length(de_ele)
            return new{T}(nRow, nCol, type, n, n, n, de_ele, Int[], Int[], T[])
        end
        return new{T}(
            nRow, nCol, type, nonZeroNumber, nonZeroNumber, 0,
            de_ele, zeros(Int, nonZeroNumber), zeros(Int, nonZeroNumber),
            _independent_zeros(T, nonZeroNumber),
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
        resize!(m.row_index, n)
        resize!(m.column_index, n)
        resize!(m.sp_ele, n)
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
    dst === src && return dst

    if src.type === :dense
        # Dense counters can be nonzero, but there is no sparse payload.
        size(src.de_ele) == size(src) && size(dst.de_ele) == size(dst) ||
            throw(DimensionMismatch("dense payload size does not match matrix dimensions"))
        _copyto_independent!(dst.de_ele, src.de_ele)
    elseif src.type === :sparse
        cap = src.nonZeroCount
        n = src.nonZeroEffect
        0 <= n <= cap || throw(ArgumentError("invalid sparse in-use count"))
        length(src.row_index) == length(src.column_index) == length(src.sp_ele) == cap ||
            throw(DimensionMismatch("sparse payload lengths do not match allocated capacity"))

        # A grown BigFloat source can have undefined values past n. Keep the
        # full allocated capacity, with independent, initialized dst values,
        # and only read src values from the in-use prefix.
        resize!(dst.row_index, cap)
        resize!(dst.column_index, cap)
        resize!(dst.sp_ele, cap)
        copyto!(dst.row_index, src.row_index)
        copyto!(dst.column_index, src.column_index)
        _fill_independent!(dst.sp_ele, zero(T))
        _copy_prefix_independent!(dst.sp_ele, src.sp_ele, n)
    else
        throw(ArgumentError("unknown matrix representation $(src.type)"))
    end

    dst.nonZeroNumber = src.nonZeroNumber
    dst.nonZeroCount = src.nonZeroCount
    dst.nonZeroEffect = src.nonZeroEffect
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
    # Dense elements start at working precision. Sparse capacity is grown
    # and initialized once, by the same path as an in-place copy.
    return copyto!(SDPASparseMatrix{T}(m.nRow, m.nCol, m.type), m)
end

"""
    change_to_dense!(m; force=false)

Convert `m` from `:sparse` to `:dense` in place. Conversion is triggered
automatically when the density (`nonZeroEffect / (nRow*nCol)`) exceeds
0.20, matching C++ `SparseMatrix::changeToDense(bool forceChange=false)`
at `sdpa_struct.cpp:491`.

After conversion: `de_ele` is a `Matrix{T}` of size `(nRow, nCol)`
(column-major). Index arrays are released. `nonZeroNumber/Count/Effect`
become the number of dense slots.

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
