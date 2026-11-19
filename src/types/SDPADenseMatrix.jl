"""
    SDPADenseMatrix{T<:Real}

A dense matrix with element type `T`. Mirrors C++ `sdpa::DenseMatrix` at
`sdpa_struct.h:111-133`.

The C++ stores `mpf_class *de_ele` as a heap pointer of length
`nRow*nCol`. The Julia port uses `Matrix{T}` directly (column-major,
GC-managed, immediately compatible with `cholesky!(Hermitian(M, :U))`,
`mul!`, `eigen`, etc., without index arithmetic).

The C++ `Type { DENSE, COMPLETION }` enum is preserved as a `Symbol`
field `:dense` or `:completion` to track whether the matrix is fully
populated or still being filled (relevant for sparse→dense conversion
and the `make_bmat` workspace pattern).

Fields
------
- `nRow::Int`
- `nCol::Int`
- `type::Symbol`:           `:dense` or `:completion`
- `de_ele::Matrix{T}`:      storage, column-major
"""
mutable struct SDPADenseMatrix{T<:Real}
    nRow::Int
    nCol::Int
    type::Symbol
    de_ele::Matrix{T}

    function SDPADenseMatrix{T}(
        nRow::Int, nCol::Int, type::Symbol=:dense
    ) where {T<:Real}
        nRow >= 0 || error("nRow must be non-negative, got $nRow")
        nCol >= 0 || error("nCol must be non-negative, got $nCol")
        (type === :dense || type === :completion) ||
            error("type must be :dense or :completion, got $type")
        return new{T}(nRow, nCol, type, _independent_zeros(T, nRow, nCol))
    end

    function SDPADenseMatrix{T}(
        nRow::Int, nCol::Int, type::Symbol, de_ele::Matrix{T}
    ) where {T<:Real}
        size(de_ele) == (nRow, nCol) ||
            throw(DimensionMismatch("de_ele size $(size(de_ele)) != ($nRow, $nCol)"))
        (type === :dense || type === :completion) ||
            error("type must be :dense or :completion, got $type")
        return new{T}(nRow, nCol, type, de_ele)
    end
end

function setzero!(m::SDPADenseMatrix{T}) where {T<:Real}
    _fill_independent!(m.de_ele, zero(T))
    return m
end

# Helper: create a Matrix{T} where each BigFloat element has its own Memory.
# Julia's `zeros(BigFloat, m, n)` fills all elements with the SAME BigFloat
# (shared Memory), so in-place MPFR mutation (mpfr_fma!, mpfr_set!, etc.)
# on one element corrupts all others. This helper creates independent elements.
function _independent_zeros(::Type{BigFloat}, nRow::Int, nCol::Int)
    M = Matrix{BigFloat}(undef, nRow, nCol)
    for i in eachindex(M)
        M[i] = BigFloat(0)
    end
    return M
end
_independent_zeros(::Type{T}, nRow::Int, nCol::Int) where {T<:Real} = zeros(T, nRow, nCol)

# 1-D version for vectors (used by SDPASparseMatrix sp_ele)
function _independent_zeros(::Type{BigFloat}, n::Int)
    V = Vector{BigFloat}(undef, n)
    for i in eachindex(V)
        V[i] = BigFloat(0)
    end
    return V
end
_independent_zeros(::Type{T}, n::Int) where {T<:Real} = zeros(T, n)

# Helper: fill a Matrix with independent copies of val (for BigFloat).
function _fill_independent!(M::Matrix{BigFloat}, val::BigFloat)
    for i in eachindex(M)
        x = BigFloat(0)  # new independent BigFloat (from Int, not conversion)
        if !iszero(val)
            ccall((:mpfr_set, Base.MPFR.libmpfr), Int32,
                  (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
                  x, val, _mpfr_rounding())
        end
        M[i] = x
    end
    return M
end
_fill_independent!(M::Matrix{T}, val::T) where {T<:Real} = fill!(M, val)

# Helper: fill a Vector with independent copies of val (for BigFloat).
function _fill_independent!(V::Vector{BigFloat}, val::BigFloat)
    for i in eachindex(V)
        x = BigFloat(0)
        if !iszero(val)
            ccall((:mpfr_set, Base.MPFR.libmpfr), Int32,
                  (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
                  x, val, _mpfr_rounding())
        end
        V[i] = x
    end
    return V
end
_fill_independent!(V::Vector{T}, val::T) where {T<:Real} = fill!(V, val)

function setidentity!(m::SDPADenseMatrix{T}, scalar::Real=one(T)) where {T<:Real}
    s = T(scalar)
    _fill_independent!(m.de_ele, zero(T))
    n = min(m.nRow, m.nCol)
    for i in 1:n
        # Independent diagonal elements: `m.de_ele[i, i] = s` would put
        # the SAME BigFloat object in every diagonal slot; any in-place
        # MPFR kernel (Rgemm!/mp_fma! — e.g. the get_inv_chol_and_inv!
        # product) would then mutate one shared object from multiple
        # positions, racing across threads and corrupting the diagonal
        # even single-threaded. `deepcopy` (NOT BigFloat(s): identity
        # convert on Julia 1.12) breaks the sharing; isbits T are values,
        # so deepcopy is trivial.
        m.de_ele[i, i] = deepcopy(s)
    end
    return m
end

Base.size(m::SDPADenseMatrix) = (m.nRow, m.nCol)
Base.size(m::SDPADenseMatrix, d::Int) = size(m.de_ele, d)
Base.eltype(::Type{SDPADenseMatrix{T}}) where {T<:Real} = T
Base.eltype(::SDPADenseMatrix{T}) where {T<:Real} = T

@inline Base.getindex(m::SDPADenseMatrix, i::Int, j::Int) = m.de_ele[i, j]
@inline Base.setindex!(m::SDPADenseMatrix, x, i::Int, j::Int) = (m.de_ele[i, j] = x; return m)

function Base.copyto!(dst::SDPADenseMatrix{T}, src::SDPADenseMatrix{T}) where {T<:Real}
    size(dst.de_ele) == size(src.de_ele) ||
        throw(DimensionMismatch("SDPADenseMatrix copyto! size mismatch"))
    if T <: BigFloat
        # Value copy (not reference copy): Julia's default copyto! for
        # Matrix{BigFloat} copies REFERENCES (dst[i] === src[i]), which
        # creates aliasing that corrupts in-place MPFR operations.
        # We must copy VALUES via mpfr_set to keep elements independent.
        @inbounds for idx in eachindex(dst.de_ele)
            mp_set!(dst.de_ele[idx], src.de_ele[idx])
        end
    else
        copyto!(dst.de_ele, src.de_ele)
    end
    dst.type = src.type
    return dst
end

Base.copy(m::SDPADenseMatrix{T}) where {T<:Real} =
    SDPADenseMatrix{T}(m.nRow, m.nCol, m.type, _copy_independent(m.de_ele))

# Helper: truly independent copy of a Matrix{BigFloat} (values, not references).
function _copy_independent(src::Matrix{BigFloat})
    dst = Matrix{BigFloat}(undef, size(src))
    @inbounds for idx in eachindex(src)
        dst[idx] = BigFloat(0)
        ccall((:mpfr_set, Base.MPFR.libmpfr), Int32,
              (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
              dst[idx], src[idx], _mpfr_rounding())
    end
    return dst
end
_copy_independent(src::Matrix{T}) where {T<:Real} = copy(src)

# 1-D version for vectors (used by SDPAVector copy)
function _copy_independent(src::Vector{BigFloat})
    dst = Vector{BigFloat}(undef, length(src))
    @inbounds for idx in eachindex(src)
        dst[idx] = BigFloat(0)
        ccall((:mpfr_set, Base.MPFR.libmpfr), Int32,
              (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
              dst[idx], src[idx], _mpfr_rounding())
    end
    return dst
end
_copy_independent(src::Vector{T}) where {T<:Real} = copy(src)

# Helper: value-copy copyto! for raw Matrix{BigFloat} (avoids reference aliasing).
# Julia's default copyto!(dst, src) for Matrix{BigFloat} copies REFERENCES
# (dst[i] === src[i]), which creates aliasing that corrupts in-place MPFR ops.
function _copyto_independent!(dst::Matrix{BigFloat}, src::Matrix{BigFloat})
    @inbounds for idx in eachindex(dst)
        mp_set!(dst[idx], src[idx])
    end
    return dst
end
_copyto_independent!(dst::Matrix{T}, src::Matrix{T}) where {T<:Real} = copyto!(dst, src)

# Helper: value-copy copyto! for raw Vector{BigFloat} (avoids reference aliasing).
# Julia's default copyto!(dst, src) for Vector{BigFloat} copies REFERENCES
# (dst[i] === src[i]), which creates aliasing that corrupts in-place MPFR ops.
function _copyto_independent!(dst::Vector{BigFloat}, src::Vector{BigFloat})
    @inbounds for idx in eachindex(dst)
        mp_set!(dst[idx], src[idx])
    end
    return dst
end
_copyto_independent!(dst::Vector{T}, src::Vector{T}) where {T<:Real} = copyto!(dst, src)
