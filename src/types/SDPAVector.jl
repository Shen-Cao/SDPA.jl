"""
    SDPAVector{T<:Real}

A 1-D real vector with element type `T`. Mirrors C++ `sdpa::Vector` at
`sdpa_struct.h:33-50`. Prefixed `SDPA` to avoid collision
with `Base.Vector`.

The SDPA-GMP C++ uses `mpf_class *ele` (raw heap pointer + size); Julia
uses a typed `Vector{T}` which is GC-managed and parametric. For `T =
BigFloat`, every element is a separate `BigFloat` (its own `mpfr_t`);
for `T = Float64`, the storage is contiguous.

Fields
------
- `nDim::Int`:        number of elements
- `ele::Vector{T}`:   element storage (length `nDim`)

Constructors
------------
- `SDPAVector{T}(nDim::Int; value=zero(T))`: length-`nDim` vector filled
  with `value`.
- `SDPAVector{T}(ele::Vector{T})`: adopt an existing vector (no copy).

The struct is mutable so that in-place operations (`setzero!`, `copyto!`,
the IPM's `update!` step) can mutate elements. The element
storage is a heap-allocated `Vector{T}`, so mutation is on the heap, not
the stack frame.
"""
mutable struct SDPAVector{T<:Real}
    nDim::Int
    ele::Vector{T}

    function SDPAVector{T}(nDim::Int; value::T=zero(T)) where {T<:Real}
        nDim >= 0 || error("nDim must be non-negative, got $nDim")
        # Independent elements: for BigFloat, `fill(value, n)` aliases ALL
        # elements to one shared object, so in-place MPFR mutation of one
        # element would corrupt the others. `_fill_independent!` creates a
        # fresh BigFloat per slot (mpfr_set value copy); for isbits types
        # it falls back to `fill!`.
        ele = Vector{T}(undef, nDim)
        _fill_independent!(ele, value)
        return new{T}(nDim, ele)
    end

    function SDPAVector{T}(ele::Vector{T}) where {T<:Real}
        return new{T}(length(ele), ele)
    end
end

function SDPAVector{T}(nDim::Int, value::Real) where {T<:Real}
    return SDPAVector{T}(nDim; value=T(value))
end

setzero!(v::SDPAVector{T}) where {T<:Real} = (_fill_independent!(v.ele, zero(T)); return v)
setidentity!(v::SDPAVector{T}, scalar::Real=one(T)) where {T<:Real} =
    (_fill_independent!(v.ele, T(scalar)); return v)

Base.length(v::SDPAVector) = v.nDim
Base.size(v::SDPAVector) = (v.nDim,)
Base.size(v::SDPAVector, d::Int) = (d == 1) ? v.nDim : 1
Base.eltype(::Type{SDPAVector{T}}) where {T<:Real} = T
Base.eltype(::SDPAVector{T}) where {T<:Real} = T

@inline Base.getindex(v::SDPAVector, i::Int) = v.ele[i]
@inline Base.setindex!(v::SDPAVector, x, i::Int) = (v.ele[i] = x; return v)

function Base.copyto!(dst::SDPAVector{T}, src::SDPAVector{T}) where {T<:Real}
    dst.nDim == src.nDim ||
        throw(DimensionMismatch("SDPAVector copyto! size mismatch: $(dst.nDim) vs $(src.nDim)"))
    # Value copy (not reference copy): Base copyto! for Vector{BigFloat}
    # copies REFERENCES (dst[i] === src[i]), which creates aliasing that
    # corrupts in-place MPFR operations. `_copyto_independent!` copies
    # values via mp_set!; for isbits types it falls back to `copyto!`.
    _copyto_independent!(dst.ele, src.ele)
    return dst
end

Base.similar(v::SDPAVector{T}) where {T<:Real} = SDPAVector{T}(v.nDim)
Base.copy(v::SDPAVector{T}) where {T<:Real} = SDPAVector{T}(_copy_independent(v.ele))

function Base.:(==)(a::SDPAVector{T}, b::SDPAVector{T}) where {T<:Real}
    return a.nDim == b.nDim && a.ele == b.ele
end

# == without a matching hash/isequal breaks the equality
# contract (value-equal vectors hashed differently → unusable as Dict keys
# / Set elements). hash is derived from the same fields == compares.
Base.isequal(a::SDPAVector{T}, b::SDPAVector{T}) where {T<:Real} =
    a.nDim == b.nDim && isequal(a.ele, b.ele)

Base.hash(v::SDPAVector{T}, h::UInt) where {T<:Real} =
    hash(v.ele, hash(v.nDim, h))
