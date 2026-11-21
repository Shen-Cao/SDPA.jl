# Internal assembly/factor storage for a contiguous, fill-closed Schur
# profile. This is NOT a general SDPASparseMatrix: explicit coordinates and
# their ability to represent holes remain on the compatibility path.
#
# Smaller index first: column i contains (i, i), …, (i, reach[i]). Closure
# is exactly nondecreasing reach: if reach[i] >= i+1 then i+1 must reach
# as far as i; otherwise reach[i] == i and the next diagonal suffices.
# Validate the structure once, before allocating any numeric values.
struct _ClosedSkyline
    # Constructed here and read-only in all consumers.
    offsets::Vector{Int}

    function _ClosedSkyline(reaches::AbstractVector{Int})
        Base.require_one_based_indexing(reaches)
        n = length(reaches)
        offsets = Vector{Int}(undef, Base.checked_add(n, 1))
        offsets[1] = 1
        previous = 0
        for i in 1:n
            r = reaches[i]
            offsets[i + 1] = _checked_skyline_end(offsets[i], n, i, r, previous)
            previous = r
        end
        return new(offsets)
    end
end

# Signed width is parameterized so overflow boundaries can also be tested
# for 32-bit offsets on a 64-bit host.
const _SchurIndexWidth = Union{Int32,Int64}

@inline function _checked_skyline_width(n::I, i::I, r::I, previous::I) where {I<:_SchurIndexWidth}
    i <= r <= n || throw(ArgumentError("skyline reach for column $i must be in $i:$n, got $r"))
    previous <= r || throw(ArgumentError(
        "skyline is not fill-in closed: reach $r in column $i is below $previous"))
    return r - i + one(I)
end

@inline function _checked_skyline_end(start::I, n::I, i::I, r::I, previous::I) where {I<:_SchurIndexWidth}
    # Include the diagonal and retain a native-width one-past-the-end sentinel.
    return Base.checked_add(start, _checked_skyline_width(n, i, r, previous))
end

_skyline_slots(shape::_ClosedSkyline) = shape.offsets[end] - 1

# Route comparisons use hypothetical sizes. In particular, the triangular
# denominator (or a candidate envelope that would be rejected by the gate)
# can exceed native Int even though the storage actually selected fits.
# Keep this scan allocation-free: only selected sparse storage needs offsets.
function _schur_route_slots(reaches::AbstractVector{I}) where {I<:_SchurIndexWidth}
    Base.require_one_based_indexing(reaches)
    n = I(length(reaches))
    W = widen(I)
    total = zero(W)
    previous = zero(I)
    for (i, r) in enumerate(reaches)
        total = Base.checked_add(total, W(_checked_skyline_width(n, I(i), r, previous)))
        previous = r
    end
    return total
end

@inline function _schur_triangle_slots(n::I) where {I<:_SchurIndexWidth}
    n >= 0 || throw(ArgumentError("Schur dimension must be nonnegative, got $n"))
    dim = widen(n)
    return Base.checked_mul(dim, Base.checked_add(dim, one(dim))) ÷ 2
end

# Exact comparison for the routing constants (1/4 and 2/5). Computing
# floor(total * numerator / denominator) by quotient and remainder avoids
# overflowing even if the widened triangle itself is close to its limit.
@inline function _schur_density_exceeds(slots::Integer, total::Integer, numerator::Int, denominator::Int)
    q, r = divrem(total, denominator)
    return slots > q * numerator + (r * numerator) ÷ denominator
end

struct _SchurSkyline{T<:Real}
    offsets::Vector{Int}
    sp_ele::Vector{T}

    function _SchurSkyline{T}(shape::_ClosedSkyline) where {T<:Real}
        return new{T}(shape.offsets, _independent_zeros(T, _skyline_slots(shape)))
    end
end

_SchurSkyline{T}(reaches::AbstractVector{Int}) where {T<:Real} =
    _SchurSkyline{T}(_ClosedSkyline(reaches))

@inline _schur_offsets(mat::_SchurSkyline, ::Vector{Int}) = mat.offsets
@inline _schur_offsets(::SDPASparseMatrix, diagonalIndex::Vector{Int}) = diagonalIndex

@inline function _schur_assembly_slot(
    ::_SchurSkyline, offsets::Vector{Int}, i::Int, j::Int, context::String,
)
    start = offsets[i]
    i <= j && j - i < offsets[i + 1] - start ||
        error("$context pair (ii=$i, jj=$j) outside block-level profile")
    return start + (j - i)
end
