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

@inline function _checked_skyline_end(start::Int, n::Int, i::Int, r::Int, previous::Int)
    i <= r <= n || throw(ArgumentError("skyline reach for column $i must be in $i:$n, got $r"))
    previous <= r || throw(ArgumentError(
        "skyline is not fill-in closed: reach $r in column $i is below $previous"))
    # Include the diagonal and retain a checked one-past-the-end sentinel.
    return Base.checked_add(start, r - i + 1)
end

_skyline_slots(shape::_ClosedSkyline) = shape.offsets[end] - 1

# Divide before multiplying so representable triangular counts are not
# rejected by an intermediate overflow.
@inline function _schur_triangle_slots(n::Int)
    n >= 0 || throw(ArgumentError("Schur dimension must be nonnegative, got $n"))
    iseven(n) && return Base.checked_mul(n ÷ 2, Base.checked_add(n, 1))
    return Base.checked_mul(n, n ÷ 2 + 1)
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
