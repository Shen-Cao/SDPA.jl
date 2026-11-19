"""
    SDPABlockVector{T<:Real}

A vector-of-vectors partitioned by blocks. Mirrors C++ `sdpa::BlockVector`
at `sdpa_struct.h:52-69`.

The k-th block has length:

  - `blockStruct[k]` for SDP blocks (`blockStruct[k] >= 2`): the SDP
    block is a symmetric `blockStruct[k] × blockStruct[k]` matrix;
    `BlockVector` stores its **upper-triangular packed** form, which
    has `blockStruct[k] * (blockStruct[k] + 1) ÷ 2` entries.
  - `abs(blockStruct[k])` for LP blocks (`blockStruct[k] < 0` or
    `blockStruct[k] == 1`): the LP block is a length-`|bs|` real vector.

Fields
------
- `nBlock::Int`:                  number of blocks
- `blockStruct::Vector{Int}`:     signed block sizes (positive=SDP,
  negative or 1=LP)
- `ele::Vector{SDPAVector{T}}`:   per-block sub-vectors

The partition rule matches C++ `BlockVector::initialize` at
`sdpa_struct.cpp:280-310`.
"""
mutable struct SDPABlockVector{T<:Real}
    nBlock::Int
    blockStruct::Vector{Int}
    ele::Vector{SDPAVector{T}}

    function SDPABlockVector{T}(
        nBlock::Int, blockStruct::AbstractVector{Int}; value::T=zero(T)
    ) where {T<:Real}
        nBlock >= 0 || error("nBlock must be non-negative, got $nBlock")
        length(blockStruct) == nBlock || throw(
            DimensionMismatch(
                "blockStruct length $(length(blockStruct)) != nBlock $nBlock",
            ),
        )
        sub_vectors = Vector{SDPAVector{T}}(undef, nBlock)
        for k in 1:nBlock
            sub_vectors[k] = SDPAVector{T}(_block_storage_dim(blockStruct[k]); value=value)
        end
        return new{T}(nBlock, collect(Int, blockStruct), sub_vectors)
    end

    function SDPABlockVector{T}(
        nBlock::Int, blockStruct::AbstractVector{Int}, ele::Vector{SDPAVector{T}}
    ) where {T<:Real}
        length(blockStruct) == nBlock ||
            throw(DimensionMismatch("blockStruct length != nBlock"))
        length(ele) == nBlock ||
            throw(DimensionMismatch("ele length != nBlock"))
        for k in 1:nBlock
            if length(ele[k]) != _block_storage_dim(blockStruct[k])
                throw(
                    DimensionMismatch(
                        "block $k size $(length(ele[k])) != storage dim $(_block_storage_dim(blockStruct[k])) for blockStruct $k=$(blockStruct[k])",
                    ),
                )
            end
        end
        return new{T}(nBlock, collect(Int, blockStruct), copy(ele))
    end
end

@doc """
    SDPABlockVector{T}(nBlock, blockStruct, ele)

Construct from pre-built per-block sub-vectors `ele` (lengths must match
the storage dims implied by `blockStruct`).

NOTE: the element `SDPAVector` objects are shared with the caller
(shallow container copy); mutate with care.
""" SDPABlockVector{T}(nBlock::Int, blockStruct::AbstractVector{Int}, ele::Vector{SDPAVector{T}}) where {T<:Real}

"""
    _block_storage_dim(bs::Int)

Return the length of the per-block sub-vector for a block of signed
size `bs`:

  - `bs >= 2` (SDP block): `bs * (bs + 1) ÷ 2` (upper-triangular
    packed symmetric matrix storage).
  - `bs == 1` (LP block, size 1): `1`.
  - `bs <= 0` (LP block, size `|bs|`): `abs(bs)`.

Mirrors C++ `BlockVector::initialize` at `sdpa_struct.cpp:280-310`.
"""
function _block_storage_dim(bs::Int)
    if bs >= 2
        return bs * (bs + 1) ÷ 2
    elseif bs == 1
        return 1
    else
        return abs(bs)
    end
end

function setzero!(bv::SDPABlockVector{T}) where {T<:Real}
    for k in 1:bv.nBlock
        setzero!(bv.ele[k])
    end
    return bv
end

function setidentity!(bv::SDPABlockVector{T}, scalar::Real=one(T)) where {T<:Real}
    s = T(scalar)
    for k in 1:bv.nBlock
        setidentity!(bv.ele[k], s)
    end
    return bv
end

Base.length(bv::SDPABlockVector) = bv.nBlock
Base.size(bv::SDPABlockVector) = (bv.nBlock,)
Base.eltype(::Type{SDPABlockVector{T}}) where {T<:Real} = SDPAVector{T}
Base.eltype(::SDPABlockVector{T}) where {T<:Real} = SDPAVector{T}

@inline Base.getindex(bv::SDPABlockVector, k::Int) = bv.ele[k]
@inline Base.setindex!(bv::SDPABlockVector, x, k::Int) = (bv.ele[k] = x; return bv)

function Base.copyto!(dst::SDPABlockVector{T}, src::SDPABlockVector{T}) where {T<:Real}
    dst.nBlock == src.nBlock ||
        throw(DimensionMismatch("SDPABlockVector copyto! nBlock mismatch"))
    dst.blockStruct == src.blockStruct ||
        throw(DimensionMismatch("SDPABlockVector copyto! blockStruct mismatch"))
    for k in 1:dst.nBlock
        copyto!(dst.ele[k], src.ele[k])
    end
    return dst
end

function Base.copy(bv::SDPABlockVector{T}) where {T<:Real}
    new_ele = [copy(bv.ele[k]) for k in 1:bv.nBlock]
    return SDPABlockVector{T}(bv.nBlock, copy(bv.blockStruct), new_ele)
end
