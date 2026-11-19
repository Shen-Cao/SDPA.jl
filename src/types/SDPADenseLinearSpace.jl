"""
    SDPADenseLinearSpace{T<:Real}

The dense-iterate counterpart of `SDPASparseLinearSpace`. Used for the
primal-dual iterates (`xMat`, `zMat`, `invCholeskyX`, `invzMat`, etc.)
where the per-block matrix is always fully populated after the first
IPM iteration.

Mirrors C++ `sdpa::DenseLinearSpace` at
`sdpa_struct.h:174-197`. SOCP fields preserved as
zero-length vectors for layout parallelism with C++ (SOCP path is
`#if 0` in C++ source).

Fields
------
- `SDP_nBlock::Int`:                          SDP block count
- `SOCP_nBlock::Int`:                         always 0
- `LP_nBlock::Int`:                           LP block count
- `SDP_block::Vector{SDPADenseMatrix{T}}`:    one dense matrix per SDP block
- `SOCP_block::Vector{SDPADenseMatrix{T}}`:   empty (SOCP unused)
- `LP_block::Vector{T}`:                      LP entries
"""
mutable struct SDPADenseLinearSpace{T<:Real}
    SDP_nBlock::Int
    SOCP_nBlock::Int
    LP_nBlock::Int
    SDP_block::Vector{SDPADenseMatrix{T}}
    SOCP_block::Vector{SDPADenseMatrix{T}}
    LP_block::Vector{T}

    function SDPADenseLinearSpace{T}(
        SDP_nBlock::Int, LP_nBlock::Int
    ) where {T<:Real}
        SDP_nBlock >= 0 || error("SDP_nBlock must be non-negative")
        LP_nBlock >= 0 || error("LP_nBlock must be non-negative")
        return new{T}(
            SDP_nBlock,
            0,
            LP_nBlock,
            Vector{SDPADenseMatrix{T}}(undef, SDP_nBlock),
            SDPADenseMatrix{T}[],
            _independent_zeros(T, LP_nBlock),
        )
    end
end

"""
    lp_scalar_count(blockStruct) -> Int

Total number of LP scalars in a block vector: every entry `bs < 2`
(including `0`, `1`, and negative LP blocks) contributes `abs(bs)`
scalars, SDP blocks (`bs >= 2`) contribute none. This classification is
the single source of truth shared by the dense-space constructor, the
readers, `SDPAProblem`, and the Schur-complement constraint lists —
use this helper instead of re-deriving the formula (an accidental
mismatch bounds-error loudly rather than silently disagreeing).
"""
@inline lp_scalar_count(blockStruct::AbstractVector{Int}) =
    sum(bs -> bs >= 2 ? 0 : abs(bs), blockStruct; init=0)

"""
    SDPADenseLinearSpace{T}(blockStruct::AbstractVector{Int})

Construct a `DenseLinearSpace` from a signed `blockStruct` vector. Each
positive entry `>= 2` becomes an SDP block of size `bs × bs`; each
entry `<= 0` or `== 1` is an LP block contributing `abs(bs)` separate
scalar slots in `LP_block` (e.g. an entry of `-k` means k LP scalars,
not 1).

The block partitioning rule matches C++ DenseLinearSpace::initialize at
`sdpa_struct.cpp:850-900`.
"""
function SDPADenseLinearSpace{T}(
    blockStruct::AbstractVector{Int}
) where {T<:Real}
    sdp_indices = findall(bs -> bs >= 2, blockStruct)
    # LP block scalar count: each LP block of size |bs| contributes |bs|
    # separate LP scalars (NOT 1). A blockStruct entry of -174 means
    # 174 LP variables, not 1. This matches C++ DenseLinearSpace::initialize.
    # init=0 so an EMPTY blockStruct (zero-block problem, e.g. an
    # empty MOI model) builds a valid 0-LP space instead of throwing
    # "reducing over an empty collection".
    nLP = lp_scalar_count(blockStruct)
    out = SDPADenseLinearSpace{T}(length(sdp_indices), nLP)
    for (k, idx) in enumerate(sdp_indices)
        bs = blockStruct[idx]
        out.SDP_block[k] = SDPADenseMatrix{T}(bs, bs)
    end
    return out
end

"""
    setelement_sdp!(D, block, i, j, value)

Set the `(i, j)` entry of the SDP block `block` in `D` to `value`.
Mirrors C++ DenseLinearSpace::setElement_SDP.
"""
function setelement_sdp!(
    D::SDPADenseLinearSpace{T}, block::Int, i::Int, j::Int, value::T
) where {T<:Real}
    1 <= block <= D.SDP_nBlock ||
        error("SDP block index $block out of [1, $(D.SDP_nBlock)]")
    # The 2-arg constructor leaves SDP_block unassigned; the first
    # write would die on a raw UndefRefError with no context. Guard with
    # a single isassigned check (happy-path cost unchanged).
    isassigned(D.SDP_block, block) || error(
        "SDPADenseLinearSpace: SDP block $block was never sized — construct via SDPADenseLinearSpace{T}(blockStruct) or assign SDP_block[$block] first"
    )
    D.SDP_block[block][i, j] = value
    return D
end

function setelement_lp!(
    D::SDPADenseLinearSpace{T}, block::Int, value::T
) where {T<:Real}
    1 <= block <= D.LP_nBlock ||
        error("LP block index $block out of [1, $(D.LP_nBlock)]")
    D.LP_block[block] = value
    return D
end

function setzero!(D::SDPADenseLinearSpace{T}) where {T<:Real}
    for k in 1:D.SDP_nBlock
        if isassigned(D.SDP_block, k)
            setzero!(D.SDP_block[k])
        end
    end
    _fill_independent!(D.LP_block, zero(T))
    return D
end

function setidentity!(D::SDPADenseLinearSpace{T}, scalar::Real=one(T)) where {T<:Real}
    s = T(scalar)
    for k in 1:D.SDP_nBlock
        if isassigned(D.SDP_block, k)
            setidentity!(D.SDP_block[k], s)
        end
    end
    _fill_independent!(D.LP_block, s)
    return D
end

function Base.copyto!(
    dst::SDPADenseLinearSpace{T}, src::SDPADenseLinearSpace{T}
) where {T<:Real}
    dst.SDP_nBlock == src.SDP_nBlock ||
        throw(DimensionMismatch("SDP_nBlock mismatch"))
    dst.LP_nBlock == src.LP_nBlock ||
        throw(DimensionMismatch("LP_nBlock mismatch"))
    for k in 1:dst.SDP_nBlock
        if isassigned(src.SDP_block, k)
            if !isassigned(dst.SDP_block, k)
                dst.SDP_block[k] = copy(src.SDP_block[k])
            else
                copyto!(dst.SDP_block[k], src.SDP_block[k])
            end
        elseif isassigned(dst.SDP_block, k)
            # src block k is unassigned — zero the stale dst block so
            # copyto! leaves dst == src (setzero! fills de_ele with
            # independent zeros).
            setzero!(dst.SDP_block[k])
        end
    end
    _copyto_independent!(dst.LP_block, src.LP_block)
    return dst
end

function Base.copy(D::SDPADenseLinearSpace{T}) where {T<:Real}
    out = SDPADenseLinearSpace{T}(D.SDP_nBlock, D.LP_nBlock)
    for k in 1:D.SDP_nBlock
        if isassigned(D.SDP_block, k)
            out.SDP_block[k] = copy(D.SDP_block[k])
        end
    end
    _copyto_independent!(out.LP_block, D.LP_block)
    return out
end
