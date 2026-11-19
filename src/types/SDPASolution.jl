"""
    SDPASolution{T<:Real}

The primal-dual iterate at any IPM step: `(X, y, Z)` plus cached
inverses and Cholesky factors that the next Newton step needs.

Mirrors C++ `sdpa::Solutions` at `sdpa_dataset.h:46-77`
and constructor at `sdpa_dataset.cpp:35-95`.

Fields
------
- `nDim::Int`:                              total variable dimension
  (`Σ_k abs(blockStruct[k])` over ALL blocks, SDP and LP; SDP `nDim`
  counts the block size, not the packed upper-triangular length)
- `mDim::Int`:                              constraint count (length of yVec)
- `xMat::SDPADenseLinearSpace{T}`:          primal variable X
- `zMat::SDPADenseLinearSpace{T}`:          dual slack Z
- `yVec::SDPAVector{T}`:                    dual variable y
- `invCholeskyX::SDPADenseLinearSpace{T}`:  lower-triangular L_x with
  `L_x * L_x' = X⁻¹` (caches the inverse Cholesky of X)
- `invCholeskyZ::SDPADenseLinearSpace{T}`:  same for Z
- `invzMat::SDPADenseLinearSpace{T}`:       full `Z⁻¹ = L_z * L_z'`

Constructor `SDPASolution{T}(m, blockStruct, lambdaStar)` initializes
the central-path starting point:
  - xMat = lambdaStar · I
  - zMat = lambdaStar · I
  - yVec = 0
  - invCholeskyX = I / sqrt(lambdaStar)
  - invCholeskyZ = I / sqrt(lambdaStar)
  - invzMat = I / lambdaStar

Mirrors C++ Solutions::initialize at `sdpa_dataset.cpp:35-95`.
"""
mutable struct SDPASolution{T<:Real}
    nDim::Int
    mDim::Int
    xMat::SDPADenseLinearSpace{T}
    zMat::SDPADenseLinearSpace{T}
    yVec::SDPAVector{T}
    invCholeskyX::SDPADenseLinearSpace{T}
    invCholeskyZ::SDPADenseLinearSpace{T}
    invzMat::SDPADenseLinearSpace{T}
end

function SDPASolution{T}(
    m::Int, blockStruct::AbstractVector{Int}, lambdaStar::Real
) where {T<:Real}
    m >= 0 || error("m must be non-negative")
    lambda = T(lambdaStar)
    lambda > 0 || error("lambdaStar must be positive, got $lambdaStar")

    # nDim = sum of |bs| over ALL blocks (SDP + LP). Matches C++
    # Solutions::initialize at sdpa_dataset.cpp:37-44:
    #   nDim = 0;
    #   for (l in SDP_blocks) nDim += SDP_blockStruct[l];
    #   nDim += LP_nBlock;
    # SDP blocks (bs >= 2) contribute their matrix size; LP blocks
    # (bs < 0) contribute |bs| (the LP scalar count); LP blocks of
    # size 1 (bs == 1) contribute 1. The common formula is nDim = Σ|bs|.
    nDim = 0
    for bs in blockStruct
        nDim += abs(bs)
    end

    xMat = SDPADenseLinearSpace{T}(blockStruct)
    zMat = SDPADenseLinearSpace{T}(blockStruct)
    invCholeskyX = SDPADenseLinearSpace{T}(blockStruct)
    invCholeskyZ = SDPADenseLinearSpace{T}(blockStruct)
    invzMat = SDPADenseLinearSpace{T}(blockStruct)

    inv_lambda = inv(lambda)
    sqrt_inv_lambda = inv(sqrt(lambda))

    # Per-block initialization. SDP blocks (bs >= 2) carry a matrix
    # initialized to lambda*I. LP blocks (bs < 2) store scalars
    # initialized to lambda (matching C++ Solutions::initialize at
    # sdpa_dataset.cpp:35-95 which sets ALL LP variables to lambdaStar).
    sdp_idx = 0
    for k in eachindex(blockStruct)
        bs = blockStruct[k]
        if bs >= 2
            sdp_idx += 1
            # xMat = lambda * I
            setidentity!(xMat.SDP_block[sdp_idx], lambda)
            # zMat = lambda * I
            setidentity!(zMat.SDP_block[sdp_idx], lambda)
            # invCholeskyX = I / sqrt(lambda)
            setidentity!(invCholeskyX.SDP_block[sdp_idx], sqrt_inv_lambda)
            # invCholeskyZ = I / sqrt(lambda)
            setidentity!(invCholeskyZ.SDP_block[sdp_idx], sqrt_inv_lambda)
            # invzMat = I / lambda
            setidentity!(invzMat.SDP_block[sdp_idx], inv_lambda)
        end
    end

    # LP block initialization: all LP variables = lambda (matching C++
    # Solutions::initialize which sets ALL LP entries to lambdaStar).
    # _fill_independent! (instead of fill!) so each BigFloat LP scalar is
    # its own object: fill! would alias every slot to the same `lambda`
    # object, and in-place MPFR mutation of one LP scalar would corrupt
    # all the others (and `lambda` itself).
    _fill_independent!(xMat.LP_block, lambda)
    _fill_independent!(zMat.LP_block, lambda)
    _fill_independent!(invCholeskyX.LP_block, sqrt_inv_lambda)
    _fill_independent!(invCholeskyZ.LP_block, sqrt_inv_lambda)
    _fill_independent!(invzMat.LP_block, inv_lambda)

    yVec = SDPAVector{T}(m)
    setzero!(yVec)

    return SDPASolution{T}(
        nDim, m, xMat, zMat, yVec,
        invCholeskyX, invCholeskyZ, invzMat,
    )
end

function Base.copy(sol::SDPASolution{T}) where {T<:Real}
    return SDPASolution{T}(
        sol.nDim,
        sol.mDim,
        copy(sol.xMat),
        copy(sol.zMat),
        copy(sol.yVec),
        copy(sol.invCholeskyX),
        copy(sol.invCholeskyZ),
        copy(sol.invzMat),
    )
end
