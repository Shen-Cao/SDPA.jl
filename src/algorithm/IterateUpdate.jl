"""
Iterate update + inverse Cholesky recompute.

Mirrors C++ `Solutions::update` at `sdpa_dataset.cpp:120-143`
and `Solutions::computeInverse` at `:96-118`.
"""

"""
    update!(current_pt, alpha, newton, work) -> Bool

Apply one IPM iterate update:
  xMat += alpha.primal * ΔxMat
  yVec += alpha.dual   * ΔyVec
  zMat += alpha.dual   * ΔzMat

Returns `false` if both step lengths are below 1e-4 ("cannot move"
safeguard) OR if the subsequent `compute_inverse!` fails (X or Z lost
positive-definiteness — triggers main-loop break in the caller).

IMPORTANT (C++ parity, `sdpa_dataset.cpp:120-143`): the xMat/yVec/zMat
updates are applied BEFORE the cannot-move check. When `false` is
returned for the cannot-move reason, the iterate therefore HAS the
(negligible) alpha step applied — the post-step point is the legitimate
terminal iterate that the caller reports. The `compute_inverse!` failure
case is the only one where the inverse factors are stale; the caller's
footer reads the iterates (not the cached inverses), so this is safe.

Mirrors C++ `Solutions::update` at `sdpa_dataset.cpp:120-143`.
"""
function update!(
    current_pt::SDPASolution{T}, alpha::StepLength{T},
    newton::NewtonState{T}, work::SDPAWorkVariables{T},
) where {T<:Real}
    # xMat += alpha.primal * DxMat
    for k in 1:current_pt.xMat.SDP_nBlock
        if isassigned(current_pt.xMat.SDP_block, k) &&
           isassigned(newton.DxMat.SDP_block, k)
            xblk = current_pt.xMat.SDP_block[k].de_ele
            dblk = newton.DxMat.SDP_block[k].de_ele
            @inbounds @simd for idx in eachindex(xblk)
                xblk[idx] += alpha.primal * dblk[idx]
            end
        end
    end
    @inbounds for k in eachindex(current_pt.xMat.LP_block)
        if k <= length(newton.DxMat.LP_block)
            current_pt.xMat.LP_block[k] += alpha.primal * newton.DxMat.LP_block[k]
        end
    end

    # yVec += alpha.dual * DyVec
    @inbounds @simd for i in 1:current_pt.mDim
        current_pt.yVec.ele[i] += alpha.dual * newton.DyVec.ele[i]
    end

    # zMat += alpha.dual * DzMat
    for k in 1:current_pt.zMat.SDP_nBlock
        if isassigned(current_pt.zMat.SDP_block, k) &&
           isassigned(newton.DzMat.SDP_block, k)
            zblk = current_pt.zMat.SDP_block[k].de_ele
            dblk = newton.DzMat.SDP_block[k].de_ele
            @inbounds @simd for idx in eachindex(zblk)
                zblk[idx] += alpha.dual * dblk[idx]
            end
        end
    end
    @inbounds for k in eachindex(current_pt.zMat.LP_block)
        if k <= length(newton.DzMat.LP_block)
            current_pt.zMat.LP_block[k] += alpha.dual * newton.DzMat.LP_block[k]
        end
    end

    # "Cannot move" safeguard.
    cannot_move = T(1e-4)
    if alpha.primal < cannot_move && alpha.dual < cannot_move
        return false
    end

    # Recompute inverse Cholesky factors and invzMat.
    return compute_inverse!(current_pt, work)
end

"""
    compute_inverse!(current_pt, work) -> Bool

Recompute the cached inverse Cholesky factors of X and Z after an
iterate update. Also recomputes `invzMat` (the full Z⁻¹).

For X: `invCholeskyX = L_x⁻¹` where `X = L_x · L_x'`.
For Z: `invCholeskyZ = L_z⁻¹` and `invzMat = L_z'⁻¹ · L_z⁻¹ = Z⁻¹`.

Returns `false` if either Cholesky fails (X or Z lost positive-
definiteness). The caller must handle this (typically by breaking the
main loop or reducing the step length).

Mirrors C++ `Solutions::computeInverse` at `sdpa_dataset.cpp:96-118`.
"""
function compute_inverse!(
    current_pt::SDPASolution{T}, work::SDPAWorkVariables{T},
) where {T<:Real}
    # Dispatch to threaded variant when multiple threads are available.
    if Threads.nthreads() > 1
        ok_x = threaded_get_inv_chol!(
            current_pt.invCholeskyX, current_pt.xMat, work.DLS1,
        )
        if !ok_x
            return false
        end
        ok_z = threaded_get_inv_chol_and_inv!(
            current_pt.invCholeskyZ, current_pt.invzMat, current_pt.zMat, work.DLS2,
        )
        if !ok_z
            return false
        end
    else
        ok_x = get_inv_chol!(
            current_pt.invCholeskyX, current_pt.xMat, work.DLS1,
        )
        if !ok_x
            return false
        end
        ok_z = get_inv_chol_and_inv!(
            current_pt.invCholeskyZ, current_pt.invzMat, current_pt.zMat, work.DLS2,
        )
        if !ok_z
            return false
        end
    end

    return true
end
