"""
Iterate update + inverse Cholesky recompute.

Mirrors C++ `Solutions::update` at `sdpa_dataset.cpp:120-143`
and `Solutions::computeInverse` at `:96-118`.
"""

# These internal outcomes distinguish a valid terminal step from a
# rejected candidate. Only accepted points may enter the solution log.
@enum _IterateResult _MOVED _TINY_STEP _NOT_POSDEF _NONFINITE_STEP

function _trial_axpy!(out::AbstractArray{T}, x::AbstractArray{T}, alpha::T,
                      dx::AbstractArray{T}) where {T<:Real}
    finite = true
    @inbounds @simd for i in eachindex(out, x, dx)
        v = x[i] + alpha * dx[i]
        out[i] = v
        finite &= isfinite(v)
    end
    return finite
end

function _trial_axpy!(out::SDPADenseLinearSpace{T}, x::SDPADenseLinearSpace{T},
                      alpha::T, dx::SDPADenseLinearSpace{T}) where {T<:Real}
    for k in eachindex(out.SDP_block, x.SDP_block, dx.SDP_block)
        _trial_axpy!(out.SDP_block[k].de_ele, x.SDP_block[k].de_ele, alpha,
                     dx.SDP_block[k].de_ele) || return false
    end
    return _trial_axpy!(out.LP_block, x.LP_block, alpha, dx.LP_block)
end

"""
    _update_iterate!(pt, alpha, newton, work) -> _IterateResult

Stage X, Z and y in the three Newton RHS buffers. Their contents expire
after the corrector direction is recovered and the next predictor writes
them before reading them. No trial point aliases the current iterate.
Validate the candidate via the *usual* inverse Cholesky computation: the
accepted first-try path does not require an extra factorization or copy.
A viable tiny step is still applied, matching the C++ stagnation rule.
"""
function _update_iterate!(pt::SDPASolution{T}, alpha::StepLength{T},
                          newton::NewtonState{T}, work::SDPAWorkVariables{T}) where {T<:Real}
    x, z, y = newton.r_zinvMat, newton.x_rd_zinvMat, newton.gvec
    if !(_trial_axpy!(x, pt.xMat, alpha.primal, newton.DxMat) &&
         _trial_axpy!(z, pt.zMat, alpha.dual, newton.DzMat) &&
         _trial_axpy!(y.ele, pt.yVec.ele, alpha.dual, newton.DyVec.ele))
        return _NONFINITE_STEP
    end

    if !_inverse_for!(pt, x, z, work)
        # Candidate factors may have been partially written. Restore them
        # from the unmodified last accepted X and Z on this failure path.
        compute_inverse!(pt, work) || error("last accepted iterate has no positive-definite factors")
        return _NOT_POSDEF
    end
    pt.xMat, newton.r_zinvMat = x, pt.xMat
    pt.zMat, newton.x_rd_zinvMat = z, pt.zMat
    pt.yVec, newton.gvec = y, pt.yVec
    return max(alpha.primal, alpha.dual) < T(1e-4) ? _TINY_STEP : _MOVED
end

"""
    update!(current_pt, alpha, newton, work) -> Bool

Apply a validated primal/dual step and refresh its inverse factors.
Returns true if the solve should continue. A valid tiny step is kept;
a non-finite or non-PD candidate leaves the last accepted point intact.
"""
function update!(pt::SDPASolution{T}, alpha::StepLength{T},
                 newton::NewtonState{T}, work::SDPAWorkVariables{T}) where {T<:Real}
    return _update_iterate!(pt, alpha, newton, work) == _MOVED
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
    return _inverse_for!(current_pt, current_pt.xMat, current_pt.zMat, work)
end

# Write the factors of the supplied (possibly still tentative) matrices.
function _inverse_for!(current_pt::SDPASolution{T}, x::SDPADenseLinearSpace{T},
                       z::SDPADenseLinearSpace{T}, work::SDPAWorkVariables{T}) where {T<:Real}
    # Dispatch to threaded variant when multiple threads are available.
    if Threads.nthreads() > 1
        ok_x = threaded_get_inv_chol!(
            current_pt.invCholeskyX, x, work.DLS1,
        )
        if !ok_x
            return false
        end
        ok_z = threaded_get_inv_chol_and_inv!(
            current_pt.invCholeskyZ, current_pt.invzMat, z, work.DLS2,
        )
        if !ok_z
            return false
        end
    else
        ok_x = get_inv_chol!(
            current_pt.invCholeskyX, x, work.DLS1,
        )
        if !ok_x
            return false
        end
        ok_z = get_inv_chol_and_inv!(
            current_pt.invCholeskyZ, current_pt.invzMat, z, work.DLS2,
        )
        if !ok_z
            return false
        end
    end

    return true
end
