"""
    StepLength{T<:Real}

Primal and dual step lengths for one IPM iteration. Mirrors C++
`sdpa::StepLength` at `sdpa_parts.h:116-132`.

Fields
------
- `primal::T`:  step length for X (the primal variable)
- `dual::T`:    step length for (y, Z) (the dual variables)

The step lengths are computed by the fraction-to-boundary rule:
  α = min(α_max, -1/λ_min(L · Δ · L'))

where L is the inverse Cholesky factor of X (or Z) and Δ is the Newton
direction. This guarantees X + α·Δx ⪰ 0 (and Z + α·Δz ⪰ 0).
"""
mutable struct StepLength{T<:Real}
    primal::T
    dual::T
end

StepLength{T}() where {T<:Real} = StepLength{T}(zero(T), zero(T))

"""
    compute_step_length!(alpha, current_pt, newton, work) where T

Compute the fraction-to-boundary step lengths via minimum eigenvalues.

Algorithm (mirrors C++ `StepLength::computeStepLength` at
`sdpa_parts.cpp:298-360`):
  alphaBD = 100  (hard cap)
  λ_x = min eigenvalue of L_x · Δx · L_x'
  if -λ_x > 1/alphaBD:  primal = -1/λ_x
  else:                 primal = alphaBD
  Same for dual with L_z, Δz.

The cap at `alphaBD = 100` prevents wild steps in late iterations.

A NaN min-eigenvalue estimate (Lanczos/QR breakdown propagating NaN)
is a LOUD `error`: NaN fails the `-λ > 1/alphaBD` comparison below,
which would silently award the full `alphaBD` step — indistinguishable
from a genuinely safe direction (loud-failure philosophy, cf.
DirectionParameter.jl / SolveInfo.jl).
"""
function compute_step_length!(
    alpha::StepLength{T}, current_pt::SDPASolution{T},
    newton::NewtonState{T}, work::SDPAWorkVariables{T},
) where {T<:Real}
    alphaBD = T(100)

    # Primal step: min eigenvalue of L_x · Δx · L_x'
    if Threads.nthreads() > 1
        min_x = threaded_min_eigen_dls(current_pt.invCholeskyX, newton.DxMat, work)
    else
        min_x = _min_eigen_dls(current_pt.invCholeskyX, newton.DxMat, work)
    end
    # A NaN min-eigen estimate must fail LOUD — NaN fails the
    # `-min_x > 1/alphaBD` test below and would silently award the full
    # alphaBD step, indistinguishable from a genuinely safe direction.
    isnan(min_x) && error("compute_step_length!: primal min-eigenvalue estimate is NaN (Lanczos/QR breakdown); refusing to take a step")
    if -min_x > one(T) / alphaBD
        alpha.primal = -one(T) / min_x
    else
        alpha.primal = alphaBD
    end

    # Dual step: min eigenvalue of L_z · Δz · L_z'
    if Threads.nthreads() > 1
        min_z = threaded_min_eigen_dls(current_pt.invCholeskyZ, newton.DzMat, work)
    else
        min_z = _min_eigen_dls(current_pt.invCholeskyZ, newton.DzMat, work)
    end
    # Dual twin of the primal NaN guard above.
    isnan(min_z) && error("compute_step_length!: dual min-eigenvalue estimate is NaN (Lanczos/QR breakdown); refusing to take a step")
    if -min_z > one(T) / alphaBD
        alpha.dual = -one(T) / min_z
    else
        alpha.dual = alphaBD
    end

    return alpha
end

"""
Helper: compute min eigenvalue across all SDP blocks of L · X · L'.
Returns the minimum across blocks. For LP blocks, the "eigenvalue"
is just L²·X (scalar).
"""
function _min_eigen_dls(
    l_mat::SDPADenseLinearSpace{T}, x_mat::SDPADenseLinearSpace{T},
    work::SDPAWorkVariables{T},
) where {T<:Real}
    # Sentinel is T(1e50) for C++ parity (min_min = 1.0e+50 in
    # sdpa_linear.cpp getMinEigen); under Float32/Float16 it converts to
    # Inf, which is expected and semantics-preserving: the sentinel is
    # only returned when no assigned block ever writes it, and the
    # consumer test `-min_val > 1/alphaBD` is false for 1e50 and Inf
    # alike (full-step "no negative curvature" reading). Keep this and
    # the threaded twin (threaded_min_eigen_dls) in lockstep.
    min_val = T(1e50)
    for k in 1:x_mat.SDP_nBlock
        if isassigned(x_mat.SDP_block, k) && isassigned(l_mat.SDP_block, k)
            L = l_mat.SDP_block[k].de_ele
            X = x_mat.SDP_block[k].de_ele
            n = size(X, 1)
            if n > 20
                lam = min_eigen_lanczos(L, X)
            else
                # QR path: compute L * X * L' explicitly.
                # Rgemm! MUTATES its C in place (mp_zero!/mp_fma! per
                # element), so the outputs must have INDEPENDENT, DEFINED
                # BigFloat elements — `similar` would leave them #undef.
                # The helper dispatches to `similar` for isbits T.
                tmp = _independent_zeros(T, n, n)
                out = _independent_zeros(T, n, n)
                # Explicit Rgemm! on the raw matrices (same kernel as
                # min_eigen_dispatch in min_eigen.jl — MUST match
                # threaded_min_eigen_dls for cross-thread-count
                # bit-identity). tmp = X * L', out = L * tmp = L * X * L'.
                Rgemm!("N", "T", n, n, n, one(T), X, L, zero(T), tmp)
                Rgemm!("N", "N", n, n, n, one(T), L, tmp, zero(T), out)
                lam = min_eigen_qr(out)
            end
            # NaN must PROPAGATE, not be dropped: `NaN < min_val` is
            # false, so a NaN block estimate (Lanczos/QR breakdown)
            # would silently leave the 1e50 sentinel — and
            # compute_step_length! would take it for a safe direction
            # (full alphaBD step). Sticky by design: once min_val is
            # NaN, later finite lams no longer compare in
            # (`lam < NaN` is false), so the NaN reaches the guard in
            # compute_step_length! and errors loudly.
            if isnan(lam) || lam < min_val
                min_val = lam
            end
        end
    end
    # LP blocks: eigenvalue = L²·X (scalar).
    for k in 1:x_mat.LP_nBlock
        if k <= length(l_mat.LP_block) && k <= length(x_mat.LP_block)
            val = l_mat.LP_block[k]^2 * x_mat.LP_block[k]
            # LP twin of the SDP sticky-NaN above — a NaN
            # LP entry must PROPAGATE to the loud guard in
            # compute_step_length!, not be dropped by the comparison
            # (`NaN < min_val` is false). Strict no-op for finite val.
            if isnan(val) || val < min_val
                min_val = val
            end
        end
    end
    return min_val
end

"""
    mehrotra_predictor!(alpha, problem, current_pt, phase, newton, work)

Predictor step length. Both step lengths are set to 0.9; the clamps
below are unreachable (0.9 < 1, 0.9 == 0.9) and are kept as a
faithful mirror of C++ StepLength::MehrotraPredictor.

Mirrors C++ `StepLength::MehrotraPredictor` at
`sdpa_parts.cpp:362-410`.
"""
function mehrotra_predictor!(
    alpha::StepLength{T}, problem::SDPAProblem{T}, current_pt::SDPASolution{T},
    phase::PhaseValue, newton::NewtonState{T}, work::SDPAWorkVariables{T},
) where {T<:Real}
    # Default: primal = dual = 0.9.
    alpha.primal = T(0.9)
    alpha.dual = T(0.9)

    # Primal feasibility / objective capping.
    if phase === noINFO || phase === dFEAS
        # Primal infeasible: cap at 1.
        if alpha.primal > 1
            alpha.primal = one(T)
        end
    else
        # Primal feasible: check if step increases <C, X>.
        inc_primal = _inner_product_C_xMat(problem, newton.DxMat)
        if inc_primal > 0
            if alpha.primal > alpha.dual
                alpha.primal = alpha.dual
            end
            if alpha.primal > 1
                alpha.primal = one(T)
            end
        end
    end

    # Dual feasibility / objective capping.
    if phase === noINFO || phase === pFEAS
        if alpha.dual > 1
            alpha.dual = one(T)
        end
    else
        inc_dual = _inner_product_b_yVec(problem, newton.DyVec)
        if inc_dual < 0
            if alpha.dual > alpha.primal
                alpha.dual = alpha.primal
            end
            if alpha.dual > 1
                alpha.dual = one(T)
            end
        end
    end

    return alpha
end

"""
    mehrotra_corrector!(alpha, problem, current_pt, phase, reduction,
                        newton, mu, theta, work, params)

Corrector step length. Calls `compute_step_length!`, scales by
`gammaStar`, applies feasibility/objective capping, then runs the
feasibility loop and the pdFEAS ratio capping.

Mirrors C++ `StepLength::MehrotraCorrector` at
`sdpa_parts.cpp:412-523`.
"""
function mehrotra_corrector!(
    alpha::StepLength{T}, problem::SDPAProblem{T}, current_pt::SDPASolution{T},
    phase::PhaseValue, reduction::Switch, newton::NewtonState{T},
    mu::AverageComplementarity{T}, theta::RatioInitResCurrentRes{T},
    work::SDPAWorkVariables{T}, params::Parameters{T},
) where {T<:Real}
    xi = T(3)
    nDim = current_pt.nDim

    compute_step_length!(alpha, current_pt, newton, work)

    # Scale by gammaStar.
    alpha.primal *= params.gammaStar
    alpha.dual *= params.gammaStar

    # Feasibility / objective capping (same logic as predictor).
    if phase === noINFO || phase === dFEAS
        if alpha.primal > 1
            alpha.primal = one(T)
        end
    else
        inc_primal = _inner_product_C_xMat(problem, newton.DxMat)
        if inc_primal > 0
            if alpha.primal > alpha.dual
                alpha.primal = alpha.dual
            end
            if alpha.primal > 1
                alpha.primal = one(T)
            end
        end
    end
    if phase === noINFO || phase === pFEAS
        if alpha.dual > 1
            alpha.dual = one(T)
        end
    else
        inc_dual = _inner_product_b_yVec(problem, newton.DyVec)
        if inc_dual < 0
            if alpha.dual > alpha.primal
                alpha.dual = alpha.primal
            end
            if alpha.dual > 1
                alpha.dual = one(T)
            end
        end
    end

    # Feasibility loop: attain feasibility before mu reduction.
    if reduction.switchType === REDUCTION_ON &&
       (phase === noINFO || phase === pFEAS || phase === dFEAS)
        xMat_vMat = _frobenius_inner_product_dls(current_pt.xMat, newton.DzMat)
        uMat_zMat = _frobenius_inner_product_dls(newton.DxMat, current_pt.zMat)
        uMat_vMat = _frobenius_inner_product_dls(newton.DxMat, newton.DzMat)

        theta_max = max((1 - alpha.primal) * theta.primal,
                        (1 - alpha.dual) * theta.dual)
        mu_new = mu.current + (
            alpha.primal * uMat_zMat
            + alpha.dual * xMat_vMat
            + alpha.primal * alpha.dual * uMat_vMat
        ) / nDim

        while theta_max * theta_max * mu.initial > xi * mu_new
            alpha_max = T(0.95) * max(alpha.primal, alpha.dual)
            alpha.primal = min(alpha.primal, alpha_max)
            alpha.dual = min(alpha.dual, alpha_max)
            theta_max = max((1 - alpha.primal) * theta.primal,
                            (1 - alpha.dual) * theta.dual)
            mu_new = mu.current + (
                alpha.primal * uMat_zMat
                + alpha.dual * xMat_vMat
                + alpha.primal * alpha.dual * uMat_vMat
            ) / nDim
            if alpha.primal < T(1e-6) && alpha.dual < T(1e-6)
                break
            end
        end
    end

    # pdFEAS ratio capping: keep objValPrimal >= objValDual.
    if phase === pdFEAS
        obj_d = _inner_product_b_yVec(problem, current_pt.yVec)
        obj_p = _inner_product_C_xMat(problem, current_pt.xMat)
        inc_dual = _inner_product_b_yVec(problem, newton.DyVec) * alpha.dual
        inc_primal = _inner_product_C_xMat(problem, newton.DxMat) * alpha.primal
        denom = inc_primal - inc_dual
        if denom != 0
            max_ratio = (obj_d - obj_p) / denom
            if max_ratio > 0 && max_ratio < 1
                alpha.primal *= max_ratio
                alpha.dual *= max_ratio
            end
        end
    end

    return alpha
end
