"""
    DirectionParameter{T<:Real}

Mutable single-field struct holding the Mehrotra centering parameter β.
Mirrors C++ `sdpa::DirectionParameter` at
`sdpa_parts.h:134-144`.

β controls how aggressively the IPM step targets the central path:
  - β = 0 means "affine-scaling" (no centering; pure feasibility)
  - β = 1 means "fully centered" (stay at the current μ)
  - β > 1 means "stretch toward larger μ" (re-center from late IPM)

The Mehrotra predictor-corrector scheme computes β twice per iteration:
once in the predictor (a fixed value depending on phase) and once in
the corrector (a value derived from the predictor's predicted step).
"""
mutable struct DirectionParameter{T<:Real}
    value::T

    # Inner constructor: default β = 1 (one(T)). Using `new` avoids the
    # recursion trap of an outer constructor that calls itself.
    function DirectionParameter{T}(betaStar::Real=one(T)) where {T<:Real}
        return new{T}(T(betaStar))
    end
end

"""
    mehrotra_predictor!(beta, phase, reduction, params)

Predictor β selection. Mirrors C++ `DirectionParameter::MehrotraPredictor`
at `sdpa_parts.cpp:545-555`.

Algorithm:
  - If `phase == pdFEAS` (both feasible): β = 0 (pure affine-scaling
    predictor toward optimum).
  - Else (some residual infeasible): β = `params.betaBar`.
    - Sub-case: if `reduction.switchType == REDUCTION_OFF` (both
      residuals are feasible from the switch's perspective, but the
      phase hasn't transitioned to pdFEAS yet — possible at the
      boundary), set β = 2.0 (an "excessive" centering value that
      delays further steps to allow feasibility to settle).

The C++ variable `nu = 2.0` is a constant in the source; we inline it.
"""
function mehrotra_predictor!(
    beta::DirectionParameter{T}, phase::PhaseValue,
    reduction::Switch, params::Parameters{T},
) where {T<:Real}
    nu = T(2)
    if phase === pdFEAS
        beta.value = zero(T)
    else
        beta.value = params.betaBar
        if reduction.switchType === REDUCTION_OFF
            beta.value = nu
        end
    end
    return beta
end

"""
    mehrotra_corrector!(beta, phase, alpha_primal, alpha_dual,
                        current_pt, dx_mat, dz_mat, mu, params)

Corrector β selection — the Mehrotra centering heuristic. Mirrors C++
`DirectionParameter::MehrotraCorrector` at `sdpa_parts.cpp:557-586`.

Inputs are the predictor's step length `alpha` and predictor directions
`dx_mat`, `dz_mat` (Newton step components Δx, Δz).

Algorithm:
  1. Predict the post-step complementarity:
     μ_target = μ_current + (α_p · ⟨X, Δz⟩ + α_d · ⟨Δx, Z⟩ + α_p·α_d · ⟨Δx, Δz⟩) / nDim
  2. β = μ_target / μ_current  (the ratio of predicted to current complementarity)
  3. Mehrotra's "square if <1" rule (acceleration toward optimum):
     if β < 1: β = β²
  4. Clamp into a phase-dependent window:
     - pdFEAS: clamp to [betaStar, 1]
     - else:   clamp to [betaBar, ∞)

!!! note "Deliberate divergence from C++"
    C++ computes `value = mu_target / mu.current` unconditionally
    (sdpa_parts.cpp:567). When `mu.current == 0` (degenerate convergence)
    that divides by zero and yields `NaN`/`Inf`. We short-circuit to
    `value = 1` — β = 1, so the corrector degenerates gracefully to a
    fully-centered step — a loud, well-defined value over silent `NaN`.
"""
function mehrotra_corrector!(
    beta::DirectionParameter{T},
    phase::PhaseValue,
    alpha_primal::T,
    alpha_dual::T,
    current_pt::SDPASolution{T},
    dx_mat::SDPADenseLinearSpace{T},
    dz_mat::SDPADenseLinearSpace{T},
    mu::AverageComplementarity{T},
    params::Parameters{T},
) where {T<:Real}
    nDim = current_pt.nDim
    xMat_vMat = _frobenius_inner_product_dls(current_pt.xMat, dz_mat)
    uMat_zMat = _frobenius_inner_product_dls(dx_mat, current_pt.zMat)
    uMat_vMat = _frobenius_inner_product_dls(dx_mat, dz_mat)
    mu_target = mu.current + (
        alpha_primal * uMat_zMat
        + alpha_dual * xMat_vMat
        + alpha_primal * alpha_dual * uMat_vMat
    ) / nDim

    # Deliberate divergence from C++ (sdpa_parts.cpp:567): C++ divides
    # unconditionally; with mu.current == 0 (degenerate convergence) that
    # yields NaN/Inf. We force value = 1 (β = 1, a fully-centered corrector)
    # so the corrector degenerates gracefully instead of silently corrupting.
    if mu.current == 0
        value = one(T)
    else
        value = mu_target / mu.current
    end
    if value < 1
        value = value * value
    end
    if phase === pdFEAS
        if value < params.betaStar
            value = params.betaStar
        end
        if value > 1
            value = one(T)
        end
    else
        if value < params.betaBar
            value = params.betaBar
        end
    end
    beta.value = value
    return beta
end
