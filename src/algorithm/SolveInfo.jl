"""
Helper: Frobenius inner product `⟨A, B⟩` over the full DenseLinearSpace
(SDP blocks + LP blocks). This is the production implementation used by
`SolveInfo` and `AverageComplementarity`; the linear `JordanAlgebra.trace`
is NOT equivalent (bilinear vs linear) and must not replace it.
"""
function _frobenius_inner_product_dls(
    a::SDPADenseLinearSpace{T}, b::SDPADenseLinearSpace{T}
) where {T<:Real}
    s = zero(T)
    for k in 1:a.SDP_nBlock
        if isassigned(a.SDP_block, k) && isassigned(b.SDP_block, k)
            ak = a.SDP_block[k].de_ele
            bk = b.SDP_block[k].de_ele
            size(ak) == size(bk) ||
                throw(DimensionMismatch("SDP block $k size mismatch"))
            @inbounds for idx in eachindex(ak)
                s += ak[idx] * bk[idx]
            end
        end
    end
    @inbounds for k in eachindex(a.LP_block)
        s += a.LP_block[k] * b.LP_block[k]
    end
    return s
end

"""
    SwitchType

State of the `reduction` switch in the Mehrotra predictor step. Mirrors
C++ `Switch::SwitchType` at `sdpa_parts.h:146-150`.
"""
@enum SwitchType begin
    REDUCTION_ON
    REDUCTION_OFF
end

"""
    Switch

Wrapper around a `SwitchType` field, mutated by `mehrotra_predictor!`
each iteration. Mirrors C++ `Switch` at `sdpa_parts.h:146-156`.
"""
mutable struct Switch
    switchType::SwitchType
end

"""
    mehrotra_predictor!(sw, phase)

Set `sw.switchType = REDUCTION_ON` if the current phase is any of
`noINFO`, `pFEAS`, `dFEAS` (i.e. at least one of primal or dual is
infeasible); else `REDUCTION_OFF`. Mirrors C++ `Switch::MehrotraPredictor`
at `sdpa_parts.cpp:605-613`.

The reduction switch controls whether `RatioInitResCurrentRes` updates
its ratios multiplicatively (when ON) or leaves them unchanged (when OFF).
"""
function mehrotra_predictor!(sw::Switch, phase::PhaseValue)
    if phase === noINFO || phase === pFEAS || phase === dFEAS
        sw.switchType = REDUCTION_ON
    else
        sw.switchType = REDUCTION_OFF
    end
    return sw
end

"""
    AverageComplementarity{T<:Real}

Average complementarity `μ = ⟨X, Z⟩ / nDim`. Mirrors C++
`AverageComplementarity` at `sdpa_parts.h:159-170`.

Fields
------
- `initial::T`: the starting μ (lambdaStar² for the central-path init)
- `current::T`: the current μ, recomputed every iteration

`μ` is the SDP analogue of the LP complementarity measure. As the IPM
converges, `μ → 0` quadratically.
"""
mutable struct AverageComplementarity{T<:Real}
    initial::T
    current::T
end

AverageComplementarity{T}(lambdaStar::Real) where {T<:Real} =
    AverageComplementarity{T}(T(lambdaStar)^2, T(lambdaStar)^2)

"""
    update!(mu, current_pt)

Recompute `current = ⟨X, Z⟩ / nDim`. Mirrors C++
`AverageComplementarity::update` at `sdpa_parts.cpp:647-652`.
"""
function update!(mu::AverageComplementarity{T}, current_pt::SDPASolution{T}) where {T<:Real}
    mu.current = _frobenius_inner_product_dls(current_pt.xMat, current_pt.zMat) / current_pt.nDim
    return mu
end

"""
    RatioInitResCurrentRes{T<:Real}

Ratios `theta.primal = current_primal_residual / initial_primal_residual`
and `theta.dual = current_dual_residual / initial_dual_residual`. Used by
`SolveInfo.update!` to compute the infeasibility measure `rho`. Mirrors
C++ `RatioInitResCurrentRes` at `sdpa_parts.h:172-184`.

These ratios are tracked in two ways:
  1. Multiplicatively: `theta *= |1 - alpha|` each step (only when
     `reduction` is ON — i.e. some residual is still infeasible).
  2. Exactly: `theta = current_res / init_res` (called after every
     iteration, overrides the multiplicative estimate).
"""
mutable struct RatioInitResCurrentRes{T<:Real}
    primal::T
    dual::T
end

"""
    RatioInitResCurrentRes{T}(param, init_res) where {T<:Real}

Initialize the ratios. If `init_res.normPrimalVec < epsilonDash`, set
primal = 0 (already feasible); else primal = 1. Same for dual.
Mirrors C++ `RatioInitResCurrentRes::initialize` at
`sdpa_parts.cpp:693-706`.
"""
function RatioInitResCurrentRes{T}(
    param::Parameters{T}, init_res::SDPAResiduals{T}
) where {T<:Real}
    accuracy = param.epsilonDash
    primal = init_res.normPrimalVec < accuracy ? zero(T) : one(T)
    dual   = init_res.normDualMat   < accuracy ? zero(T) : one(T)
    return RatioInitResCurrentRes{T}(primal, dual)
end

"""
    update!(theta, reduction::Switch, alpha_primal::T, alpha_dual::T)

Multiplicative update — multiply by `|1 - alpha|` per step. Only runs
when `reduction.switchType == REDUCTION_ON`. Mirrors C++
`RatioInitResCurrentRes::update(Switch, StepLength)` at
`sdpa_parts.cpp:708-715`.
"""
function update!(
    theta::RatioInitResCurrentRes{T}, reduction::Switch, alpha_primal::T, alpha_dual::T
) where {T<:Real}
    if reduction.switchType === REDUCTION_ON
        theta.primal = abs((1 - alpha_primal) * theta.primal)
        theta.dual   = abs((1 - alpha_dual)   * theta.dual)
    end
    return theta
end

"""
    update_exact!(theta, init_res, current_res)

Exact update — `theta = current_res / init_res`. Called after every
iteration as a correction to the multiplicative estimate. Uses
`init_res / 1e-10` threshold to avoid divide-by-zero. Mirrors C++
`RatioInitResCurrentRes::update_exact` at `sdpa_parts.cpp:717-727`.
"""
function update_exact!(
    theta::RatioInitResCurrentRes{T},
    init_res::SDPAResiduals{T},
    current_res::SDPAResiduals{T},
) where {T<:Real}
    if init_res.normPrimalVec > T(1e-10)
        theta.primal = current_res.normPrimalVec / init_res.normPrimalVec
    else
        theta.primal = zero(T)
    end
    if init_res.normDualMat > T(1e-10)
        theta.dual = current_res.normDualMat / init_res.normDualMat
    else
        theta.dual = zero(T)
    end
    return theta
end

"""
    SolveInfo{T<:Real}

Per-iteration tracking scalars: objective values, infeasibility ratios,
running minimums of `X0·Z` and `X·Z0`. Used by the phase machine to
detect infeasibility (`rho > 1 + 1e-6`) and unboundedness (`objVal ≤
lowerBound` etc.). Mirrors C++ `SolveInfo` at `sdpa_parts.h:186-209`.

Fields
------
- `rho::T`:        infeasibility indicator (computed each iteration)
- `etaPrimal::T`:  running min of `lambda · trace(Z)` while primal-feasible
- `etaDual::T`:    running min of `lambda · trace(X)` while dual-feasible
- `objValPrimal::T`:  current `⟨C, X⟩`
- `objValDual::T`:    current `⟨b, y⟩`

Adapter fields (no algorithmic weight): the following fields are
populated by `update!` and read by the print/write layers for
`write_result` / `print_one_iteration`. They exist so callers that hold
a `SolveInfo` can render an iteration row without re-threading values
through extra arguments.
- `mu_current::T`:    current complementarity μ
- `theta_primal::T`:  current primal θ
- `theta_dual::T`:    current dual θ
- `alpha_primal::T`:  last primal step length
- `alpha_dual::T`:    last dual step length
- `beta_value::T`:    last β
- `rel_gap::T`:       relative gap `|objP - objD| / max(1, mean)`
- `gap::T`:           absolute gap `|objP - objD|`
- `digits::T`:        `-log10(relative gap)`
- `primal_res::T`:    primal residual norm
- `dual_res::T`:      dual residual norm

`mu_current`, `theta_primal`, `theta_dual`, `rel_gap`, `gap`, `digits`,
`primal_res` and `dual_res` are populated by `update!`. `alpha_primal`,
`alpha_dual` and `beta_value` are NOT (they depend on step-length /
centering results that `update!` does not receive); the caller populates
them in the solve loop after each iteration's step computation (see
`solve!` in `src/solve.jl`).
"""
mutable struct SolveInfo{T<:Real}
    rho::T
    etaPrimal::T
    etaDual::T
    objValPrimal::T
    objValDual::T
    # Adapter fields.
    mu_current::T
    theta_primal::T
    theta_dual::T
    alpha_primal::T
    alpha_dual::T
    beta_value::T
    rel_gap::T
    gap::T
    digits::T
    primal_res::T
    dual_res::T
end

"""
    SolveInfo{T}(problem, current_pt, mu0, omegaStar)

Constructor for iteration 0. Initializes `objValPrimal = ⟨C, xMat⟩`,
`objValDual = ⟨b, yVec⟩`, `rho = 1.0`, and
`etaPrimal = etaDual = omegaStar * nDim * mu0`.
Mirrors C++ `SolveInfo::initialize` at `sdpa_parts.cpp:735-745`.

All adapter fields initialize to `zero(T)`; the ones `update!`
produces (mu/theta/gap/residuals) get populated once the iteration loop
starts, while `alpha_primal`/`alpha_dual`/`beta_value` are populated by
the caller in the solve loop (see `solve!`).
"""
function SolveInfo{T}(
    problem::SDPAProblem{T}, current_pt::SDPASolution{T},
    mu0::Real, omega_star::Real,
) where {T<:Real}
    obj_p = _inner_product_C_xMat(problem, current_pt.xMat)
    obj_d = _inner_product_b_yVec(problem, current_pt.yVec)
    eta_init = T(omega_star) * T(current_pt.nDim) * T(mu0)
    z = zero(T)
    return SolveInfo{T}(
        one(T),         # rho = 1.0 (C++ sdpa_parts.cpp:740)
        eta_init,       # etaPrimal = omegaStar * nDim * mu0 (C++ :741)
        eta_init,       # etaDual   = omegaStar * nDim * mu0 (C++ :742)
        obj_p,
        obj_d,
        # Adapter fields — populated by update! during iteration loop.
        z, z, z, z, z, z, z, z, z, z, z,
    )
end

# Objective inner-product helpers — production implementation (the
# linear JordanAlgebra.trace is not an equivalent replacement).
# objValPrimal = ⟨C, xMat⟩ where C is sparse, xMat is dense.
function _inner_product_C_xMat(problem::SDPAProblem{T}, xMat::SDPADenseLinearSpace{T}) where {T<:Real}
    s = zero(T)
    for sdp_k in 1:problem.C.SDP_sp_nBlock
        global_k = problem.C.SDP_sp_index[sdp_k]
        C_block = problem.C.SDP_sp_block[sdp_k]
        x_block = _lookup_sdp_block(xMat, global_k, problem.blockStruct)
        if C_block.type === :sparse
            for nz in 1:C_block.nonZeroEffect
                r = C_block.row_index[nz]
                c = C_block.column_index[nz]
                v = C_block.sp_ele[nz]
                s += v * x_block.de_ele[r, c]
                if r != c
                    s += v * x_block.de_ele[c, r]
                end
            end
        else
            size(C_block.de_ele) == size(x_block.de_ele) ||
                throw(DimensionMismatch("C block size mismatch"))
            @inbounds for k in eachindex(C_block.de_ele)
                s += C_block.de_ele[k] * x_block.de_ele[k]
            end
        end
    end
    # LP blocks: iterate C's compact nonzero list only.
    for lp_pos in 1:problem.C.LP_nnz
        lp_k = problem.C.LP_sp_index[lp_pos]
        s += problem.C.LP_sp_val[lp_pos] * xMat.LP_block[lp_k]
    end
    return s
end

# objValDual = ⟨b, yVec⟩ (just a dot product).
function _inner_product_b_yVec(problem::SDPAProblem{T}, yVec::SDPAVector{T}) where {T<:Real}
    s = zero(T)
    @inbounds for k in 1:problem.m
        s += problem.b.ele[k] * yVec.ele[k]
    end
    return s
end

"""
    update!(solve_info, lambda_star, problem, current_pt, current_res, mu, theta, params)

Recompute per-iteration scalars. The `lambda_star` is the
central-path parameter (`X = lambda · I` initial point); the formula
computes `X0 · Z` and `X · Z0` for the rho calculation assuming that
the initial point is `lambda · I` in both primal and dual. Mirrors C++
`SolveInfo::update(double lambda, ...)` at `sdpa_parts.cpp:798-867`.

Formula (C++ line numbers in brackets):

  obj_p = ⟨C, X⟩                                         [800-801]
  obj_d = ⟨b, y⟩                                         [802]
  x0z0 = nDim * mu.initial   (= initial complementarity) [806]
  xzMat = nDim * mu.current (= current  complementarity) [807]
  x0zMat = sum over SDP blocks of lambda * trace(Z)      [815-820]
         + sum over LP blocks of lambda * Z_lp            [824-826]
  xMatz0 = sum over SDP blocks of lambda * trace(X)      [815-820]
         + sum over LP blocks of lambda * X_lp            [824-826]

  accuracy = epsilonDash
  etaPrimal = min(etaPrimal, xMatz0)  IF primal feasible   [832-836]
  etaDual   = min(etaDual,   x0zMat)  IF dual feasible     [837-841]

  rho formula depends on which residual is infeasible (3 cases) —
  see the three branches below.

!!! note "Deliberate divergence from C++"
    Each rho denominator has the form `(...)·eta + nDim·mu.current`. In
    degenerate convergence the denominator is exactly zero and C++ divides,
    producing `NaN` that poisons the phase machine's `rho > 1 + 1e-6` test
    (sdpa_parts.cpp:845-857). We set `rho = typemax(T)` instead — a loud,
    non-NaN sentinel that makes `update_check!` declare the iterate
    infeasible (loud failure) rather than hang on `NaN`.
"""
function update!(
    solve_info::SolveInfo{T},
    lambda_star::Real,
    problem::SDPAProblem{T},
    current_pt::SDPASolution{T},
    current_res::SDPAResiduals{T},
    mu::AverageComplementarity{T},
    theta::RatioInitResCurrentRes{T},
    params::Parameters{T},
) where {T<:Real}
    lambda = T(lambda_star)
    nDim = current_pt.nDim

    # Objective values.
    solve_info.objValPrimal = _inner_product_C_xMat(problem, current_pt.xMat)
    solve_info.objValDual = _inner_product_b_yVec(problem, current_pt.yVec)

    # Inner products with the initial point (X0 = lambda*I, Z0 = lambda*I).
    # x0zMat = sum lambda * Z_diag, xMatz0 = sum lambda * X_diag, per-block.
    x0zMat = zero(T)
    xMatz0 = zero(T)
    for sdp_k in 1:current_pt.xMat.SDP_nBlock
        bs = size(current_pt.xMat.SDP_block[sdp_k].de_ele, 1)
        xblk = current_pt.xMat.SDP_block[sdp_k].de_ele
        zblk = current_pt.zMat.SDP_block[sdp_k].de_ele
        @inbounds for i in 1:bs
            x0zMat += lambda * zblk[i, i]
            xMatz0 += lambda * xblk[i, i]
        end
    end
    @inbounds for k in 1:current_pt.xMat.LP_nBlock
        x0zMat += lambda * current_pt.zMat.LP_block[k]
        xMatz0 += lambda * current_pt.xMat.LP_block[k]
    end

    accuracy = params.epsilonDash
    primal_feasible = current_res.normPrimalVec <= accuracy
    dual_feasible   = current_res.normDualMat   <= accuracy

    if primal_feasible && xMatz0 < solve_info.etaPrimal
        solve_info.etaPrimal = xMatz0
    end
    if dual_feasible && x0zMat < solve_info.etaDual
        solve_info.etaDual = x0zMat
    end

    primal_theta = theta.primal
    dual_theta = theta.dual
    omega = params.omegaStar
    xzMat = nDim * mu.current

    # rho computation: depends on which residual is infeasible.
    # Deliberate divergence from C++ (sdpa_parts.cpp:845-857): if a rho
    # denominator is exactly zero (degenerate convergence) C++ divides and
    # silently yields NaN, which poisons update_check!'s `rho > 1 + 1e-6`
    # test. We set rho = typemax(T) (Inf) instead — a loud, non-NaN
    # sentinel that makes update_check! declare the iterate infeasible.
    if !primal_feasible && dual_feasible
        # primal infeasible, dual feasible
        rho_denom = (primal_theta + (1 - primal_theta) * omega) * solve_info.etaDual + xzMat
        if iszero(rho_denom)
            solve_info.rho = typemax(T)
        else
            solve_info.rho = primal_theta * x0zMat / rho_denom
        end
    elseif primal_feasible && !dual_feasible
        # primal feasible, dual infeasible
        rho_denom = (dual_theta + (1 - dual_theta) * omega) * solve_info.etaPrimal + xzMat
        if iszero(rho_denom)
            solve_info.rho = typemax(T)
        else
            solve_info.rho = dual_theta * xMatz0 / rho_denom
        end
    elseif !primal_feasible && !dual_feasible
        # both infeasible
        rho_denom = (
            primal_theta * dual_theta
            + omega * (primal_theta * (1 - dual_theta) + (1 - primal_theta) * dual_theta)
        ) * (nDim * mu.initial) + xzMat
        if iszero(rho_denom)
            solve_info.rho = typemax(T)
        else
            solve_info.rho = (dual_theta * xMatz0 + primal_theta * x0zMat) / rho_denom
        end
    else
        # both feasible — rho stays 0
        solve_info.rho = zero(T)
    end

    # Populate adapter fields so callers that hold a SolveInfo
    # can render an iteration row without re-threading values. These have
    # no algorithmic weight; they mirror inputs already consumed above.
    solve_info.mu_current   = mu.current
    solve_info.theta_primal = primal_theta
    solve_info.theta_dual   = dual_theta
    solve_info.primal_res   = current_res.normPrimalVec
    solve_info.dual_res     = current_res.normDualMat
    gap = solve_info.objValPrimal - solve_info.objValDual
    abs_gap = abs(gap)
    mean_obj = (abs(solve_info.objValPrimal) + abs(solve_info.objValDual)) / 2
    denom = max(one(T), mean_obj)
    rel_gap = abs_gap / denom
    solve_info.gap     = abs_gap
    solve_info.rel_gap = rel_gap
    # digits = -log10(rel_gap); guard against rel_gap == 0.
    solve_info.digits  = rel_gap > 0 ? -log10(rel_gap) : typemax(T)
    # alpha_primal/dual and beta_value are updated by the caller after
    # step-length / centering computation (kept zero here).

    return solve_info
end
