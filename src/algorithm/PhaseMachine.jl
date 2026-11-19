"""
    PhaseValue

The 11-state phase machine for the IPM convergence test. Mirrors C++
`SolveInfo::phaseType` enum (referenced via `Phase::value` at
`sdpa_parts.h:215`).

States:
- `noINFO`:        no information yet (initial state on each `update_check!`)
- `pFEAS`:         primal feasible only (primal residual within epsilonDash)
- `dFEAS`:         dual feasible only (dual residual within epsilonDash)
- `pdFEAS`:        both primal and dual feasible (but not yet optimal)
- `pdINF`:         both primal and dual infeasible (rho > 1)
- `pFEAS_dINF`:    primal feasible, dual became infeasible
- `pINF_dFEAS`:    primal became infeasible, dual feasible
- `pdOPT`:         both feasible AND optimal — the success state
- `pUNBD`:         primal unbounded (objValPrimal below lowerBound)
- `dUNBD`:         dual unbounded (objValDual above upperBound)
- `not_called`:    solve never invoked (initial state at startup)

The `update_check!` function below implements the state-transition
logic from C++ `Phase::updateCheck` at `sdpa_parts.cpp:932-1011`.
"""
@enum PhaseValue begin
    noINFO
    pFEAS
    dFEAS
    pdFEAS
    pdINF
    pFEAS_dINF
    pINF_dFEAS
    pdOPT
    pUNBD
    dUNBD
    not_called
end

"""
    update_check!(phase_ref, norm_primal_vec, norm_dual_mat, obj_val_primal,
                  obj_val_dual, rho, params) -> Bool

Phase machine transition + termination test. Returns `false` when the
solve should stop (terminal state reached), `true` to continue iterating.

Mirrors C++ `Phase::updateCheck` at `sdpa_parts.cpp:932-1011` exactly.

Algorithm (line numbers in [brackets] are sdpa_parts.cpp):

  accuracy = params.epsilonDash                          [935]
  phase = noINFO                                         [936]

  if primal residual ≤ accuracy:                         [938-943]
    if dual residual ≤ accuracy: phase = pdFEAS
    else:                     phase = pFEAS
  elseif dual residual ≤ accuracy:                       [944-946]
    phase = dFEAS

  if phase == pdFEAS:                                    [947-962]
    mean = (|objP| + |objD|) / 2
    PDgap = |objP - objD|
    dominator = max(1, mean)
    if PDgap / dominator ≤ epsilonStar:                  [962]
      phase = pdOPT                                      [OPTIMAL — STOP]
      return false

  if phase == noINFO && rho > 1 + 1e-6:                  [964-967]
    phase = pdINF                                        [BOTH INFEASIBLE — STOP]
    return false

  if phase == pFEAS:                                     [969-977]
    if objP ≤ lowerBound:
      phase = pUNBD                                      [PRIMAL UNBOUNDED — STOP]
      return false
    if rho > 1 + 1e-6:
      phase = pFEAS_dINF                                 [STOP]
      return false

  if phase == dFEAS:                                     [979-987]
    if objD ≥ upperBound:
      phase = dUNBD                                      [DUAL UNBOUNDED — STOP]
      return false
    if rho > 1 + 1e-6:
      phase = pINF_dFEAS                                 [STOP]
      return false

  return true                                            [CONTINUE]
"""
function update_check!(
    phase_ref::Ref{PhaseValue},
    norm_primal_vec::T,
    norm_dual_mat::T,
    obj_val_primal::T,
    obj_val_dual::T,
    rho::T,
    params::Parameters{T},
) where {T<:Real}
    const_NONZERO = T(1e-6)
    accuracy = params.epsilonDash
    phase = noINFO

    if norm_primal_vec <= accuracy
        if norm_dual_mat <= accuracy
            phase = pdFEAS
        else
            phase = pFEAS
        end
    elseif norm_dual_mat <= accuracy
        phase = dFEAS
    end

    if phase == pdFEAS
        mean = (abs(obj_val_primal) + abs(obj_val_dual)) / 2
        PDgap = abs(obj_val_primal - obj_val_dual)
        dominator = mean < 1 ? one(T) : mean
        if PDgap / dominator <= params.epsilonStar
            phase_ref[] = pdOPT
            return false
        end
    end

    if phase == noINFO && rho > 1 + const_NONZERO
        phase_ref[] = pdINF
        return false
    end

    if phase == pFEAS
        if obj_val_primal <= params.lowerBound
            phase_ref[] = pUNBD
            return false
        end
        if rho > 1 + const_NONZERO
            phase_ref[] = pFEAS_dINF
            return false
        end
    end

    if phase == dFEAS
        if obj_val_dual >= params.upperBound
            phase_ref[] = dUNBD
            return false
        end
        if rho > 1 + const_NONZERO
            phase_ref[] = pINF_dFEAS
            return false
        end
    end

    phase_ref[] = phase
    return true
end
