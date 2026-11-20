"""
MOI (MathOptInterface) wrapper for SDPA.jl.

Provides `Optimizer{T}` so users can solve SDPs via JuMP or Convex.jl
with arbitrary precision (`BigFloat`).
"""

const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

# ---------------------------------------------------------------------------
# Optimizer struct
# ---------------------------------------------------------------------------

"""
    Optimizer{T<:Real}

MOI optimizer wrapping the SDPA primal-dual interior-point solver.

## Type parameter
- `T`: the number type used for computation. Use `BigFloat` for
  arbitrary-precision SDP. Defaults to `BigFloat`.

## Constructor
```julia
Optimizer{BigFloat}(; precision=200, maxIteration=200, ...)
```

Keyword arguments are forwarded to [`SDPA.Parameters`](@ref).
"""
mutable struct Optimizer{T<:Real} <: MOI.AbstractOptimizer
    """Solver parameters (precision, iteration limits, etc.)."""
    params::SDPA.Parameters{T}
    """The last solved problem (nothing before first optimize!)."""
    problem::Union{Nothing,SDPA.SDPAProblem{T}}
    """The solution log from the last solve."""
    solution_log::Union{Nothing,SDPA.SolutionLog{T}}
    """Wall-clock solve time in seconds."""
    solve_time::Float64
    """Silence verbose output when true."""
    silent::Bool
    """Wall-clock time limit in seconds; `nothing` means no limit. When it
    fires, TerminationStatus is TIME_LIMIT regardless of the phase."""
    time_limit_sec::Union{Nothing,Float64}
    """Variable index mapping from MOI.copy_to (for result accessors)."""
    _index_map::Union{Nothing,Dict{MOI.VariableIndex,Tuple{Symbol,Int,Int,Int}}}
    """All constraint indices (for ConstraintPrimal/ConstraintDual)."""
    _constraint_cis::Vector{MOI.ConstraintIndex}
    """Equality constraint CI → yVec position map."""
    _eq_ci_map::Dict{MOI.ConstraintIndex,Int}
    """Equality CI → its SAF terms as (variable, coefficient) pairs.
    The equality ConstraintPrimal reports the constraint *function* value
    at the solution point (`Σ c·x + const`, the activity), never the RHS
    `b`: returning `b` would claim exact feasibility even for
    infeasible/limit terminations. At a well-solved optimum the activity
    ≈ the user-set rhs. Terms are stored per-CI in `copy_to` order."""
    _eq_ci_terms::Dict{MOI.ConstraintIndex,Vector{Tuple{MOI.VariableIndex,T}}}
    """Equality CI → its SAF function constant: the additive term of the
    activity. Deliberately NOT folded into the solver's A-row/b (ingest
    stores the effective rhs `set.value - const` separately); kept raw so
    ConstraintPrimal can report the true function value."""
    _eq_ci_const::Dict{MOI.ConstraintIndex,T}
    """PSD cone CI → SDP block index map."""
    _psd_ci_map::Dict{MOI.ConstraintIndex,Int}
    """LP-cone CI → its variables in user order; ConstraintPrimal returns
    THIS constraint's slice, not the whole LP_block vector."""
    _lp_ci_vars::Dict{MOI.ConstraintIndex,Vector{MOI.VariableIndex}}
    """LP-cone CI → its variables' 1-based LP_block positions (derived
    once in `copy_to`; stable by construction: the ingest collision guard
    rejects conflicting claims and linking only fills unmapped variables).
    ConstraintDual returns THIS constraint's slice at the same positions
    VariablePrimal reads via var_map."""
    _lp_ci_positions::Dict{MOI.ConstraintIndex,Vector{Int}}
    """Reals-cone CI → its free variables; ConstraintPrimal returns the
    reconstructed free vector x⁺−x⁻, ConstraintDual returns zeros."""
    _reals_ci_vars::Dict{MOI.ConstraintIndex,Vector{MOI.VariableIndex}}
    """Objective sense of the last ingested model (for MAX_SENSE negation)."""
    objective_sense::Union{Nothing,MOI.OptimizationSense}
    """Objective-function constant offset. SDPAProblem's C holds only
    linear coefficients, so the SAF constant is stored here and added
    back by ObjectiveValue/DualObjectiveValue."""
    objective_constant::T
end

"""
    Optimizer(; kwargs...)

Construct a default `Optimizer{BigFloat}` with the given parameters.
"""
Optimizer(; kwargs...) = Optimizer{BigFloat}(; kwargs...)

"""
    Optimizer{T}(; precision, maxIteration, ...)

Construct an `Optimizer` with number type `T`. Keyword arguments are
forwarded to `SDPA.Parameters{T}`.
"""
function Optimizer{T}(;
        precision::Union{Nothing,Integer}=nothing,
        silent::Bool=false,
        kwargs...) where {T<:Real}
    # Build parameters from kwargs, applying defaults.
    if precision !== nothing
        p = SDPA.Parameters{T}(; kwargs..., precision=precision)
    else
        p = SDPA.Parameters{T}(; kwargs...)
    end
    return Optimizer{T}(p, nothing, nothing, 0.0, silent, nothing, nothing,
        MOI.ConstraintIndex[],
        Dict{MOI.ConstraintIndex,Int}(),
        Dict{MOI.ConstraintIndex,Vector{Tuple{MOI.VariableIndex,T}}}(),
        Dict{MOI.ConstraintIndex,T}(),
        Dict{MOI.ConstraintIndex,Int}(),
        Dict{MOI.ConstraintIndex,Vector{MOI.VariableIndex}}(),
        Dict{MOI.ConstraintIndex,Vector{Int}}(),
        Dict{MOI.ConstraintIndex,Vector{MOI.VariableIndex}}(),
        nothing, zero(T))
end

# ---------------------------------------------------------------------------
# MOI attribute queries
# ---------------------------------------------------------------------------

MOI.get(::Optimizer, ::MOI.SolverName) = "SDPA"
MOI.get(opt::Optimizer, ::MOI.SolveTimeSec) = opt.solve_time
MOI.get(opt::Optimizer, ::MOI.RawSolver) = opt

# The optimizer-attribute inventory MOI machinery copies through
# (get on src, set on dest — e.g. MOIU.reset_optimizer). Only attributes
# with BOTH get and set methods may be listed: SolverName/SolverVersion
# are get-only (advertising them made the copy loop die with
# SetAttributeNotAllowed mid-copy), and SolveTimeSec and RawSolver are
# AbstractModelAttributes in MOI and do not belong here.
function MOI.get(::Optimizer, ::MOI.ListOfOptimizerAttributesSet)
    return MOI.AbstractOptimizerAttribute[
        MOI.Silent(),
        MOI.TimeLimitSec(),
    ]
end

# Attribute-surface gaps. For settable attributes the
# supports/get/set triplets are kept consistent (supports == settable) so
# the raw optimizer also honors the MOI contract when driven without a
# caching layer. Get-only attributes (SolverName, SolverVersion) define
# NO supports method — MOI's `supports` advertises settability, and
# A stale `supports(SolverVersion) = true` would claim it.
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
function MOI.get(opt::Optimizer, ::MOI.ObjectiveSense)
    return opt.objective_sense === nothing ? MOI.FEASIBILITY_SENSE : opt.objective_sense
end
function MOI.set(opt::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    opt.objective_sense = sense
    return
end

function MOI.get(::Optimizer, ::MOI.SolverVersion)
    return string(Base.pkgversion(SDPA))
end

# NOTE: MOI.Name is deliberately NOT implemented here: the caching layer
# (JuMP/MOI.Utilities.CachingOptimizer) owns the Name attribute and would
# otherwise disagree with a solver-side copy (get routes to the optimizer
# while set is cache-only in MOIU).

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.get(opt::Optimizer, ::MOI.Silent) = opt.silent
function MOI.set(opt::Optimizer, ::MOI.Silent, v::Bool)
    opt.silent = v
    return
end

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.get(opt::Optimizer, ::MOI.TimeLimitSec) = opt.time_limit_sec
function MOI.set(opt::Optimizer, ::MOI.TimeLimitSec, v::Union{Nothing,Real})
    opt.time_limit_sec = v === nothing ? nothing : Float64(v)
    return
end

# Model lifecycle (required by JuMP's MOI.instantiate)
MOI.is_empty(opt::Optimizer) = opt.problem === nothing
function MOI.empty!(opt::Optimizer{T}) where {T<:Real}
    opt.problem = nothing
    opt.solution_log = nothing
    opt._index_map = nothing
    opt._constraint_cis = MOI.ConstraintIndex[]
    opt._eq_ci_map = Dict{MOI.ConstraintIndex,Int}()
    opt._eq_ci_terms = Dict{MOI.ConstraintIndex,Vector{Tuple{MOI.VariableIndex,T}}}()
    opt._eq_ci_const = Dict{MOI.ConstraintIndex,T}()
    opt._psd_ci_map = Dict{MOI.ConstraintIndex,Int}()
    opt._lp_ci_vars = Dict{MOI.ConstraintIndex,Vector{MOI.VariableIndex}}()
    opt._lp_ci_positions = Dict{MOI.ConstraintIndex,Vector{Int}}()
    opt._reals_ci_vars = Dict{MOI.ConstraintIndex,Vector{MOI.VariableIndex}}()
    opt.objective_sense = nothing
    opt.objective_constant = zero(T)
    opt.solve_time = 0.0
    return
end

# This optimizer is not incremental: the model is ingested wholesale in
# copy_to. Explicit declaration (the MOI default is false for ModelLike,
# but being explicit is part of the contract and avoids relying on it).
MOI.supports_incremental_interface(::Optimizer) = false

# MOI.copy_to: transfers model from src to our optimizer's internal
# SDPAProblem representation. Returns an IndexMap that JuMP uses for
# result queries.
function MOI.copy_to(
    dest::Optimizer{T}, src::MOI.ModelLike
) where {T<:Real}
    MOI.empty!(dest)
    problem, var_map, all_cis, eq_ci_list, psd_ci_list, lp_ci_list,
        lp_ci_vars, reals_ci_list, reals_ci_vars, eq_ci_activity, obj_const =
        _build_sdpa_problem_from_moi(src, T)
    dest.problem = problem
    dest.objective_constant = obj_const
    dest._index_map = var_map
    dest._constraint_cis = all_cis
    dest._eq_ci_map = Dict(ci => i for (i, ci) in enumerate(eq_ci_list))
    # Per-CI equality activity data ((terms, SAF constant) in
    # eq_ci_list order) for the ConstraintPrimal accessor.
    dest._eq_ci_terms = Dict(
        ci => terms for (ci, (terms, _)) in zip(eq_ci_list, eq_ci_activity))
    dest._eq_ci_const = Dict(
        ci => c0 for (ci, (_, c0)) in zip(eq_ci_list, eq_ci_activity))
    dest._psd_ci_map = Dict(ci => i for (i, ci) in enumerate(psd_ci_list))
    dest._lp_ci_vars = Dict(
        ci => vars for (ci, vars) in zip(lp_ci_list, lp_ci_vars))
    # Derive the per-CI LP_block positions ONCE here instead of
    # per-query through _index_map: every LP-cone variable's var_map entry
    # is (:LP, 1, pos, 0) with field 3 = its 1-based LP_block position.
    # That entry is stable after ingest — the collision guard rejects
    # any conflicting second claim, and the linking pass only fills in
    # variables that have NO mapping yet (LP-cone vars always have one).
    dest._lp_ci_positions = Dict(
        ci => Int[var_map[vi][3] for vi in vars]
        for (ci, vars) in zip(lp_ci_list, lp_ci_vars))
    dest._reals_ci_vars = Dict(
        ci => vars for (ci, vars) in zip(reals_ci_list, reals_ci_vars))
    dest.objective_sense = MOI.get(src, MOI.ObjectiveSense())
    # Build IndexMap with identity mappings for BOTH variables and constraints.
    # CachingOptimizer requires every constraint to be mapped, else post-solve
    # lookups (ConstraintPrimal, ConstraintDual) throw KeyError.
    idx_map = MOIU.IndexMap()
    for vi in keys(var_map)
        idx_map[vi] = vi
    end
    for ci in all_cis
        idx_map[ci] = ci
    end
    return idx_map
end

# Single-argument optimize!: solves the problem stored by copy_to.
function MOI.optimize!(opt::Optimizer{T}) where {T<:Real}
    opt.problem === nothing && error("no problem loaded; call copy_to first")
    params = opt.params
    t0 = time()
    opt.solution_log = if T === BigFloat
        setprecision(BigFloat, max(params.precision, 53)) do
            SDPA.solve!(opt.problem, params; verbose=!opt.silent,
                        reverse_primal_dual=false,
                        time_limit_sec=opt.time_limit_sec)
        end
    else
        SDPA.solve!(opt.problem, params; verbose=!opt.silent,
                    reverse_primal_dual=false,
                    time_limit_sec=opt.time_limit_sec)
    end
    opt.solve_time = time() - t0
    return
end

# ---------------------------------------------------------------------------
# Termination status mapping
# ---------------------------------------------------------------------------

# Phase -> MOI.TerminationStatus. With reverse_primal_dual=false the phase
# machine reports the USER's true problem phases directly:
#   - pUNBD (primal unbounded)   ⟺ dual infeasible
#   - dUNBD (dual unbounded)     ⟺ primal infeasible   [KEY]
#   - pINF_dFEAS / pdINF         ⟺ primal infeasible
#   - pFEAS_dINF                 ⟺ dual infeasible (primal unbounded)
#   - pFEAS/dFEAS/pdFEAS         = feasible but not converged -> ALMOST_OPTIMAL
function _moi_termination(phase::SDPA.PhaseValue)
    if phase === SDPA.pdOPT
        return MOI.OPTIMAL
    elseif phase === SDPA.pFEAS ||
           phase === SDPA.dFEAS ||
           phase === SDPA.pdFEAS
        return MOI.ALMOST_OPTIMAL
    elseif phase === SDPA.pUNBD
        return MOI.DUAL_INFEASIBLE
    elseif phase === SDPA.dUNBD ||
           phase === SDPA.pINF_dFEAS ||
           phase === SDPA.pdINF
        return MOI.INFEASIBLE
    elseif phase === SDPA.pFEAS_dINF
        return MOI.DUAL_INFEASIBLE
    elseif phase === SDPA.noINFO
        return MOI.OTHER_LIMIT
    else  # not_called
        return MOI.OPTIMIZE_NOT_CALLED
    end
end

function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    if opt.solution_log === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    end
    # A wall-clock time limit firing wins over any phase-based mapping
    # (the phase is left mid-flight and would otherwise map to OTHER_LIMIT).
    if opt.solution_log.timed_out
        return MOI.TIME_LIMIT
    end
    # A non-terminal exit FREEZES the phase mid-solve, so the phase
    # alone misreports (a frozen pdFEAS would read ALMOST_OPTIMAL, a
    # frozen noINFO OTHER_LIMIT). exit_reason disambiguates exactly those
    # cases: budget exhausted → ITERATION_LIMIT, mid-solve numerical
    # breakdown (Cholesky failure / iterate cannot move) → OTHER_LIMIT.
    # Decisive phases (pdOPT / pdINF / pUNBD / ...) end the loop through
    # update_check! returning false, which leaves exit_reason at
    # OPTIMAL_TERMINAL, so they keep riding the phase mapping below —
    # as do EMPTY_PROBLEM solves (trivial early exit, phase decisive by
    # construction).
    reason = opt.solution_log.exit_reason
    if reason === SDPA.MAX_ITERATION
        return MOI.ITERATION_LIMIT
    elseif reason === SDPA.CHOLESKY_FAILURE || reason === SDPA.CANNOT_MOVE
        return MOI.OTHER_LIMIT
    end
    return _moi_termination(opt.solution_log.terminal_phase)
end

# Phase -> MOI.PrimalStatus. The primal ray (X) certifies dual
# infeasibility for pUNBD; all other non-solution phases report NO_SOLUTION
# (not UNKNOWN_RESULT_STATUS).
function _moi_primal_status(phase::SDPA.PhaseValue)
    if phase === SDPA.pdOPT || phase === SDPA.pFEAS || phase === SDPA.pdFEAS ||
       phase === SDPA.pFEAS_dINF
        # pFEAS_dINF (DUAL_INFEASIBLE termination): the primal iterate IS
        # feasible. Report it as FEASIBLE_POINT so ResultCount and
        # ObjectiveValue stay consistent with the dual side; no
        # dual-infeasibility RAY is extracted (documented interim).
        return MOI.FEASIBLE_POINT
    elseif phase === SDPA.pUNBD
        return MOI.INFEASIBILITY_CERTIFICATE
    else
        return MOI.NO_SOLUTION
    end
end

function MOI.get(opt::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index != 1 || opt.solution_log === nothing
        return MOI.NO_SOLUTION
    end
    return _moi_primal_status(opt.solution_log.terminal_phase)
end

# Phase -> MOI.DualStatus. Independent of PrimalStatus: the dual ray (yVec /
# zMat) certifies primal infeasibility for dUNBD / pINF_dFEAS / pdINF, and
# the dual objective is a feasible point for the feasible dual phases.
function _moi_dual_status(phase::SDPA.PhaseValue)
    if phase === SDPA.pdOPT || phase === SDPA.dFEAS || phase === SDPA.pdFEAS
        return MOI.FEASIBLE_POINT
    elseif phase === SDPA.dUNBD ||
           phase === SDPA.pINF_dFEAS ||
           phase === SDPA.pdINF
        return MOI.INFEASIBILITY_CERTIFICATE
    else
        return MOI.NO_SOLUTION
    end
end

function MOI.get(opt::Optimizer, attr::MOI.DualStatus)
    if attr.result_index != 1 || opt.solution_log === nothing
        return MOI.NO_SOLUTION
    end
    return _moi_dual_status(opt.solution_log.terminal_phase)
end

# The solver provides at most one result: ResultCount is 0 only when neither
# the primal nor the dual side has any solution.
function MOI.get(opt::Optimizer, ::MOI.ResultCount)
    if opt.solution_log === nothing
        return 0
    end
    if MOI.get(opt, MOI.PrimalStatus()) === MOI.NO_SOLUTION &&
       MOI.get(opt, MOI.DualStatus()) === MOI.NO_SOLUTION
        return 0
    end
    return 1
end

function MOI.get(opt::Optimizer, ::MOI.BarrierIterations)
    opt.solution_log === nothing &&
        throw(MOI.GetAttributeNotAllowed(MOI.BarrierIterations()))
    return opt.solution_log.iterations
end

function MOI.get(opt::Optimizer, attr::MOI.ObjectiveValue)
    opt.solution_log === nothing && throw(MOI.GetAttributeNotAllowed(attr))
    MOI.check_result_index_bounds(opt, attr)
    # With reverse_primal_dual=false the solver minimizes ⟨C,X⟩ directly.
    # Ingest builds C = -coeff for MAX_SENSE, so the solver's optimum is
    # -max f and must be negated back to the user's objective value.
    # Add back the stored objective constant offset.
    const_term = opt.objective_constant
    if opt.objective_sense === MOI.MAX_SENSE
        return -opt.solution_log.final_obj_primal + const_term
    end
    return opt.solution_log.final_obj_primal + const_term
end

# ---------------------------------------------------------------------------
# Supported constraints and objective
# ---------------------------------------------------------------------------

# LP variables: vector in Nonnegatives
MOI.supports_add_constrained_variables(
    ::Optimizer, ::Type{<:MOI.Nonnegatives}) = true

# SDP variables: vector in PositiveSemidefiniteConeTriangle
MOI.supports_add_constrained_variables(
    ::Optimizer, ::Type{<:MOI.PositiveSemidefiniteConeTriangle}) = true

# Free variables (ingest splits x = x⁺ - x⁻ into Nonnegatives).
# The EXACT ::Type{MOI.Reals} signature is required: MOI itself defines
# a fallback for ::Type{MOI.Reals} on ModelLike, and the <:-form is
# ambiguous with it.
MOI.supports_add_constrained_variables(
    ::Optimizer, ::Type{MOI.Reals}) = true

# Objective: ScalarAffineFunction
MOI.supports(::Optimizer,
    ::MOI.ObjectiveFunction{<:MOI.ScalarAffineFunction}) = true

# The optimizer ingests exactly four constraint types; advertise the three
# that are safe to route directly. `VectorAffineFunction` in
# `PositiveSemidefiniteConeTriangle` is deliberately NOT advertised: the
# ingest handles it (the direct VAF path), but its solutions can lie on the
# PSD cone boundary, where the IPM stalls at dFEAS (ALMOST_OPTIMAL) instead
# of converging to pdOPT — advertising it would redirect Convex.jl / JuMP
# matrix-form constraints onto that path and regress convergence (the
# VectorFunctionizeBridge converts VAF-in-PSD into the advertised
# VectorOfVariables-in-PSD + ScalarAffineFunction-in-EqualTo forms, which
# converge reliably). This is a deliberate, documented API-surface choice.
#
# Equality constraints: ScalarAffineFunction in EqualTo
MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{<:MOI.ScalarAffineFunction{T}},
    ::Type{MOI.EqualTo{T}},
) where {T<:Real} = true

# SDP variables: VectorOfVariables in PositiveSemidefiniteConeTriangle
MOI.supports_constraint(
    ::Optimizer, ::Type{MOI.VectorOfVariables},
    ::Type{MOI.PositiveSemidefiniteConeTriangle}) = true

# LP variables: VectorOfVariables in Nonnegatives
MOI.supports_constraint(
    ::Optimizer, ::Type{MOI.VectorOfVariables},
    ::Type{MOI.Nonnegatives}) = true

# Free variables in Reals.
MOI.supports_constraint(
    ::Optimizer, ::Type{MOI.VectorOfVariables},
    ::Type{MOI.Reals}) = true

# ---------------------------------------------------------------------------
# MOI.optimize!
# ---------------------------------------------------------------------------

# The two-argument form MOI.optimize!(dest::Optimizer, src::ModelLike) is
# provided by MOI's default fallback:
#   optimize!(dest, src) = (index_map = copy_to(dest, src); optimize!(dest);
#                            return index_map, true)
# copy_to populates all of dest's state and returns an identity IndexMap, so
# the fallback is equivalent to a dedicated two-arg method and avoids a
# second ingest path duplicating state.

# ---------------------------------------------------------------------------
# VariablePrimal result accessor
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Result accessors
# ---------------------------------------------------------------------------

# BigFloat values are mutable MPFR objects; SCALAR result reads must not
# alias solver state (mutation isolation — the vector getters return value
# copies via `_copy_independent`; Float64 is immutable and returned as-is).
_scalar_value(v::T) where {T<:Real} = v
_scalar_value(v::BigFloat) = deepcopy(v)

"""
    MOI.get(opt::Optimizer, ::MOI.VariablePrimal, vi::VariableIndex)

Return the primal value of variable `vi` from the last solution, looked
up in `solution_log.xMat` via the block mapping stored during `copy_to`.
`:SDP` reads `SDP_block`, `:FREE` returns x⁺−x⁻, `:LP` reads `LP_block`.
BigFloat scalars are deep-copied (no aliasing of solver state).
"""
function MOI.get(opt::Optimizer{T}, attr::MOI.VariablePrimal,
                 vi::MOI.VariableIndex) where {T<:Real}
    opt.solution_log === nothing && throw(MOI.GetAttributeNotAllowed(attr))
    MOI.check_result_index_bounds(opt, attr)
    # MOI conformance: an index absent from the stored mapping throws
    # MOI.InvalidIndex (was a bare error()) — the contract error MOI
    # machinery recognizes.
    opt._index_map === nothing && throw(MOI.InvalidIndex(vi))
    var_map = opt._index_map  # already typed; no cast needed
    haskey(var_map, vi) || throw(MOI.InvalidIndex(vi))
    block_type, blk, row, col = var_map[vi]
    # With reverse_primal_dual=false (the MOI convention), the raw
    # SolutionLog has xMat = primal X and zMat = dual slack Z (no swap).
    # Read the primal values from xMat.
    primalMat = opt.solution_log.xMat
    if block_type === :SDP
        return _scalar_value(primalMat.SDP_block[blk].de_ele[row, col])
    elseif block_type === :FREE
        # x = x⁺ − x⁻. The subtraction allocates a fresh value
        # (BigFloat included), so the caller cannot alias solver state.
        pos_plus, pos_minus = blk, row
        return primalMat.LP_block[pos_plus] - primalMat.LP_block[pos_minus]
    else  # :LP
        return _scalar_value(primalMat.LP_block[row])
    end
end

# ---------------------------------------------------------------------------
# ConstraintPrimal / ConstraintDual result accessors
# ---------------------------------------------------------------------------

"""
Pack a dense symmetric matrix into MOI's PositiveSemidefiniteConeTriangle
vector format (column-major upper triangle). Each element is VALUE-copied
(`_scalar_value`: BigFloat elements are deep-copied), so the returned
vector never aliases solver state — mutating it must not affect
subsequent result queries.
"""
function _dense_to_triangle_vector(block::Matrix{T}) where {T<:Real}
    n = size(block, 1)
    v = Vector{T}(undef, div(n * (n + 1), 2))
    idx = 0
    for col in 1:n, row in 1:col
        idx += 1
        v[idx] = _scalar_value(block[row, col])
    end
    return v
end

function MOI.get(opt::Optimizer{T}, attr::MOI.ConstraintPrimal,
                 ci::MOI.ConstraintIndex) where {T<:Real}
    opt.solution_log === nothing && throw(MOI.GetAttributeNotAllowed(attr))
    MOI.check_result_index_bounds(opt, attr)
    if haskey(opt._eq_ci_map, ci)
        # Report the constraint FUNCTION value at the solution point — the
        # activity Σ coeff·x + const — not the RHS b: returning b would claim
        # exact feasibility even for infeasible/limit terminations. Valid
        # whenever a result point exists (ResultCount ≥ 1), including
        # truncated solves. The solver's b-row is rhs − const; the activity
        # of a well-solved model ≈ the user's rhs. Accumulation starts at
        # zero(T) + const and every += yields a fresh value for BigFloat, so
        # the returned scalar never aliases stored state; the empty-terms
        # case degenerates to the constant.
        terms = opt._eq_ci_terms[ci]
        activity = zero(T) + opt._eq_ci_const[ci]
        for (vi, coeff) in terms
            activity += coeff * MOI.get(opt, MOI.VariablePrimal(), vi)
        end
        return activity
    elseif haskey(opt._reals_ci_vars, ci)
        # The Reals cone's primal is the reconstructed free vector.
        vars = opt._reals_ci_vars[ci]
        return T[MOI.get(opt, MOI.VariablePrimal(), vi) for vi in vars]
    elseif haskey(opt._psd_ci_map, ci)
        blk = opt._psd_ci_map[ci]
        # xMat = X (primal) with reverse_primal_dual=false
        return _dense_to_triangle_vector(opt.solution_log.xMat.SDP_block[blk].de_ele)
    elseif haskey(opt._lp_ci_vars, ci)
        # LP cone — return THIS CI's variables' values in user order, not
        # the entire LP_block vector (wrong length/content whenever the
        # model has >1 LP cone or split free variables). Mirrors the
        # Reals branch: VariablePrimal value-copies each scalar
        # (_scalar_value), so no caller aliasing of solver state.
        # PSD(1)-routed CIs land here too and get exactly their 1 variable.
        vars = opt._lp_ci_vars[ci]
        return T[MOI.get(opt, MOI.VariablePrimal(), vi) for vi in vars]
    elseif ci in opt._constraint_cis
        # Copied CI that was filtered from the solver state: ingest drops
        # redundant equality constraints (e.g. variable-linking equalities
        # that become 0=0 after linking), so they have no A-row — their
        # primal is the (satisfied) residual, i.e. zero.
        return zero(T)
    end
    # CI never part of the copied model: invalid index per the MOI contract
    # (by construction every copied CI is either mapped or filtered above,
    # so a legal query never reaches here — this is defensive).
    throw(MOI.InvalidIndex(ci))
end

function MOI.get(opt::Optimizer{T}, attr::MOI.ConstraintDual,
                 ci::MOI.ConstraintIndex) where {T<:Real}
    opt.solution_log === nothing && throw(MOI.GetAttributeNotAllowed(attr))
    MOI.check_result_index_bounds(opt, attr)
    # MOI keeps constraint duals in the same dual cone for either sense.
    # For max ⟨C,X⟩, ingestion already changes the stationarity equation
    # to -C = A'y + Z. These y and Z are the MOI duals; only the
    # objective values change sign when converting back from min to max.
    if haskey(opt._eq_ci_map, ci)
        return _scalar_value(opt.solution_log.yVec.ele[opt._eq_ci_map[ci]])
    elseif haskey(opt._reals_ci_vars, ci)
        # The dual of the free-variable cone constraint is zero.
        return _independent_zeros(T, length(opt._reals_ci_vars[ci]))
    elseif haskey(opt._psd_ci_map, ci)
        blk = opt._psd_ci_map[ci]
        # zMat = Z (dual slack) with reverse_primal_dual=false
        return _dense_to_triangle_vector(opt.solution_log.zMat.SDP_block[blk].de_ele)
    elseif haskey(opt._lp_ci_positions, ci)
        # LP cone — THIS CI's slice of zMat.LP_block, at the same LP
        # positions its variables occupy (see copy_to). Return a value copy.
        return _copy_independent(
            opt.solution_log.zMat.LP_block[opt._lp_ci_positions[ci]])
    elseif ci in opt._constraint_cis
        # Copied CI filtered from the solver state (redundant equality after
        # ingest's linking pass) — its dual is zero.
        return zero(T)
    end
    # CI never part of the copied model: invalid index per the MOI contract
    # (see ConstraintPrimal).
    throw(MOI.InvalidIndex(ci))
end

# ---------------------------------------------------------------------------
# RawOptimizerAttribute
# ---------------------------------------------------------------------------

function MOI.set(
    opt::Optimizer{T}, ra::MOI.RawOptimizerAttribute, value
) where {T<:Real}
    field_name = Symbol(ra.name)
    hasfield(SDPA.Parameters{T}, field_name) ||
        throw(MOI.UnsupportedAttribute(ra))
    FT = fieldtype(SDPA.Parameters{T}, field_name)
    converted = convert(FT, value)
    # Re-run all validators by reconstructing Parameters{T}.
    opt.params = SDPA.Parameters{T}(;
        maxIteration=opt.params.maxIteration,
        precision=opt.params.precision,
        epsilonStar=opt.params.epsilonStar,
        lambdaStar=opt.params.lambdaStar,
        omegaStar=opt.params.omegaStar,
        lowerBound=opt.params.lowerBound,
        upperBound=opt.params.upperBound,
        betaStar=opt.params.betaStar,
        betaBar=opt.params.betaBar,
        gammaStar=opt.params.gammaStar,
        epsilonDash=opt.params.epsilonDash,
        field_name => converted,
    )
    return
end

# supports must reflect whether the attribute is settable; a
# random name is NOT supported even though set() would throw for it.
function MOI.supports(::Optimizer{T}, ra::MOI.RawOptimizerAttribute) where {T<:Real}
    return hasfield(SDPA.Parameters{T}, Symbol(ra.name))
end

# ---------------------------------------------------------------------------
# Custom result accessors
# ---------------------------------------------------------------------------

struct PrimalSolutionMatrix <: MOI.AbstractModelAttribute end
struct DualSolutionVector  <: MOI.AbstractVariableAttribute end
struct DualSlackMatrix     <: MOI.AbstractModelAttribute end

function _wrap_dls(dls::SDPA.SDPADenseLinearSpace{T}) where {T<:Real}
    out = Matrix{T}[]
    for k in 1:dls.SDP_nBlock
        if isassigned(dls.SDP_block, k)
            push!(out, _copy_independent(dls.SDP_block[k].de_ele))
        end
    end
    if !isempty(dls.LP_block)
        push!(out, reshape(_copy_independent(dls.LP_block), 1, length(dls.LP_block)))
    end
    return out
end

# With reverse_primal_dual=false: xMat holds primal X, zMat holds dual slack Z.
function MOI.get(opt::Optimizer{T}, attr::PrimalSolutionMatrix) where {T<:Real}
    opt.solution_log === nothing && throw(MOI.GetAttributeNotAllowed(attr))
    return _wrap_dls(opt.solution_log.xMat)
end

function MOI.get(opt::Optimizer{T}, attr::DualSolutionVector) where {T<:Real}
    opt.solution_log === nothing && throw(MOI.GetAttributeNotAllowed(attr))
    return _copy_independent(opt.solution_log.yVec.ele)
end

function MOI.get(opt::Optimizer{T}, attr::DualSlackMatrix) where {T<:Real}
    opt.solution_log === nothing && throw(MOI.GetAttributeNotAllowed(attr))
    return _wrap_dls(opt.solution_log.zMat)
end

# ---------------------------------------------------------------------------
# RawStatusString + DualObjectiveValue
# ---------------------------------------------------------------------------

function _raw_status_string(ph::SDPA.PhaseValue)
    ph === SDPA.pdOPT      && return "PD_OPTIMAL"
    ph === SDPA.pFEAS      && return "P_FEASIBLE"
    ph === SDPA.dFEAS      && return "D_FEASIBLE"
    ph === SDPA.pdFEAS     && return "PD_FEASIBLE"
    ph === SDPA.pINF_dFEAS && return "P_INFEASIBLE_D_FEASIBLE"
    ph === SDPA.pFEAS_dINF && return "P_FEASIBLE_D_INFEASIBLE"
    ph === SDPA.pdINF      && return "PD_INFEASIBLE"
    ph === SDPA.pUNBD      && return "PRIMAL_UNBOUNDED"
    ph === SDPA.dUNBD      && return "DUAL_UNBOUNDED"
    ph === SDPA.noINFO     && return "NO_INFO"
    ph === SDPA.not_called && return "NOT_CALLED"
    return "OTHER"
end

function MOI.get(opt::Optimizer, ::MOI.RawStatusString)
    opt.solution_log === nothing && return "OPTIMIZE_NOT_CALLED"
    base = _raw_status_string(opt.solution_log.terminal_phase)
    # When a non-terminal exit froze the phase mid-solve, name WHY the
    # loop ended (e.g. "PD_FEASIBLE (MAX_ITERATION)"). Healthy
    # terminations (OPTIMAL_TERMINAL / EMPTY_PROBLEM) keep the plain
    # phase string, so the converged case still reads exactly
    # "PD_OPTIMAL".
    reason = opt.solution_log.exit_reason
    if reason === SDPA.OPTIMAL_TERMINAL || reason === SDPA.EMPTY_PROBLEM
        return base
    end
    return string(base, " (", reason, ")")
end

function MOI.get(opt::Optimizer{T}, attr::MOI.DualObjectiveValue) where {T<:Real}
    opt.solution_log === nothing && throw(MOI.GetAttributeNotAllowed(attr))
    MOI.check_result_index_bounds(opt, attr)
    # Same MAX_SENSE negation as ObjectiveValue: ingest builds C = -coeff,
    # so final_obj_dual = ⟨b,y⟩ is the dual optimum of the negated problem.
    # Add back the stored objective constant offset (see DualObjectiveValue).
    const_term = opt.objective_constant
    if opt.objective_sense === MOI.MAX_SENSE
        return -opt.solution_log.final_obj_dual + const_term
    end
    return opt.solution_log.final_obj_dual + const_term
end
