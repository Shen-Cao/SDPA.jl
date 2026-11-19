"""
    ExitReason

Why the IPM loop terminated: `terminal_phase` alone cannot
distinguish a healthy terminal phase from one frozen mid-solve by a
non-terminal exit (maxIteration / Cholesky failure / cannot-move /
time limit).

Assignment rule: the loop starts at `OPTIMAL_TERMINAL` and every
NON-terminal break overwrites it — `TIME_LIMIT` exactly where `timed_out`
is set, `CHOLESKY_FAILURE` at the two `mehrotra!` failure breaks,
`CANNOT_MOVE` when `update!` reports the iterate cannot move. The pdFEAS
early stop (`beta.value > 5 || objP < objD`, C++ sdpa_main.cpp:337-340
parity) is a healthy termination and keeps `OPTIMAL_TERMINAL`. The
while-condition's `maxIteration` leg is not a break site, so it is
disambiguated after the loop: `OPTIMAL_TERMINAL` is upgraded to
`MAX_ITERATION` iff `continue_iter` was still true AND
`iter >= params.maxIteration` (every mid-body break fires before the
`iter` increment, so a break can never satisfy that inequality).
`EMPTY_PROBLEM` is set by the trivial-problem early exit, before the
loop ever runs.
"""
@enum ExitReason begin
    OPTIMAL_TERMINAL
    MAX_ITERATION
    CHOLESKY_FAILURE
    CANNOT_MOVE
    TIME_LIMIT
    EMPTY_PROBLEM
end

"""
    SolutionLog{T<:Real}

Capture of the IPM solve trajectory for testing, output writing, and
MOI result accessors. Returned by `solve!`.

Fields
------
- `iterations::Int`:                final iteration count
- `terminal_phase::PhaseValue`:    phase at termination
- `trajectory::Vector{IterationRow{T}}`: per-iteration scalars
  (iter, mu, thetaP, thetaD, objP, objD, alphaP, alphaD, beta)
- `final_obj_primal::T`:           final `⟨C, X⟩`
- `final_obj_dual::T`:             final `⟨b, y⟩`
- `final_mu::T`:                   final complementarity μ
- `final_primal_res::T`:           final `‖b - ⟨A, X⟩‖_∞`
- `final_dual_res::T`:             final `‖C - Σ y·A - Z‖_∞`
- `xMat::SDPADenseLinearSpace{T}`: final primal iterate X
- `yVec::SDPAVector{T}`:           final dual iterate y
- `zMat::SDPADenseLinearSpace{T}`: final dual slack Z
- `timings::NamedTuple`:           wall-clock split
  `(mehrotra=..., eigen=..., residual=..., total=...)`
  populated via `time_ns()` around each phase in `_solve_inner!`.
- `timed_out::Bool`:               `true` when the `time_limit_sec` wall-clock
  limit fired, cutting the iteration loop short. Kept for backward
  compatibility; exactly equivalent to `exit_reason == TIME_LIMIT`.
- `exit_reason::ExitReason`:       why the IPM loop terminated; see the
  `ExitReason` docstring for the assignment rule. `terminal_phase` alone
  cannot distinguish a healthy terminal phase from one frozen mid-solve
  by a non-terminal exit — this field can.
"""
struct SolutionLog{T<:Real}
    iterations::Int
    terminal_phase::PhaseValue
    trajectory::Vector{IterationRow{T}}
    final_obj_primal::T
    final_obj_dual::T
    final_mu::T
    final_primal_res::T
    final_dual_res::T
    xMat::SDPADenseLinearSpace{T}
    yVec::SDPAVector{T}
    zMat::SDPADenseLinearSpace{T}
    timings::NamedTuple
    timed_out::Bool
    exit_reason::ExitReason
end

# ---------------------------------------------------------------------------
# REVERSE_PRIMAL_DUAL convention support
# ---------------------------------------------------------------------------
# C++ SDPA-GMP is compiled with `#define REVERSE_PRIMAL_DUAL 1`
# (sdpa_tool.h:84). This flag swaps primal/dual labels in the display
# AND negates both objective values. Empirically, negating C before
# solving causes the solver to follow a central path that converges
# to the published optimum, matching C++ output exactly.
#
# The mechanism: with `REVERSE_PRIMAL_DUAL=1`, the C++ solver effectively
# solves `min ⟨-C, X⟩` (the "reversed" problem), then displays the
# results with objP/objD swapped and negated so the user sees the
# standard-form values. Our Julia port replicates this by:
#   1. Solving a shallow problem copy whose C is a negated value copy
#      (the caller's C is never touched)
#   2. Transforming the SolutionLog for REVERSE display
#
# Parity with the C++ reference output: example1 → -41.9 (39 iters),
# arch0 → +0.5665 (53 iters).

"""Negate all entries of the objective matrix C in-place."""
function _negate_C!(problem::SDPAProblem{T}) where {T<:Real}
    for sdp_k in 1:problem.C.SDP_sp_nBlock
        blk = problem.C.SDP_sp_block[sdp_k]
        if blk.type === :sparse
            for nz in 1:blk.nonZeroEffect
                blk.sp_ele[nz] = -blk.sp_ele[nz]
            end
        else
            for k in eachindex(blk.de_ele)
                blk.de_ele[k] = -blk.de_ele[k]
            end
        end
    end
    # LP entries: iterate the compact nonzero list of C. Negating a
    # nonzero keeps it nonzero, so the compact form is preserved.
    for lp_pos in 1:problem.C.LP_nnz
        problem.C.LP_sp_val[lp_pos] = -problem.C.LP_sp_val[lp_pos]
    end
    return problem
end

# Shallow problem copy with a negated VALUE copy of C, so the
# REVERSE_PRIMAL_DUAL convention never mutates the caller's problem.
# b and the A matrices are shared (read-only during a solve); the index
# tables are rebuilt on the copy by the SDPAProblem constructor +
# initialize_index! (a one-time O(nonzeros) scan per solve).
function _problem_with_negated_C(problem::SDPAProblem{T}) where {T<:Real}
    C_copy = SDPASparseLinearSpace{T}(
        problem.C.SDP_sp_nBlock, problem.C.LP_sp_nBlock)
    copyto!(C_copy, problem.C)
    problem_copy = SDPAProblem{T}(
        problem.m, problem.nBlock, problem.blockStruct,
        problem.b, C_copy, problem.A,
    )
    initialize_index!(problem_copy)
    _negate_C!(problem_copy)
    return problem_copy
end

# Populate the derived aggregate index tables if they have never been
# initialized (all counts zero AND all per-block lists empty — the state a
# manually constructed SDPAProblem is born in). Readers and MOI ingest
# always call initialize_index! themselves, so an initialized problem is
# never rebuilt here; a genuinely zero-constraint problem rebuilds to the
# same empty tables (no-op).
function _ensure_index_tables!(problem::SDPAProblem)
    tables_untouched = all(iszero, problem.SDP_nConstraint) &&
                       all(iszero, problem.LP_nConstraint) &&
                       all(isempty, problem.SDP_constraint) &&
                       all(isempty, problem.LP_constraint)
    if tables_untouched
        initialize_index!(problem)
    end
    return problem
end

"""
Reverse the terminal phase to match C++ `Phase::reverse()` at
sdpa_parts.cpp:1013-1024. Swaps pFEAS↔dFEAS, pFEAS_dINF↔pINF_dFEAS,
pUNBD↔dUNBD. Symmetric phases (noINFO, pdFEAS, pdINF, pdOPT) are
unchanged.
"""
function _reverse_phase(phase::PhaseValue)
    phase === pFEAS      ? dFEAS :
    phase === dFEAS      ? pFEAS :
    phase === pFEAS_dINF ? pINF_dFEAS :
    phase === pINF_dFEAS ? pFEAS_dINF :
    phase === pUNBD      ? dUNBD :
    phase === dUNBD      ? pUNBD :
    phase  # noINFO, pdFEAS, pdINF, pdOPT, not_called
end

"""
Transform a SolutionLog for REVERSE_PRIMAL_DUAL display.

Mirrors C++ conventions:
- `IO::printOneIteration` REVERSE branch at sdpa_io.cpp:765-781:
  swap theta, negate+swap objectives, swap alpha.
- `printLastInfo` at sdpa_io.cpp:905-912: swap residuals.
- `Phase::reverse()` at sdpa_parts.cpp:1013: swap terminal phase.
- Solution output at sdpa_io.cpp:1175+: swap xMat↔zMat.

**Note on iterate values**: Because we negate C before solving, our
internal X' (argmin of ⟨-C,X⟩) differs from C++'s internal X (argmin
of ⟨C,X⟩). The xMat↔zMat swap is structurally correct (matching C++
display convention), but the actual matrix values may differ from C++
output for problems where X* ≠ Z* at the optimum. Objective values
are verified correct (strong duality holds at 1e-20 rtol).
"""
function _reverse_display(log::SolutionLog{T}) where {T<:Real}
    new_traj = map(log.trajectory) do row
        IterationRow{T}(
            row.iter, row.mu,
            row.thetaD, row.thetaP,     # swap theta
            -row.objD, -row.objP,       # swap + negate objectives
            row.alphaD, row.alphaP,     # swap alpha
            row.beta,
        )
    end
    return SolutionLog{T}(
        log.iterations,
        _reverse_phase(log.terminal_phase),  # swap phase
        new_traj,
        -log.final_obj_dual,            # swapped + negated objP
        -log.final_obj_primal,          # swapped + negated objD
        log.final_mu,
        log.final_dual_res,             # swapped residuals
        log.final_primal_res,
        log.zMat,                       # displayed xMat = internal zMat
        log.yVec,                       # yVec stays (no counterpart to swap)
        log.xMat,                       # displayed zMat = internal xMat
        log.timings,                    # passthrough
        log.timed_out,                  # passthrough
        log.exit_reason,                # passthrough: why the loop ended
                                        # is invariant under the display swap
    )
end

"""
    solve!(problem::SDPAProblem{T}, params::Parameters{T};
           verbose::Bool=true, io::IO=stderr,
           reverse_primal_dual::Bool=true,
           force_sparse::Bool=false,
           init::Union{Nothing, SDPASolution{T}}=nothing,
           time_limit_sec::Union{Nothing,Real}=nothing) -> SolutionLog{T}

Solve the SDP problem using primal-dual Mehrotra predictor-corrector
interior-point method. Returns the solution log.

If `init` is an `SDPASolution` (e.g. from `read_ini_s`), the solve
warm-starts from a **value copy** of that point — the caller's `init`
object is never mutated, so it can be inspected or reused after the
solve. In the copy, X/y/Z are used as-is (Cholesky inverses must
already be computed — `read_ini_s` does this) and the initial μ is set
to ⟨X, Z⟩/nDim, mirroring C++
`AverageComplementarity::initialize(Solutions&)`.

`time_limit_sec`: a wall-clock limit in seconds, checked at the top of
each IPM iteration. When it fires, the loop breaks early and the returned
`SolutionLog` has `timed_out=true` (`exit_reason == TIME_LIMIT`; the
terminal phase left as-is). `nothing` (default) imposes no limit.

## REVERSE_PRIMAL_DUAL convention
C++ SDPA-GMP uses `REVERSE_PRIMAL_DUAL=1` which effectively solves
`min⟨-C,X⟩` and displays results with primal/dual swapped and negated.
When `reverse_primal_dual=true` (default), this function:
1. Negates C before solving
2. Restores C after solving
3. Transforms the SolutionLog for REVERSE display

`reverse_primal_dual=true` is the C++ SDPA-GMP `REVERSE_PRIMAL_DUAL=1`
parity default: the solver negates C internally, displayed values follow
the C++ convention, and the file format stores the constant with the
flipped sign (`read_dat_s` reads +value where C++ reads -value).

**Limitation**: with the negated problem bounded, unboundedness or
infeasibility of the standard-form problem may NOT be detected — the
negated problem terminates `pdOPT` and the display maps it back. Set
`reverse_primal_dual=false` for the standard SDPA formulation
(`min ⟨C,X⟩` semantics with direct detection of unbounded/infeasible
status).

## Precision contract
- If `T == BigFloat`, the entire solve is wrapped in
  `setprecision(BigFloat, params.precision) do ... end`.
- If `T == Float64`, no precision wrapping is needed.

## Thread safety
`solve!` does NOT mutate the caller's problem: with `reverse_primal_dual=true`
it solves a shallow copy whose `C` is a negated value copy (the caller's
`problem.C` is never touched, on any path). On the
`reverse_primal_dual=false` path the problem object is used directly and its
derived aggregate index tables are populated via `initialize_index!` **only
if never initialized** (readers and MOI ingest always initialize them
first). The process-global BLAS thread policy is installed for the
duration of the solve, so `_BLAS_POLICY_LOCK` (a `ReentrantLock`) is held
across the entire solve (allocation, ordering, and the full IPM
iteration loop, including the policy set/restore): concurrent `solve!`
calls fully serialize — one solve at a time — and independent problems
cannot be solved concurrently within one process. Remaining shared
state: the `SDPA_*` ENV
knobs are read once at solve start
(`_validate_sdpa_env_vars` / ordering routing).
"""
function solve!(
    problem::SDPAProblem{T}, params::Parameters{T};
    verbose::Bool=true, io::IO=stderr,
    reverse_primal_dual::Bool=true,
    force_sparse::Bool=false,
    init::Union{Nothing, SDPASolution{T}}=nothing,
    time_limit_sec::Union{Nothing,Real}=nothing,
    # Internal testing hook: throws after the BLAS policy is set. Not
    # part of the public API.
    _test_throw_after_blas::Bool=false,
) where {T<:Real}
    # Never mutate the caller's problem. With the REVERSE
    # convention the solver needs -C internally (C++ sdpa_tool.h:84), so
    # solve a shallow copy whose C is a negated value copy. The caller's
    # problem is left untouched on every path, making solve! safe to call
    # concurrently on the same problem object.
    if reverse_primal_dual
        problem = _problem_with_negated_C(problem)
    else
        # This path solves the caller's problem object directly. Readers
        # and MOI ingest populate the derived aggregate index tables, but
        # a manually constructed SDPAProblem never has initialize_index!
        # called — with empty tables the solve would silently produce a
        # zero Schur complement, wrong objectives, and noINFO. Populate
        # the tables if they were never initialized (derived data only;
        # no user-visible mutation).
        _ensure_index_tables!(problem)
    end
    if T <: BigFloat
        log = setprecision(BigFloat, params.precision) do
            return _solve_inner!(problem, params, verbose, io, force_sparse, init;
                                 time_limit_sec=time_limit_sec,
                                 _test_throw_after_blas=_test_throw_after_blas)
        end
    else
        log = _solve_inner!(problem, params, verbose, io, force_sparse, init;
                            time_limit_sec=time_limit_sec,
                            _test_throw_after_blas=_test_throw_after_blas)
    end
    # Apply the REVERSE display transformation whenever reverse_primal_dual
    # is set (regardless of the solve's terminal phase).
    if reverse_primal_dual
        log = _reverse_display(log)
    end
    return log
end

# ---------------------------------------------------------------------------
# Block-level ordering graph (mirrors C++ Spooles ND/MS on the BLOCK-level
# graph, sdpa_chordal.cpp:141-211). Returns the METIS ordering of the
# block-level adjacency graph (constraints connected iff they share an SDP
# or LP block). Used by the SDPA_ORDERING / SDPA_BLOCK_ORDERING routing
# branches below.
function _block_level_ordering(problem::SDPAProblem{T}) where {T<:Real}
    m = problem.m
    # Pre-size to the exact edge count to avoid 2×-growth reallocation
    # fragmentation (the edge lists reach tens of millions of entries on
    # large problems; un-sized push! growth then fragments the GC heap).
    n_edges = sum(nC * (nC - 1) for nC in problem.SDP_nConstraint if nC > 1; init=0) +
              sum(nC * (nC - 1) for nC in problem.LP_nConstraint if nC > 1; init=0)
    rows = Int[]; cols = Int[]
    sizehint!(rows, n_edges)
    sizehint!(cols, n_edges)
    for l in 1:length(problem.SDP_nConstraint)
        cs = problem.SDP_constraint[l]
        for a in 1:length(cs), b in (a+1):length(cs)
            push!(rows, cs[a]); push!(cols, cs[b]); push!(rows, cs[b]); push!(cols, cs[a])
        end
    end
    for l in 1:length(problem.LP_nConstraint)
        cs = problem.LP_constraint[l]
        for a in 1:length(cs), b in (a+1):length(cs)
            push!(rows, cs[a]); push!(cols, cs[b]); push!(rows, cs[b]); push!(cols, cs[a])
        end
    end
    adj_blk = SparseArrays.sparse(rows, cols, trues(length(rows)), m, m, |)
    return SDPA.compute_ordering(adj_blk; method=:metis)
end

# Validate the SDPA_* env-gated ordering/profile knobs up front so an invalid
# value fails fast, BEFORE analyze_sparsity does the expensive aggregate-graph
# + METIS ordering work. SDPA_ORDERING / SDPA_BLOCK_ORDERING error on invalid
# values; SDPA_FORCE_SPARSE_PROFILE is a bench-only knob, so it only warns.
function _validate_sdpa_env_vars()
    ordering = get(ENV, "SDPA_ORDERING", "")
    if !isempty(ordering) && ordering != "block" && ordering != "auto"
        error("SDPA_ORDERING must be 'block', 'auto', or unset; got '$ordering'")
    end
    block_ordering = get(ENV, "SDPA_BLOCK_ORDERING", "")
    if !isempty(block_ordering) && block_ordering != "1"
        error("SDPA_BLOCK_ORDERING must be '1' or unset; got '$block_ordering'")
    end
    force_sparse = get(ENV, "SDPA_FORCE_SPARSE_PROFILE", "")
    if !isempty(force_sparse) && force_sparse != "1"
        @warn "SDPA_FORCE_SPARSE_PROFILE ignored (expected '1'), got '$force_sparse'"
    end
    return nothing
end

# Serialize the process-global BLAS thread policy so concurrent
# solves cannot clobber each other's setting. ReentrantLock: a nested
# solve! from the same task does not deadlock.
const _BLAS_POLICY_LOCK = ReentrantLock()

function _solve_inner!(
    problem::SDPAProblem{T}, params::Parameters{T},
    verbose::Bool, io::IO, force_sparse::Bool=false,
    init::Union{Nothing, SDPASolution{T}}=nothing;
    time_limit_sec::Union{Nothing,Real}=nothing,
    _test_throw_after_blas::Bool=false,
) where {T<:Real}
    lock(_BLAS_POLICY_LOCK)
    try
        return _solve_inner_locked!(
            problem, params, verbose, io, force_sparse, init;
            time_limit_sec=time_limit_sec,
            _test_throw_after_blas=_test_throw_after_blas,
        )
    finally
        unlock(_BLAS_POLICY_LOCK)
    end
end

function _solve_inner_locked!(
    problem::SDPAProblem{T}, params::Parameters{T},
    verbose::Bool, io::IO, force_sparse::Bool=false,
    init::Union{Nothing, SDPASolution{T}}=nothing;
    time_limit_sec::Union{Nothing,Real}=nothing,
    _test_throw_after_blas::Bool=false,
) where {T<:Real}
    _validate_sdpa_env_vars()
    # ---------------------------------------------------------------
    # 1. Allocate state (mirror C++ pinpal at sdpa_main.cpp:228-282).
    # ---------------------------------------------------------------
    # Set BLAS thread policy based on numeric type:
    #   * BigFloat: BLAS is unused (dispatches to GenericLinearAlgebra);
    #     pin to 1 to avoid competing with Julia threads.
    #   * Float64: LAPACK is the hot path; match BLAS threads to Julia
    #     threads so the user's JULIA_NUM_THREADS governs parallelism.
    # See the design note in src/SDPA.jl __init__ for the full rationale.
    prev_blas = BLAS.get_num_threads()
    try
        if T <: BigFloat
            BLAS.set_num_threads(1)
        else
            BLAS.set_num_threads(Threads.nthreads())
        end
        if _test_throw_after_blas
            error("test injection: throw after BLAS policy set")
        end
        t_start = time_ns()
        # The solve mutates current_pt in place (IterateUpdate.update!
        # writes X/y/Z directly), so warm-start from a VALUE COPY — the
        # caller's init object must come back untouched, matching the
        # protection problem already has.
        current_pt = init === nothing ?
            SDPASolution{T}(problem.m, problem.blockStruct, params.lambdaStar) :
            copy(init)
        # Trivial-problem early exit, m == 0: with no constraints the
        # problem min ⟨C, X⟩ s.t. X ⪰ 0 is decidable without any IPM work.
        if problem.m == 0 && current_pt.nDim == 0
            # Nothing to optimize at all: no constraints and no variables.
            # The general path would run the full allocation +
            # sparsity-analysis machinery and return pdOPT with 0
            # iterations, but with a WRONG final_mu: the central-path
            # initial μ = lambdaStar is never updated because ⟨X, Z⟩/nDim
            # is 0/0. Return a deterministic empty pdOPT log directly,
            # before any Newton state, residual, or iteration work. The
            # REVERSE display path tolerates this log: -0.0 == 0.0 and
            # all swaps are over empty structures.
            return SolutionLog{T}(
                0,                     # iterations: no IPM work done
                pdOPT,                 # the empty point is trivially optimal
                IterationRow{T}[],     # empty trajectory
                zero(T),               # ⟨C, ∅⟩ = 0
                zero(T),               # ⟨b, ∅⟩ = 0
                zero(T),               # μ defined 0 (0/0 avoided)
                zero(T),               # ‖b - ⟨A, X⟩‖_∞ = 0
                zero(T),               # ‖C - Σ y·A - Z‖_∞ = 0
                copy(current_pt.xMat),
                copy(current_pt.yVec),
                copy(current_pt.zMat),
                (mehrotra=0.0, eigen=0.0, residual=0.0, total=0.0),
                false,                 # timed_out
                EMPTY_PROBLEM,
            )
        elseif problem.m == 0
            # Variables but no constraints. If C is semidefinite the
            # optimum is 0 at X = 0 (dual: Z = C, y = ∅) — return that
            # directly. The general path cannot conclude this: with no
            # constraints to anchor it, the constraint-free iterate
            # diverges and the run reports an unbounded-looking phase
            # even though the true answer is a finite optimum of 0.
            # (If C has a negative direction the problem IS unbounded and
            # we deliberately fall through to the general path, which
            # reaches that conclusion via the standard loop.)
            z = SDPADenseLinearSpace{T}(problem.blockStruct)
            sdp_slot_of_block = Dict{Int,Int}()
            for l in 1:problem.nBlock
                if problem.blockStruct[l] >= 2
                    sdp_slot_of_block[l] = length(sdp_slot_of_block) + 1
                end
            end
            semidefinite = true
            for k in eachindex(problem.C.SDP_sp_index)
                l = problem.C.SDP_sp_index[k]
                pos = sdp_slot_of_block[l]
                smat = problem.C.SDP_sp_block[k]
                n = smat.nRow
                if smat.type === :dense
                    dense = smat.de_ele
                else
                    # Materialize the upper-triangle triplets into a dense
                    # symmetric matrix (read-only use below; the copy into
                    # z below is value-copying, so no aliasing escapes).
                    dense = zeros(T, n, n)
                    for t in 1:smat.nonZeroEffect
                        r = smat.row_index[t]
                        c = smat.column_index[t]
                        v = smat.sp_ele[t]
                        dense[r, c] = v
                        dense[c, r] = v
                    end
                end
                _copyto_independent!(z.SDP_block[pos].de_ele, dense)
                if min_eigen_qr_block(dense) < 0
                    semidefinite = false
                    break
                end
            end
            if semidefinite
                for lp_k in 1:z.LP_nBlock
                    if get_lp_value(problem.C, lp_k) < 0
                        semidefinite = false
                        break
                    end
                end
            end
            if semidefinite
                x0 = SDPADenseLinearSpace{T}(problem.blockStruct)  # X = 0
                return SolutionLog{T}(
                    0,                     # iterations: decidable, no IPM work
                    pdOPT,                 # optimum 0 at X = 0, Z = C
                    IterationRow{T}[],
                    zero(T),               # ⟨C, 0⟩ = 0
                    zero(T),               # ⟨b=∅, y⟩ = 0
                    zero(T),               # ⟨0, C⟩/nDim = 0
                    zero(T),               # no constraints: primal residual 0
                    zero(T),               # Z = C exactly: dual residual 0
                    x0,
                    copy(current_pt.yVec), # empty
                    z,                     # Z = C (the optimal dual slack)
                    (mehrotra=0.0, eigen=0.0, residual=0.0, total=0.0),
                    false,
                    OPTIMAL_TERMINAL,
                )
            end
        end
        work = SDPAWorkVariables{T}(problem.blockStruct)
        init_res = SDPAResiduals{T}(problem, current_pt)
        compute!(init_res, problem, current_pt)
        current_res = deepcopy(init_res)
        alpha = StepLength{T}()
        beta = DirectionParameter{T}(params.betaStar)
        reduction = Switch(REDUCTION_ON)
        mu = AverageComplementarity{T}(params.lambdaStar)
        # Warm start: mu = ⟨X, Z⟩/nDim from the initial point (mirrors C++
        # AverageComplementarity::initialize(Solutions&), sdpa_parts.cpp:640-645,
        # called after IO::read in pinpal).
        if init !== nothing
            update!(mu, current_pt)
            mu.initial = mu.current
        end
        theta = RatioInitResCurrentRes{T}(params, init_res)
        solve_info = SolveInfo{T}(problem, current_pt, mu.initial, params.omegaStar)
        phase_ref = Ref{PhaseValue}(not_called)
        # Lazy dense-bmat allocation: the m×m dense Schur matrix is only
        # needed on the dense path, so defer it until use_sparse is decided
        # (the forced-sparse path never allocates it — multi-GB at scale).
        newton = NewtonState{T}(problem.m, problem.blockStruct; allocate_dense_bmat=false)
        # Timing accumulators — wall-clock nanoseconds.
        t_mehrotra = Ref{UInt64}(0)
        t_eigen    = Ref{UInt64}(0)
        t_residual = Ref{UInt64}(0)

        # Compute formula selection once (F1/F2/F3 per constraint).
        compute_formula!(newton, problem)

        # Analyze sparsity to decide dense vs sparse Schur complement.
        # force_sparse is threaded through so analyze_sparsity's early
        # gate (skip the position-level graph when m/nBlock already route
        # dense) does not fire when the override below needs the graph.
        use_dense, ordering, adj = analyze_sparsity(problem; force_sparse=force_sparse)
        # EXPERIMENTAL (bench-only, mirrors C++ Spooles ND/MS on the BLOCK-level
        # graph, sdpa_chordal.cpp:141-211): SDPA_BLOCK_ORDERING=1 replaces the
        # position-level aggregate graph with the BLOCK-level graph for ordering.
        # C++ keeps the 4 dense SDP cliques contiguous → ~13M profile;
        # the position-level graph interleaves them → 52M profile (16× more
        # Cholesky work). This env var restores the C++-comparable profile.
        #
        # SDPA_ORDERING generalizes the same switch (precedence: SDPA_ORDERING
        # wins if both are set; SDPA_BLOCK_ORDERING=1 behaves as
        # SDPA_ORDERING=block; SDPA_BLOCK_ORDERING=1 + SDPA_ORDERING=auto → auto):
        #   - "block"  → force the block-level ordering (identical to
        #                SDPA_BLOCK_ORDERING=1);
        #   - "auto"   → route by the POSITION-level closed-profile density
        #                (the fill-in-closed envelope computed with the position
        #                ordering, hoisted before the ordering decision — the
        #                legacy code computes the envelope only after the branch,
        #                for the chosen ordering): block iff density > 0.4.
        #                0.4 is a heuristic threshold (C++ `extend_threshold`
        #                precedent); the mid-density crossover is unvalidated.
        #                The block graph + METIS ordering are built LAZILY,
        #                only when routed to block — the block-graph build
        #                cost is never paid on sparse problems.
        #   - absent   → current behavior exactly (literal pass-through of the
        #                legacy branch structure below).
        # The force_sparse override (position-METIS when force_sparse &&
        # use_dense) below applies with the same precedence to both env names.
        sdp_ordering = get(ENV, "SDPA_ORDERING", "")
        sdp_block_ordering = get(ENV, "SDPA_BLOCK_ORDERING", "")
        if !isempty(sdp_ordering)
            if sdp_ordering == "block"
                if !use_dense
                    ordering = _block_level_ordering(problem)
                    if verbose
                        @info "ordering routing: block (SDPA_ORDERING=block)"
                    end
                end
            elseif sdp_ordering == "auto"
                if !use_dense
                    reverse_ordering = Vector{Int}(undef, problem.m)
                    for (pos, orig) in enumerate(ordering)
                        reverse_ordering[orig] = pos
                    end
                    pos_skyline = SDPA.build_skyline_from_problem!(
                        problem, ordering, reverse_ordering)
                    full_nnz = problem.m * (problem.m + 1) ÷ 2
                    pos_density = sum(pos_skyline[j] - j + 1 for j in 1:problem.m) / full_nnz
                    if pos_density > 0.4
                        ordering = _block_level_ordering(problem)
                        if verbose
                            @info "ordering routing: block (position-level " *
                                  "density=$(round(pos_density*100; digits=2))%)"
                        end
                    elseif verbose
                        @info "ordering routing: position-metis (position-level " *
                              "density=$(round(pos_density*100; digits=2))%)"
                    end
                end
            end
        elseif sdp_block_ordering == "1" && !use_dense
            ordering = _block_level_ordering(problem)
        end
        if force_sparse && use_dense
            problem.m > 0 || error(
                "force_sparse requires at least one constraint (m >= 1); " *
                "with m = 0 there is no Schur complement to factor",
            )
            # Override to sparse path with Metis nested-dissection ordering
            # (matches C++ SDPA-GMP's Spooles ND/MS, more numerically robust
            # than AMD for position-level sparse approximation).
            ordering = SDPA.compute_ordering(adj; method=:metis)
            use_dense = false
        end
        if !use_dense
            newton.use_sparse = true
            newton.ordering = ordering
            # Build reverse ordering for fast lookup.
            newton.reverse_ordering = Vector{Int}(undef, problem.m)
            for (pos, orig) in enumerate(ordering)
                newton.reverse_ordering[orig] = pos
            end
            if verbose
                @info "Using sparse Schur complement path (m=$(problem.m))"
            end
            # Use BLOCK-LEVEL skyline for the density decision (matching C++).
            # C++ Chordal::ordering_bMat checks nBlock <= 5 and
            # SDP_nConstraint[b] > m * sqrt(0.25). If either triggers, dense path.
            # Block-level is broader than position-level because F1/F2/F3 couple
            # all positions within a block via X, Z⁻¹.
            block_skyline = SDPA.build_skyline_from_problem!(
                problem, ordering, newton.reverse_ordering)
            full_nnz = problem.m * (problem.m + 1) ÷ 2
            block_profile_nnz = sum(block_skyline[j] - j + 1 for j in 1:problem.m)
            block_density = block_profile_nnz / full_nnz
            if verbose
                @info "Block-level skyline: $block_profile_nnz entries " *
                      "(density=$(round(block_density*100; digits=2))%)"
            end
            if block_density > 0.25
                if verbose
                    @info "Block-level density >25% — using dense path " *
                          "(F1/F2/F3 couple all constraints within blocks)"
                end
                # EXPERIMENTAL (bench-only, mirrors C++ `#if 1 // SPARSE
                # computation for debugging` in sdpa_chordal.cpp:53-57):
                # SDPA_FORCE_SPARSE_PROFILE=1 forces the sparse path even at
                # 100% density, to benchmark the profile-form Cholesky
                # against the C++ forced-sparse build. Not a documented user
                # feature; dense remains the default.
                if get(ENV, "SDPA_FORCE_SPARSE_PROFILE", "") == "1"
                    sparse_bmat_pre, diagonalIndex_pre = SDPA.profile_from_skyline!(
                        block_skyline, T)
                    newton.sparse_bmat = sparse_bmat_pre
                    newton.diagonalIndex = diagonalIndex_pre
                    make_aggrigate_index_sdp!(newton, problem, ordering)
                    if verbose
                        @info "FORCED sparse profile (SDPA_FORCE_SPARSE_PROFILE=1): " *
                              "$block_profile_nnz entries (density=100%)"
                    end
                else
                    ensure_dense_bmat!(newton)   # dense decision: materialize the bmat
                    newton.use_sparse = false
                end
            else
                # Block-level density is low enough for sparse.
                # Build the profile from the BLOCK-LEVEL skyline — the same
                # `block_skyline` used for the density gate above (F1). This
                # covers every block-level (i,j) pair that
                # make_aggrigate_index_sdp! emits (C++ Chordal::makeGraph
                # connectivity, sdpa_chordal.cpp:141-211), so the F1 @assert
                # inside it can never fire. A position-level skyline would
                # truncate the Schur matrix (it misses block-level pairs).
                if verbose
                    @info "Block-level profile: $block_profile_nnz entries " *
                          "(density=$(round(block_density*100; digits=2))%)"
                end
                # Allocate sparse profile + location table.
                sparse_bmat_pre, diagonalIndex_pre = SDPA.profile_from_skyline!(
                    block_skyline, T)
                newton.sparse_bmat = sparse_bmat_pre
                newton.diagonalIndex = diagonalIndex_pre
                make_aggrigate_index_sdp!(newton, problem, ordering)
            end
        else
            ensure_dense_bmat!(newton)   # dense decision: materialize the bmat
            newton.use_sparse = false
            if verbose
                @info "Using dense Schur complement path"
            end
        end

        trajectory = IterationRow{T}[]
        # Set true when the wall-clock time limit fires (see loop top).
        timed_out = false
        # Why the loop terminated (see the ExitReason docstring for
        # the full assignment rule). OPTIMAL_TERMINAL is the default: it
        # stays set when the loop ends because update_check! returned
        # false (terminal phase reached — including the pdFEAS early stop
        # below, a healthy C++-parity termination) or when the pre-loop
        # update_check! already reported convergence. Every NON-terminal
        # break overwrites it; the maxIteration leg is disambiguated
        # after the loop.
        exit_reason = OPTIMAL_TERMINAL

        # ---------------------------------------------------------------
        # 2. Main loop (mirror sdpa_main.cpp:294-385).
        # ---------------------------------------------------------------
        # First call to update_check! transitions not_called → noINFO.
        phase_ref[] = noINFO
        iter = 0
        local continue_iter::Bool
        continue_iter = update_check!(
            phase_ref,
            current_res.normPrimalVec,
            current_res.normDualMat,
            solve_info.objValPrimal,
            solve_info.objValDual,
            solve_info.rho,
            params,
        )

        while continue_iter && iter < params.maxIteration
            # --- WALL-CLOCK TIME LIMIT ---
            # Checked at the top of every iteration so the limit can never
            # be overshot by more than one iteration's work. Firing leaves
            # phase_ref[] at its current value and marks the log timed out.
            if time_limit_sec !== nothing &&
               (time_ns() - t_start) / 1e9 > time_limit_sec
                timed_out = true
                exit_reason = TIME_LIMIT
                break
            end
            # --- PREDICTOR ---
            mehrotra_predictor!(reduction, phase_ref[])
            mehrotra_predictor!(beta, phase_ref[], reduction, params)
            t0 = time_ns()
            ok = mehrotra!(
                newton, :predictor, problem, current_pt, current_res,
                mu, beta, phase_ref[], work,
            )
            t_mehrotra[] += time_ns() - t0
            if !ok
                # Cholesky failed — break.
                exit_reason = CHOLESKY_FAILURE
                break
            end
            t0 = time_ns()
            mehrotra_predictor!(alpha, problem, current_pt, phase_ref[], newton, work)
            t_eigen[] += time_ns() - t0

            # --- CORRECTOR ---
            mehrotra_corrector!(
                beta, phase_ref[], alpha.primal, alpha.dual,
                current_pt, newton.DxMat, newton.DzMat, mu, params,
            )
            # Early stop (mirror sdpa_main.cpp:337-340):
            # pdFEAS + (beta.value > 5 || objP < objD)
            if phase_ref[] === pdFEAS &&
               (beta.value > 5 || solve_info.objValPrimal < solve_info.objValDual)
                break
            end
            t0 = time_ns()
            ok2 = mehrotra!(
                newton, :corrector, problem, current_pt, current_res,
                mu, beta, phase_ref[], work,
            )
            t_mehrotra[] += time_ns() - t0
            if !ok2
                exit_reason = CHOLESKY_FAILURE
                break
            end
            t0 = time_ns()
            mehrotra_corrector!(
                alpha, problem, current_pt, phase_ref[], reduction,
                newton, mu, theta, work, params,
            )
            t_eigen[] += time_ns() - t0

            # --- LOG + STEP ---
            row = IterationRow{T}(
                iter, mu.current, theta.primal, theta.dual,
                solve_info.objValPrimal, solve_info.objValDual,
                alpha.primal, alpha.dual, beta.value,
            )
            push!(trajectory, row)
            if verbose
                Printf.@printf(io, "%2d %4.1e %4.1e %4.1e %+7.2e %+7.2e %4.1e %4.1e %4.2e\n",
                    row.iter, row.mu, row.thetaP, row.thetaD,
                    row.objP, row.objD, row.alphaP, row.alphaD, row.beta)
                flush(io)
            end
            moved = update!(current_pt, alpha, newton, work)
            if !moved
                # "Cannot move" or Cholesky failure.
                exit_reason = CANNOT_MOVE
                break
            end

            # --- UPDATE BOOKKEEPING ---
            update!(theta, reduction, alpha.primal, alpha.dual)
            update!(mu, current_pt)
            t0 = time_ns()
            compute!(current_res, problem, current_pt)
            t_residual[] += time_ns() - t0
            update_exact!(theta, init_res, current_res)
            update!(solve_info, params.lambdaStar, problem, current_pt,
                    current_res, mu, theta, params)
            # Populate the SolveInfo adapter fields alpha_primal/alpha_dual/
            # beta_value (F1): they are read by the write_result /
            # print_one_iteration adapters but `SolveInfo.update!` leaves them
            # zero. Alpha/beta are computed above and must be recorded here,
            # after the iteration's step-length / centering computation.
            solve_info.alpha_primal = alpha.primal
            solve_info.alpha_dual   = alpha.dual
            solve_info.beta_value   = beta.value

            iter += 1

            # --- TERMINATION CHECK ---
            continue_iter = update_check!(
                phase_ref,
                current_res.normPrimalVec,
                current_res.normDualMat,
                solve_info.objValPrimal,
                solve_info.objValDual,
                solve_info.rho,
                params,
            )
        end

        # The while-condition's maxIteration leg is not a break site,
        # so disambiguate it here. If no non-terminal reason was recorded,
        # the loop was still willing to continue (continue_iter true —
        # false means the terminal check won and OPTIMAL_TERMINAL stands,
        # even when iter happens to equal maxIteration), and the budget is
        # exhausted, the exit was MAX_ITERATION. Every mid-body break above
        # fires BEFORE `iter += 1`, so at any break `iter < maxIteration`
        # still holds — this inequality can only be met via the while leg.
        if exit_reason == OPTIMAL_TERMINAL && continue_iter &&
           iter >= params.maxIteration
            exit_reason = MAX_ITERATION
        end

        # ---------------------------------------------------------------
        # 3. Footer: final residual and return.
        # ---------------------------------------------------------------
        t0 = time_ns()
        compute!(current_res, problem, current_pt)
        t_residual[] += time_ns() - t0
        t_total = time_ns() - t_start
        timings = (
            mehrotra = t_mehrotra[] / 1e9,
            eigen    = t_eigen[]    / 1e9,
            residual = t_residual[] / 1e9,
            total    = t_total      / 1e9,
        )
        return SolutionLog{T}(
            iter,
            phase_ref[],
            trajectory,
            solve_info.objValPrimal,
            solve_info.objValDual,
            mu.current,
            current_res.normPrimalVec,
            current_res.normDualMat,
            copy(current_pt.xMat),
            copy(current_pt.yVec),
            copy(current_pt.zMat),
            timings,
            timed_out,
            exit_reason,
        )
    finally
        BLAS.set_num_threads(prev_blas)
    end
end
