"""
Direct MOI → SDPAProblem conversion (in-memory, no file round-trip).

Reads an MOI model's variable cones, objective, and constraints,
and constructs an SDPAProblem without writing to a temporary file.

Variable cone mapping:
- MOI.PositiveSemidefiniteConeTriangle(n) → SDPA SDP block of size n
- MOI.Nonnegatives(k)                     → SDPA LP block (merged)

The variable-to-block mapping is stored in the Optimizer for
VariablePrimal result queries.
"""

"""
    _triangle_index_to_rowcol(k::Int) -> (row, col)

Convert 1-based index k into column-major upper-triangle to (row, col).
MOI PositiveSemidefiniteConeTriangle uses this ordering:
  k=1 → (1,1), k=2 → (1,2), k=3 → (2,2), k=4 → (1,3), ...

This matches the triangle packing used by MOI.
"""
function _triangle_index_to_rowcol(k::Int)
    col = 1
    remaining = k
    while remaining > col
        remaining -= col
        col += 1
    end
    return (remaining, col)
end

"""
    _assign_var_map!(var_map, vi, target, ::Type{T}) -> var_map

Collision-guarded `var_map` assignment.

`var_map` (VariableIndex → (:SDP, blk, row, col) / (:LP, 1, k, 0) /
(:FREE, ...)) is populated by two passes — the VectorAffineFunction-PSD
pass and the variable-cone pass. A variable claimed by two DIFFERENT
block positions has no faithful direct representation, so a conflicting
second assignment throws instead of silently overwriting — the solved
problem must never silently differ from the user's model (silent
last-writer-wins made the outcome depend on the iteration order of
ListOfConstraintTypesPresent and dropped the first constraint's
reification).

A conflicting assignment throws
`MOI.UnsupportedConstraint{VectorAffineFunction{T},
PositiveSemidefiniteConeTriangle}` — the constraint type whose direct
ingest carries this structural limitation (every collision involves an
affine/cone position claim on the same variable; for a pure cone-vs-cone
collision the same type is used for simplicity). Models with colliding
claims remain solvable through the bridged path
(VectorFunctionizeBridge creates fresh aux variables per PSD
constraint, so no variable is ever claimed twice). Idempotent
same-target re-assignment stays allowed.

The equality-linking pass (`var_map[v2] = var_map[v1]`,
equality-derived identity) and the free-variable split
(`haskey`-guarded) are separate mechanisms and deliberately do NOT
route through this guard.
"""
function _assign_var_map!(
    var_map::Dict{MOI.VariableIndex,Tuple{Symbol,Int,Int,Int}},
    vi::MOI.VariableIndex,
    target::Tuple{Symbol,Int,Int,Int},
    ::Type{T},
) where {T<:Real}
    existing = get(var_map, vi, nothing)
    if existing !== nothing && existing != target
        # Same variable claimed by two different block positions: no
        # faithful direct-ingest representation exists — reject loudly
        # instead of silently corrupting the model.
        throw(MOI.UnsupportedConstraint{
            MOI.VectorAffineFunction{T}, MOI.PositiveSemidefiniteConeTriangle}())
    end
    var_map[vi] = target
    return var_map
end

"""
    _build_sdpa_problem_from_moi(src::MOI.ModelLike, T::Type) -> (problem, var_map, ...)

Convert an MOI model to an SDPAProblem{T}. Returns the problem, a
mapping from MOI.VariableIndex to (block_type, block_idx, row, col),
the per-cone/equality index lists and per-CI variable slices needed by
the result accessors, and the objective-function constant
offset (type T). See the `return` statement for the exact 11-tuple.

Supported MOI structures:
- Variables in PositiveSemidefiniteConeTriangle → SDP blocks
- Variables in Nonnegatives → LP block
- ScalarAffineFunction objective → C matrix
- ScalarAffineFunction in EqualTo → equality constraints (A_i, b_i)

Any other constraint type throws `MOI.UnsupportedConstraint{F,S}` rather than
being silently dropped (the solver would otherwise report OPTIMAL on a
relaxed problem).

Free variables declared via `MOI.Reals` are supported through the
standard x = x⁺ − x⁻ split into two Nonnegatives halves; the
primal value is reconstructed as x⁺ − x⁻. Cone-less scalars added via
`MOI.add_variable` (no cone at all) are rejected with
`MOI.UnsupportedConstraint{VariableIndex, Reals}`: silently modeling
them as nonnegatives produced wrong optima.
"""
function _build_sdpa_problem_from_moi(
    src::MOI.ModelLike, ::Type{T}
) where {T<:Real}
    # Query src directly — the CachingOptimizer's model_cache supports all
    # standard MOI queries. Querying src directly (no intermediate copy)
    # avoids the constraint index renumbering that UniversalFallback/Model
    # copies introduce.
    model = src
    # Collect ALL constraint indices for IndexMap (CachingOptimizer requires
    # every constraint to be mapped, else post-solve lookups throw KeyError).
    all_cis = MOI.ConstraintIndex[]
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        for ci in MOI.get(model, MOI.ListOfConstraintIndices{F, S}())
            push!(all_cis, ci)
        end
    end
    # ---- Discover variable cones and build blockStruct ----
    sdp_dims = Int[]           # side dimension of each SDP block
    sdp_vars = Vector{Vector{MOI.VariableIndex}}()  # vars per SDP block
    lp_vars = MOI.VariableIndex[]
    # Declare var_map early so VectorAffineFunction PSD handling can populate it.
    var_map = Dict{MOI.VariableIndex,Tuple{Symbol,Int,Int,Int}}()
    # Track constant entries in VectorAffineFunction PSD constraints.
    # Each entry: (block_idx, row, col, constant_value)
    psd_constants = Tuple{Int,Int,Int,T}[]
    psd_ci_list = MOI.ConstraintIndex[]  # PSD cone CIs in block order
    lp_ci_list = MOI.ConstraintIndex[]   # LP cone CIs
    # Vars per LP-cone CI, parallel to lp_ci_list (user order) — the
    # per-CI primal/dual slices are built from this (mirrors reals_ci_vars).
    lp_ci_vars = Vector{Vector{MOI.VariableIndex}}()
    free_vars = MOI.VariableIndex[]          # Reals-constrained variables
    reals_ci_list = MOI.ConstraintIndex[]    # Reals cone CIs in block order
    reals_ci_vars = Vector{Vector{MOI.VariableIndex}}()  # vars per Reals CI

    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        if F === MOI.VectorOfVariables
            cis = MOI.get(model, MOI.ListOfConstraintIndices{F, S}())
            for ci in cis
                set = MOI.get(model, MOI.ConstraintSet(), ci)
                func = MOI.get(model, MOI.ConstraintFunction(), ci)
                if S <: MOI.PositiveSemidefiniteConeTriangle
                    if MOI.side_dimension(set) == 1
                        # A 1×1 PSD cone is mathematically identical to
                        # nonnegativity (R_+). Ingest it as Nonnegatives so the
                        # solver convention holds (SDP block >= 2, LP == 1 or
                        # < 0): a PSD(1) SDP block would otherwise break the
                        # global_blk mapping (blockStruct[k] == 1 is not >= 2)
                        # and route the variable's coefficients into the wrong
                        # SDP block, or crash with "SDP block index 0".
                        # The Nonnegatives path below already handles the
                        # var_map / blockStruct building for these variables.
                        append!(lp_vars, func.variables)
                        push!(lp_ci_list, ci)
                        push!(lp_ci_vars, copy(func.variables))
                    else
                        push!(sdp_dims, MOI.side_dimension(set))
                        push!(sdp_vars, func.variables)
                        push!(psd_ci_list, ci)
                    end
                elseif S <: MOI.Nonnegatives
                    append!(lp_vars, func.variables)
                    push!(lp_ci_list, ci)
                    push!(lp_ci_vars, copy(func.variables))
                elseif S <: MOI.Reals
                    # Free variables are split x = x⁺ − x⁻ into two
                    # Nonnegatives halves further below.
                    append!(free_vars, func.variables)
                    push!(reals_ci_list, ci)
                    push!(reals_ci_vars, copy(func.variables))
                else
                    # Any cone other than PSD/Nonnegatives/Reals is
                    # unsupported — it must not be silently dropped.
                    throw(MOI.UnsupportedConstraint{F, S}())
                end
            end
        elseif F <: MOI.VectorAffineFunction && S <: MOI.PositiveSemidefiniteConeTriangle
            # Handle VectorAffineFunction PSD constraints (e.g., from Convex.jl).
            # These express "M(v) ⪰ 0" where M is an affine matrix function.
            # We convert to: X = M(v) as a new SDP block, with equality
            # constraints for constant entries.
            cis = MOI.get(model, MOI.ListOfConstraintIndices{F, S}())
            for ci in cis
                set = MOI.get(model, MOI.ConstraintSet(), ci)
                func = MOI.get(model, MOI.ConstraintFunction(), ci)
                n = MOI.side_dimension(set)
                if n == 1
                    # Affine 1×1 PSD is out of scope for the direct conversion:
                    # the term-mapping below assumes a triangle position, and a
                    # scalar PSD is just nonnegativity (handled by the
                    # VectorOfVariables path). Honest error rather than a
                    # silently mis-built problem.
                    throw(MOI.UnsupportedConstraint{F, S}())
                end
                block_idx = length(sdp_dims) + 1
                push!(sdp_dims, n)
                push!(sdp_vars, MOI.VariableIndex[])  # placeholder
                push!(psd_ci_list, ci)

                # Group terms by output_index.
                n_tri = div(n * (n + 1), 2)
                terms_by_output = Dict{Int, Vector{Tuple{T, MOI.VariableIndex}}}()
                for term in func.terms
                    d = term.output_index
                    coeff = term.scalar_term.coefficient
                    var = term.scalar_term.variable
                    if !haskey(terms_by_output, d)
                        terms_by_output[d] = Tuple{T, MOI.VariableIndex}[]
                    end
                    push!(terms_by_output[d], (coeff, var))
                end

                for d in 1:n_tri
                    row, col = _triangle_index_to_rowcol(d)
                    terms = get(terms_by_output, d, Tuple{T, MOI.VariableIndex}[])
                    constant = func.constants[d]

                    if length(terms) == 1 && isone(terms[1][1]) && iszero(constant)
                        # Simple: X[row,col] = variable. Map directly
                        # (guarded — a second, different claim on the
                        # same variable throws instead of overwriting).
                        _assign_var_map!(
                            var_map, terms[1][2], (:SDP, block_idx, row, col), T)
                    elseif length(terms) == 0 && !iszero(constant)
                        # Constant entry: X[row,col] = constant → equality constraint.
                        push!(psd_constants, (block_idx, row, col, constant))
                    elseif length(terms) == 0 && iszero(constant)
                        # X[row,col] = 0 → equality constraint X[row,col] = 0.
                        # (Only needed if the position must be explicitly zeroed.)
                        push!(psd_constants, (block_idx, row, col, constant))
                    else
                        throw(MOI.UnsupportedConstraint{F, S}())
                    end
                end
            end
        elseif !(F <: MOI.ScalarAffineFunction)
            # Any other constraint type is unsupported — it must not be
            # silently dropped. ScalarAffineFunction is deferred to the
            # equality pass below, where only EqualTo is accepted.
            throw(MOI.UnsupportedConstraint{F, S}())
        end
    end

    # Build variable → (block_type, block_idx, row, col) mapping.
    # var_map was declared before constraint discovery to allow
    # VectorAffineFunction PSD handling to populate it directly.
    # (blockStruct is built AFTER the cone-less-scalar check, so that
    # LP scalar variables appended there are included in the LP dimension.)
    # Both cone-pass loops assign through the collision guard — a
    # variable already claimed by the VAF-PSD pass (or by another cone)
    # throws instead of being silently re-mapped.
    for (blk, vars) in enumerate(sdp_vars)
        for (k, vi) in enumerate(vars)
            r, c = _triangle_index_to_rowcol(k)
            _assign_var_map!(var_map, vi, (:SDP, blk, r, c), T)
        end
    end
    for (k, vi) in enumerate(lp_vars)
        _assign_var_map!(var_map, vi, (:LP, 1, k, 0), T)
    end

    # ---- Collect equality constraints in ONE pass ----
    # We collect them first so we can do variable linking without
    # querying the source model a second time (which can fail with
    # KeyError on some MOI model layers).
    eq_constraints = Tuple{MOI.ConstraintIndex, Vector{<:MOI.ScalarAffineTerm{T}}, T}[]
    # eq_constraints stores (ci, func.terms, EFFECTIVE rhs =
    # set.value - func.constant). The user's constraint is
    # `Σc·x + const == rhs`, i.e. `Σc·x == rhs - const`; storing the
    # effective rhs folds the SAF constant into the solver model (b-row
    # construction below and the redundancy filter both consume this
    # tuple). The RAW constant is still recorded per-CI in
    # eq_func_constants so ConstraintPrimal reports the true function
    # value `Σc·x + const` (the activity) at the solution point; an
    # all-cancelling equality with nonzero constant (e.g. `0·x + 5 == 0`)
    # is infeasible and is KEPT as a zero-A row instead of being
    # filtered as redundant.
    eq_func_constants = Dict{MOI.ConstraintIndex,T}()
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        if F <: MOI.ScalarAffineFunction
            if S <: MOI.EqualTo
                cis = MOI.get(model, MOI.ListOfConstraintIndices{F, S}())
                for ci in cis
                    func = MOI.get(model, MOI.ConstraintFunction(), ci)
                    set = MOI.get(model, MOI.ConstraintSet(), ci)
                    # T-mismatch: an equality with a different coefficient
                    # type (e.g. ScalarAffineFunction{Float64} into an
                    # Optimizer{BigFloat}) is rejected up front with a
                    # clean UnsupportedConstraint (a raw conversion would
                    # die on a MethodError).
                    # (The objective path intentionally tolerates mismatched
                    # coefficients by conversion — Convex.jl may emit
                    # Float64 objectives — but equality terms are stored
                    # typed in eq_constraints, so mismatch is rejected.)
                    func isa MOI.ScalarAffineFunction{T} ||
                        throw(MOI.UnsupportedConstraint{F, S}())
                    push!(eq_constraints,
                          (ci, func.terms, T(set.value) - func.constant))
                    # Record the raw SAF constant for the activity
                    # (see eq_func_constants declaration). func.constant is
                    # already ::T (checked above).
                    eq_func_constants[ci] = func.constant
                end
            else
                # ScalarAffineFunction in anything but EqualTo (LessThan,
                # GreaterThan, Interval, ...) is unsupported — it must not
                # be silently dropped.
                throw(MOI.UnsupportedConstraint{F, S}())
            end
        end
    end

    # Resolve variable links from collected equality constraints.
    # Convex/JuMP may create auxiliary PSD variables and link them to
    # objective/constraint variables through identity equalities like:
    #   1.0*v_psd - 1.0*v_free = 0
    # We detect these and map the free variable to the same position.
    # Linking identifies two variables ONLY for a pure identity
    # ±1·v1 ∓ 1·v2 == 0 (unit coefficients, zero rhs): a scaled pair
    # (2.0*v1 - 1.0*v2 == 0 means v2 = 2·v1, NOT v2 = v1) or a nonzero
    # rhs (v1 - v2 == 3) is a different relation and stays an ordinary
    # row — linking those would silently replace the user's model.
    # Run to convergence (order-independent).
    changed = true
    while changed
        changed = false
        for (_, terms, rhs) in eq_constraints
            length(terms) == 2 || continue
            iszero(rhs) || continue
            c1, c2 = terms[1].coefficient, terms[2].coefficient
            (isone(c1) && isone(-c2)) || (isone(-c1) && isone(c2)) || continue
            v1, v2 = terms[1].variable, terms[2].variable
            if haskey(var_map, v1) && !haskey(var_map, v2)
                var_map[v2] = var_map[v1]
                changed = true
            elseif haskey(var_map, v2) && !haskey(var_map, v1)
                var_map[v1] = var_map[v2]
                changed = true
            end
        end
    end

    # Map the Reals-constrained free variables to their x⁺ / x⁻ LP halves
    # (position nLP_base + k is x⁺, nLP_base + nfree + k is x⁻).
    # A free variable that the linking pass already mapped to an SDP
    # position (Convex/JuMP auxiliary pattern) keeps that mapping; its
    # split halves are simply unused.
    nLP_base = length(lp_vars)
    nfree = length(free_vars)
    for (k, vi) in enumerate(free_vars)
        haskey(var_map, vi) ||
            (var_map[vi] = (:FREE, nLP_base + k, nLP_base + nfree + k, 0))
    end

    # ---- Reject cone-less scalar variables loudly ----
    # Variables referenced by an equality or the objective but present in
    # NO cone (plain MOI.add_variable scalars) must NOT be silently
    # modeled as LP-nonnegatives: a genuinely free variable
    # (e.g. min x s.t. x + y = 1 with x, y free → unbounded below) would
    # be solved as x, y ≥ 0 (→ min 0) — a silent wrong answer.
    # Reals-DECLARED variables are supported via the split above;
    # genuinely cone-less scalars fail with the MOI-contract error.
    # (JuMP users are already bridged to Nonnegatives and are unaffected.)
    F_obj = MOI.get(model, MOI.ObjectiveFunctionType())
    obj_type = MOI.ObjectiveFunction{F_obj}
    # The solver represents objectives as ScalarAffineFunction coefficients
    # in C (MIN: +coeff, MAX: -coeff). Any other function type (e.g.
    # VectorAffineFunction) has no representation here — reject it cleanly
    # with the MOI-contract error instead of failing later with a raw
    # MethodError on term.variable (VectorAffineTerm has .scalar_term,
    # not .variable).
    F_obj <: MOI.ScalarAffineFunction ||
        throw(MOI.UnsupportedAttribute(obj_type()))
    obj = MOI.get(model, obj_type())
    unmapped = Set{MOI.VariableIndex}()
    for (_, terms, _) in eq_constraints
        for term in terms
            haskey(var_map, term.variable) || push!(unmapped, term.variable)
        end
    end
    for term in obj.terms
        haskey(var_map, term.variable) || push!(unmapped, term.variable)
    end
    isempty(unmapped) ||
        throw(MOI.UnsupportedConstraint{MOI.VariableIndex, MOI.Reals}())

    # MOI conformance: a bare MOI.add_variable scalar used by NO cone,
    # equality, or objective never enters var_map (and thus never enters
    # the returned IndexMap) — copy_to would succeed and the first
    # post-solve value(x) would die with a raw KeyError inside
    # MOIU.IndexMap. Cross-check every model variable here: cone'd vars,
    # Reals vars (:FREE split), and linked vars are all in var_map by this
    # point, so only truly cone-less strays are rejected — at copy time,
    # with the same contract error as above.
    for vi in MOI.get(model, MOI.ListOfVariableIndices())
        haskey(var_map, vi) ||
            throw(MOI.UnsupportedConstraint{MOI.VariableIndex, MOI.Reals}())
    end

    # Build blockStruct: positive for SDP, negative for LP (the LP block
    # covers the Nonnegatives scalars plus the 2·nfree split halves).
    nLP_total = length(lp_vars) + 2 * nfree
    nBlock = length(sdp_dims) + (nLP_total > 0 ? 1 : 0)
    blockStruct = copy(sdp_dims)
    if nLP_total > 0
        push!(blockStruct, -nLP_total)
    end

    # Filter redundant constraints. Variable linking creates constraints
    # like 1.0*VI3 - 1.0*VI_x = 0 that become 0=0 after linking resolves
    # both variables to the same SDP position. These zero-A zero-b constraints
    # cause solver degeneracy (unbounded dual variables), so we skip them.
    # `rhs` here is the EFFECTIVE rhs (set.value - SAF constant), so an
    # all-cancelling equality with a nonzero constant (`0·x + 5 == 0`,
    # infeasible) has rhs = -5 ≠ 0 and is correctly KEPT as a zero-A row
    # instead of being silently dropped as redundant.
    filtered_eq = Tuple{MOI.ConstraintIndex,Vector{<:MOI.ScalarAffineTerm{T}},T}[]
    for (ci, terms, rhs) in eq_constraints
        accum = Dict{Tuple{Symbol,Int,Int,Int},T}()
        for term in terms
            info = var_map[term.variable]
            accum[info] = get(accum, info, zero(T)) + term.coefficient
        end
        is_redundant = all(iszero(v) for v in values(accum))
        if !(is_redundant && iszero(rhs))
            push!(filtered_eq, (ci, terms, rhs))
        end
    end
    eq_constraints = filtered_eq
    eq_ci_list = [ci for (ci, _, _) in eq_constraints]

    # Activity data for the equality ConstraintPrimal accessor —
    # per surviving CI (in eq_ci_list / A-row order): the SAF terms as
    # (variable, coefficient) pairs plus the SAF function constant.
    # Built from the FINAL filtered list so it aligns 1:1 with
    # eq_ci_list (filtered/linking-dropped equalities get no entry and
    # fall to Optimizer's zero-residual branch).
    eq_ci_activity = Vector{Tuple{Vector{Tuple{MOI.VariableIndex,T}},T}}(
        undef, length(eq_constraints))
    for (k, (ci, terms, _)) in enumerate(eq_constraints)
        eq_ci_activity[k] = (
            Tuple{MOI.VariableIndex,T}[
                (term.variable, term.coefficient) for term in terms],
            eq_func_constants[ci],
        )
    end

    m = length(eq_constraints) + length(psd_constants)

    # ---- Build SDPAProblem ----
    # Count SDP and LP blocks for SparseLinearSpace construction (the LP
    # dimension includes the 2·nfree split halves).
    nSDP = length(sdp_dims)
    nLP = nLP_total

    # b vector. rhs is the EFFECTIVE rhs (set.value - SAF constant):
    # the solver row enforces Σc·x = rhs - const, exactly the user's
    # constraint `Σc·x + const == set.value`.
    b_vec = SDPAVector{T}(m)
    for (i, (_, _, rhs)) in enumerate(eq_constraints)
        b_vec.ele[i] = rhs
    end
    # Add psd_constants values (constant PSD entries from VectorAffineFunction)
    for (i, (_, _, _, val)) in enumerate(psd_constants)
        b_vec.ele[length(eq_constraints) + i] = val
    end

    # C matrix (objective)
    # F_obj/obj_type/obj were collected during the cone-less-scalar check.
    # ObjectiveFunctionType() may be ScalarAffineFunction{Float64} even when
    # T=BigFloat (e.g. when Convex.jl creates the model with Float64
    # coefficients).
    C_sls = SDPASparseLinearSpace{T}(nSDP, nLP)
    # Pre-allocate SDP blocks with correct dimensions (mirrors read_dat_s.jl).
    for k in 1:nSDP
        C_sls.SDP_sp_block[k] = SDPASparseMatrix{T}(sdp_dims[k], sdp_dims[k], :sparse, 0)
    end
    # Accumulate objective coefficients per position BEFORE storing
    # (mirrors the A_i path below): setelement_sdp! APPENDS, and MOI
    # permits duplicate objective terms on one variable — without
    # accumulation the stored value would be path-dependent (summed on
    # the sparse path, last-wins after change_to_dense!).
    sense = MOI.get(model, MOI.ObjectiveSense())
    if sense === MOI.MIN_SENSE || sense === MOI.MAX_SENSE
        # SDPA minimizes; negate C for maximization.
        sign = sense === MOI.MAX_SENSE ? -one(T) : one(T)
        c_sdp = Dict{Tuple{Int,Int,Int},T}()
        c_lp = Dict{Int,T}()
        for term in obj.terms
            info = var_map[term.variable]
            c = sign * term.coefficient
            if info[1] === :SDP
                _, blk, row, col = info
                # 0.5 symmetric-fill factor for off-diagonal coefficients
                # (inner products count both triangles, so a coefficient c
                # on the (i,j) triangle variable is stored as c/2).
                entry = row == col ? c : c / 2
                key = (blk, row, col)
                c_sdp[key] = get(c_sdp, key, zero(T)) + entry
            elseif info[1] === :FREE
                # x = x⁺ − x⁻ → +c on the x⁺ slot, −c on x⁻.
                _, pos_plus, pos_minus, _ = info
                c_lp[pos_plus] = get(c_lp, pos_plus, zero(T)) + c
                c_lp[pos_minus] = get(c_lp, pos_minus, zero(T)) - c
            else
                _, _, pos, _ = info
                c_lp[pos] = get(c_lp, pos, zero(T)) + c
            end
        end
        for ((gblk, row, col), coeff) in c_sdp
            SDPA.setelement_sdp!(C_sls, gblk, row, col, coeff)
        end
        for (pos, coeff) in c_lp
            SDPA.setelement_lp!(C_sls, pos, coeff)
        end
    end

    # A_i matrices (constraints)
    A_vec = [SDPASparseLinearSpace{T}(nSDP, nLP) for _ in 1:m]
    for A_sls in A_vec
        for k in 1:nSDP
            A_sls.SDP_sp_block[k] = SDPASparseMatrix{T}(sdp_dims[k], sdp_dims[k], :sparse, 0)
        end
    end
    for (i, (_, terms, _)) in enumerate(eq_constraints)
        # Accumulate coefficients per SDP position to correctly handle
        # variable linking constraints where multiple terms map to the
        # same SDP position (e.g., 1.0*VI3 - 1.0*VI_x = 0 where both
        # VI3 and VI_x map to the same (block,row,col) after linking).
        # Without accumulation, setelement_sdp! would append duplicate
        # sparse entries instead of summing, producing wrong A matrices.
        sdp_accum = Dict{Tuple{Int,Int,Int},T}()
        lp_accum = Dict{Int,T}()
        for term in terms
            info = var_map[term.variable]
            if info[1] === :SDP
                _, blk, row, col = info
                # blk is the SDP-local block index (into sdp_vars/sdp_dims),
                # which IS the global blockStruct index: blockStruct is always
                # [sdp_dims...; -nLP] (the LP block is appended last), and after
                # the PSD(1)→LP mapping every SDP block has size >= 2.
                global_blk = blk
                key = (global_blk, row, col)
                # Off-diagonal coefficients get the 0.5 symmetric-fill
                # factor — the inner products count both triangles, so a
                # coefficient c on the (i,j) triangle variable must be stored
                # as c/2 to contribute c·X[i,j] (same convention as the
                # psd_constants path below).
                entry_coeff = row == col ? term.coefficient : term.coefficient / 2
                sdp_accum[key] = get(sdp_accum, key, zero(T)) + entry_coeff
            elseif info[1] === :FREE
                # x = x⁺ − x⁻ → +c on the x⁺ slot, −c on x⁻.
                _, pos_plus, pos_minus, _ = info
                lp_accum[pos_plus] =
                    get(lp_accum, pos_plus, zero(T)) + term.coefficient
                lp_accum[pos_minus] =
                    get(lp_accum, pos_minus, zero(T)) - term.coefficient
            else
                _, _, pos, _ = info
                lp_accum[pos] = get(lp_accum, pos, zero(T)) + term.coefficient
            end
        end
        for ((gblk, row, col), coeff) in sdp_accum
            SDPA.setelement_sdp!(A_vec[i], gblk, row, col, coeff)
        end
        for (pos, coeff) in lp_accum
            # Out-of-range positions are an internal invariant violation
            # (var_map LP positions are within bounds by construction):
            # setelement_lp! errors loudly on them — do not silently drop.
            SDPA.setelement_lp!(A_vec[i], pos, coeff)
        end
    end
    # Add psd_constants entries (constant PSD positions from VectorAffineFunction).
    # Each entry: ⟨A_i, X⟩ = constant where A_i has a single nonzero.
    # Diagonal: coeff = 1.0. Off-diagonal: coeff = 0.5 (inner product counts both triangles).
    for (idx, (blk, row, col, _)) in enumerate(psd_constants)
        A_sls = A_vec[length(eq_constraints) + idx]
        coeff = row == col ? one(T) : T(1) / 2
        SDPA.setelement_sdp!(A_sls, blk, row, col, coeff)
    end

    # Convert sparse blocks to dense where needed (mirrors read_dat_s.jl).
    for k in 1:nSDP
        if isassigned(C_sls.SDP_sp_block, k)
            SDPA.change_to_dense!(C_sls.SDP_sp_block[k])
        end
        for A_sls in A_vec
            if isassigned(A_sls.SDP_sp_block, k)
                SDPA.change_to_dense!(A_sls.SDP_sp_block[k])
            end
        end
    end

    # Construct the problem.
    b_sdpa = SDPAVector{T}(m)
    copyto!(b_sdpa.ele, b_vec.ele)
    problem = SDPAProblem{T}(m, nBlock, blockStruct, b_sdpa, C_sls, A_vec)
    SDPA.initialize_index!(problem)

    # The ScalarAffineFunction constant offset has no home in
    # SDPAProblem (C holds only linear coefficients), so it is returned
    # separately; Optimizer stores it and adds it back in the
    # ObjectiveValue/DualObjectiveValue accessors.
    # lp_ci_vars (parallel to lp_ci_list) carries each LP-cone CI's
    # variables in user order so the result accessors can return per-CI
    # slices instead of the whole LP_block vector.
    # eq_ci_activity (parallel to eq_ci_list) carries each surviving
    # equality CI's (terms, SAF constant) so ConstraintPrimal can report
    # the constraint function value at the solution point instead of the
    # RHS b (which claimed exact feasibility for every termination).
    return problem, var_map, all_cis, eq_ci_list, psd_ci_list, lp_ci_list,
           lp_ci_vars, reals_ci_list, reals_ci_vars, eq_ci_activity,
           T(obj.constant)
end
