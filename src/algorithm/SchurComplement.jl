"""
Schur complement system for the primal-dual IPM.

This file implements the core linear algebra of the SDPA interior-point
method: the Schur complement matrix B and RHS g in the reduced Newton
system `B · Δy = g`.

Mathematical background
=======================

After eliminating ΔX and ΔZ via the complementarity condition, the
Newton system reduces to the m×m Schur complement system. Using the
Monteiro-Zhang A5 (Jordan-product / HKM-XS) symmetrization (the same
direction as C++ SDPA-GMP):

    B[i,j] = ⟨X · A_i · Z⁻¹, A_j⟩_F     (Frobenius inner product)
    g[i]   = ⟨R · Z⁻¹, A_i⟩_F + primal_residual[i]

where R = βμI − X·Z (predictor) or βμI − X·Z − ΔX·ΔZ (corrector).

Three computational formulas (all numerically equivalent) are selected
per (constraint, block) by a cost model with KAPPA = 1.2:

  F1 (dense A_j): Precompute G_i = X·(A_i·Z⁻¹), then B[i,j] = ⟨G_i, A_j⟩.
  F2 (sparse A_j, dense X): Exploit A_j's sparsity in the contraction.
  F3 (fully sparse): 4-index contraction over nonzero entries of A_i, A_j.

C++ selects per (i, block) by `Newton::computeFormula_SDP`
(sdpa_newton.cpp:694-774) using KAPPA = 1.2.
F3 is NEVER selected for DENSE blocks (asserted in C++ at :768-774).

References
----------
- SDPA paper: Fujisawa-Kojima-Nakata 1997, Mathematical Programming 82.
- Monteiro-Zhang 1998, Math Programming 84:3 (the A5 direction taxonomy).
- C++ source: sdpa_newton.cpp:694-1272.
"""

const KAPPA_FORMULA = 1.2  # Cost-model constant from C++ sdpa_newton.cpp:694.

"""
    compute_rmat!(newton_state, direction, mu, beta, current_pt, work)

Compute `r_zinvMat = R · Z⁻¹` where:
  - Predictor: R = βμI − X·Z, so r_zinv = βμ·Z⁻¹ − X.
  - Corrector: R = βμI − X·Z − Δx·Δz, so r_zinv = βμ·Z⁻¹ − X − Δx·Δz·Z⁻¹.

The corrector's `Δx·Δz·Z⁻¹` term uses the PREDICTOR's Δx, Δz (the only
ones available at this point in the iteration).

Mirrors C++ `Newton::compute_rMat` at `sdpa_newton.cpp:821-838`.
"""
function compute_rmat!(
    newton_state::NewtonState{T},
    direction::Symbol,
    mu::AverageComplementarity{T},
    beta::DirectionParameter{T},
    current_pt::SDPASolution{T},
    work::SDPAWorkVariables{T},
) where {T<:Real}
    target = beta.value * mu.current
    # r_zinvMat = target * invzMat - xMat = βμ·Z⁻¹ − X
    for k in 1:current_pt.invzMat.SDP_nBlock
        if isassigned(current_pt.invzMat.SDP_block, k)
            rblk = newton_state.r_zinvMat.SDP_block[k].de_ele
            iblk = current_pt.invzMat.SDP_block[k].de_ele
            xblk = current_pt.xMat.SDP_block[k].de_ele
            @inbounds for idx in eachindex(rblk)
                rblk[idx] = target * iblk[idx] - xblk[idx]
            end
        end
    end
    @inbounds for k in eachindex(newton_state.r_zinvMat.LP_block)
        newton_state.r_zinvMat.LP_block[k] =
            target * current_pt.invzMat.LP_block[k] - current_pt.xMat.LP_block[k]
    end

    if direction === :corrector
        # work.DLS1 = Δx · Δz · Z⁻¹ (non-symmetric triple product)
        ns_jordan_triple_product!(
            work.DLS1, newton_state.DxMat, newton_state.DzMat,
            current_pt.invzMat, work.DLS2,
        )
        # r_zinvMat -= work.DLS1
        for k in 1:newton_state.r_zinvMat.SDP_nBlock
            if isassigned(newton_state.r_zinvMat.SDP_block, k)
                rblk = newton_state.r_zinvMat.SDP_block[k].de_ele
                wblk = work.DLS1.SDP_block[k].de_ele
                @inbounds for idx in eachindex(rblk)
                    rblk[idx] -= wblk[idx]
                end
            end
        end
        @inbounds for k in eachindex(newton_state.r_zinvMat.LP_block)
            newton_state.r_zinvMat.LP_block[k] -= work.DLS1.LP_block[k]
        end
    end
    return newton_state
end

"""
    make_gvec!(newton_state, direction, problem, current_pt, current_res,
               mu, beta, phase, work)

Assemble the Schur RHS vector `gvec`.

Formula (mirrors C++ `Newton::Make_gVec` at `sdpa_newton.cpp:840-895`):
  1. Compute `r_zinvMat` via `compute_rmat!`.
  2. work.DLS1 = r_zinvMat − X·dual_residual·Z⁻¹   (if dual-infeasible)
     or just r_zinvMat                                (if dual-feasible)
  3. gvec = −⟨A_i, work.DLS1⟩ + primal_residual

The `X·dual_residual·Z⁻¹` term is cached in `x_rd_zinvMat` across
predictor→corrector (computed once in the predictor, reused in the
corrector).
"""
function make_gvec!(
    newton_state::NewtonState{T},
    direction::Symbol,
    problem::SDPAProblem{T},
    current_pt::SDPASolution{T},
    current_res::SDPAResiduals{T},
    mu::AverageComplementarity{T},
    beta::DirectionParameter{T},
    phase::PhaseValue,
    work::SDPAWorkVariables{T},
) where {T<:Real}
    compute_rmat!(newton_state, direction, mu, beta, current_pt, work)

    if phase === pFEAS || phase === noINFO
        if direction === :corrector
            # x_rd_zinvMat was computed in PREDICTOR step.
            # work.DLS1 = r_zinvMat - x_rd_zinvMat
            for k in 1:newton_state.r_zinvMat.SDP_nBlock
                if isassigned(newton_state.r_zinvMat.SDP_block, k)
                    wblk = work.DLS1.SDP_block[k].de_ele
                    rblk = newton_state.r_zinvMat.SDP_block[k].de_ele
                    xblk = newton_state.x_rd_zinvMat.SDP_block[k].de_ele
                    @inbounds for idx in eachindex(wblk)
                        wblk[idx] = rblk[idx] - xblk[idx]
                    end
                end
            end
            @inbounds for k in eachindex(work.DLS1.LP_block)
                work.DLS1.LP_block[k] =
                    newton_state.r_zinvMat.LP_block[k] -
                    newton_state.x_rd_zinvMat.LP_block[k]
            end
        else
            # PREDICTOR: compute x_rd_zinvMat = X · dual_residual · Z⁻¹
            ns_jordan_triple_product!(
                newton_state.x_rd_zinvMat,
                current_pt.xMat, current_res.dualMat, current_pt.invzMat,
                work.DLS2,
            )
            # work.DLS1 = r_zinvMat - x_rd_zinvMat
            for k in 1:newton_state.r_zinvMat.SDP_nBlock
                if isassigned(newton_state.r_zinvMat.SDP_block, k)
                    wblk = work.DLS1.SDP_block[k].de_ele
                    rblk = newton_state.r_zinvMat.SDP_block[k].de_ele
                    xblk = newton_state.x_rd_zinvMat.SDP_block[k].de_ele
                    @inbounds for idx in eachindex(wblk)
                        wblk[idx] = rblk[idx] - xblk[idx]
                    end
                end
            end
            @inbounds for k in eachindex(work.DLS1.LP_block)
                work.DLS1.LP_block[k] =
                    newton_state.r_zinvMat.LP_block[k] -
                    newton_state.x_rd_zinvMat.LP_block[k]
            end
        end
    else
        # dualMat == 0 (dual-feasible): work.DLS1 = r_zinvMat
        copyto!(work.DLS1, newton_state.r_zinvMat)
    end

    # gvec = -<A, work.DLS1> + primal_residual
    multi_inner_product_to_a!(newton_state.gvec, problem, work.DLS1)
    @inbounds for i in 1:problem.m
        newton_state.gvec.ele[i] = -newton_state.gvec.ele[i] + current_res.primalVec.ele[i]
    end
    return newton_state
end

"""
    cal_f1!(G, Aj) -> T

F1 formula: `B[i,j] = ⟨G, A_j⟩_F` where `G = X · A_i · Z⁻¹` is
precomputed by the caller. Just a Frobenius inner product.

Mirrors C++ `Newton::calF1` at `sdpa_newton.cpp:897-898`.
"""
function cal_f1!(G::AbstractMatrix{T}, Aj::SDPASparseMatrix{T}) where {T<:Real}
    # Compute ⟨G, A_j⟩_F. For sparse A_j, the SDPA format stores only
    # upper-triangle entries (i ≤ j). The full Frobenius inner product
    # of two symmetric matrices requires doubling off-diagonal contributions:
    #   ⟨G, A_j⟩ = Σ_{stored (i,j)} A_j[i,j] * G[i,j]
    #            + Σ_{stored (i,j), i≠j} A_j[i,j] * G[j,i]  (mirror)
    # This matches cal_f2!'s off-diagonal handling.
    s = zero(T)
    if Aj.type === :sparse
        @inbounds for nz in 1:Aj.nonZeroEffect
            r = Aj.row_index[nz]
            c = Aj.column_index[nz]
            v = Aj.sp_ele[nz]
            s += v * G[r, c]
            if r != c
                s += v * G[c, r]
            end
        end
    else  # :dense — full Frobenius inner product over all entries
        @inbounds for idx in eachindex(Aj.de_ele)
            s += Aj.de_ele[idx] * G[idx]
        end
    end
    return s
end

"""
    cal_f2!(F, G, X, Aj, hasF2Gcal) -> T

F2 formula for sparse A_j. Mirrors C++ `Newton::calF2` at
`sdpa_newton.cpp:899-937`.

For sparse A_j:
  B[i,j] = Σ_{(α,β)∈nnz(A_j)} A_j[α,β] · (X[:,α] · F[:,β])
         + Σ_{(α,β)∈nnz(A_j), α≠β} A_j[α,β] · (X[:,β] · F[:,α])

where F = A_i · Z⁻¹ (precomputed). The `(X[:,α] · F[:,β])` term is
a dot product of the α-th column of X with the β-th column of F.

For dense A_j: falls back to F1 (computes G = X·F, then ⟨G, A_j⟩).
"""
function cal_f2!(
    F::AbstractMatrix{T}, G::AbstractMatrix{T}, X::AbstractMatrix{T},
    Aj::SDPASparseMatrix{T}, hasF2Gcal::Ref{Bool},
) where {T<:Real}
    s = zero(T)
    n = Aj.nRow
    if Aj.type === :sparse
        @inbounds for idx in 1:Aj.nonZeroEffect
            alpha = Aj.row_index[idx]
            beta = Aj.column_index[idx]
            v1 = Aj.sp_ele[idx]
            # value2 = X[:,alpha] · F[:,beta] = Σ_c X[c,alpha] * F[c,beta]
            v2 = zero(T)
            @simd for c in 1:n
                v2 += X[c, alpha] * F[c, beta]
            end
            s += v1 * v2
            if alpha != beta
                # value2 = X[:,beta] · F[:,alpha]
                v2 = zero(T)
                @simd for c in 1:n
                    v2 += X[c, beta] * F[c, alpha]
                end
                s += v1 * v2
            end
        end
        return s
    else
        # Dense A_j: use F1 path (compute G = X·F if not yet done).
        if !hasF2Gcal[]
            mul!(G, X, F)
            hasF2Gcal[] = true
        end
        @inbounds for idx in eachindex(Aj.de_ele)
            s += Aj.de_ele[idx] * G[idx]
        end
        return s
    end
end

"""
    cal_f3!(X, invZ, Ai, Aj) -> T

F3 formula (fully sparse). Mirrors C++ `Newton::calF3` at
`sdpa_newton.cpp:939-1005`. Both A_i and A_j must be sparse.

4-index contraction:
  B[i,j] = Σ_{(α,β)∈nnz(A_j)} Σ_{(γ,δ)∈nnz(A_i)}
           A_j[α,β] · A_i[γ,δ] · invZ[δ,β] · X[α,γ]
         + (symmetric terms for off-diagonal entries)

The formula exploits the symmetry of both A_i and A_j by doubling
off-diagonal contributions.
"""
function cal_f3!(
    X::AbstractMatrix{T}, invZ::AbstractMatrix{T},
    Ai::SDPASparseMatrix{T}, Aj::SDPASparseMatrix{T},
) where {T<:Real}
    s = zero(T)
    @inbounds for idx1 in 1:Aj.nonZeroEffect
        alpha = Aj.row_index[idx1]
        beta = Aj.column_index[idx1]
        v1 = Aj.sp_ele[idx1]
        sum_val = zero(T)
        for idx2 in 1:Ai.nonZeroEffect
            gamma = Ai.row_index[idx2]
            delta = Ai.column_index[idx2]
            v2 = Ai.sp_ele[idx2]
            plu = v2 * invZ[delta, beta] * X[alpha, gamma]
            sum_val += plu
            if gamma != delta
                plu = v2 * invZ[gamma, beta] * X[alpha, delta]
                sum_val += plu
            end
        end
        s += v1 * sum_val
        if alpha == beta
            continue
        end
        # Off-diagonal of A_j: symmetric contribution with alpha <-> beta
        sum_val = zero(T)
        for idx2 in 1:Ai.nonZeroEffect
            gamma = Ai.row_index[idx2]
            delta = Ai.column_index[idx2]
            v2 = Ai.sp_ele[idx2]
            plu = v2 * invZ[delta, alpha] * X[beta, gamma]
            sum_val += plu
            if gamma != delta
                plu = v2 * invZ[gamma, alpha] * X[beta, delta]
                sum_val += plu
            end
        end
        s += v1 * sum_val
    end
    return s
end

# ---------------------------------------------------------------------------
# BigFloat-specialized in-place variants.
#
# These take a pre-allocated accumulator `s::BigFloat` (and pre-allocated
# scratch BigFloats for intermediate products in F2/F3) and write the
# result in-place via `mp_fma!` — zero allocation per call.
#
# The generic `cal_f1!(G, Aj)` / `cal_f2!(...)` / `cal_f3!(...)` above are
# retained for Float64 (where `s += v * G[r,c]` already lowers to a CPU
# FMA and `zero(T)` is allocation-free).
#
# Aliasing note: `mp_fma!(s, v, G[r,c], s)` is well-defined — MPFR
# supports full aliasing of input and output.
# ---------------------------------------------------------------------------

"""
    cal_f1!(s::BigFloat, G, Aj) -> s

In-place BigFloat F1: writes result into s via mp_fma! (which calls mpfr_fma!).
Zero-allocation; caller pre-allocates `s`.
"""
function cal_f1!(
    s::BigFloat, G::AbstractMatrix{BigFloat}, Aj::SDPASparseMatrix{BigFloat},
)
    mp_zero!(s)
    if Aj.type === :sparse
        @inbounds for nz in 1:Aj.nonZeroEffect
            r = Aj.row_index[nz]
            c = Aj.column_index[nz]
            v = Aj.sp_ele[nz]
            mp_fma!(s, v, G[r, c], s)
            if r != c
                mp_fma!(s, v, G[c, r], s)
            end
        end
    else  # dense
        @inbounds for idx in eachindex(Aj.de_ele)
            mp_fma!(s, Aj.de_ele[idx], G[idx], s)
        end
    end
    return s
end

"""
    cal_f2!(s, v2, F, G, X, Aj, hasF2Gcal) -> s

In-place BigFloat F2. Uses s (mpfr_fma!) for outer accumulator and `v2` for
the inner column-dot-product accumulator. Both must be pre-allocated
BigFloats; zero-allocation per call.
"""
function cal_f2!(
    s::BigFloat, v2::BigFloat,
    F::AbstractMatrix{BigFloat}, G::AbstractMatrix{BigFloat},
    X::AbstractMatrix{BigFloat}, Aj::SDPASparseMatrix{BigFloat},
    hasF2Gcal::Ref{Bool},
)
    mp_zero!(s)
    n = Aj.nRow
    if Aj.type === :sparse
        @inbounds for idx in 1:Aj.nonZeroEffect
            alpha = Aj.row_index[idx]
            beta = Aj.column_index[idx]
            v1 = Aj.sp_ele[idx]
            # v2 = X[:, alpha] · F[:, beta]  (column dot product)
            mp_zero!(v2)
            @inbounds for c in 1:n
                mp_fma!(v2, X[c, alpha], F[c, beta], v2)
            end
            # s += v1 * v2
            mp_fma!(s, v1, v2, s)
            if alpha != beta
                # v2 = X[:, beta] · F[:, alpha]
                mp_zero!(v2)
                @inbounds for c in 1:n
                    mp_fma!(v2, X[c, beta], F[c, alpha], v2)
                end
                # s += v1 * v2
                mp_fma!(s, v1, v2, s)
            end
        end
        return s
    else
        # Dense A_j: use F1 path (compute G = X·F if not yet done).
        if !hasF2Gcal[]
            mul!(G, X, F)
            hasF2Gcal[] = true
        end
        @inbounds for idx in eachindex(Aj.de_ele)
            mp_fma!(s, Aj.de_ele[idx], G[idx], s)
        end
        return s
    end
end

"""
    cal_f3!(s, sum_val, tmp, X, invZ, Ai, Aj) -> s

In-place BigFloat F3 (mpfr_fma! for 4-index contraction). Uses three
pre-allocated BigFloats:
  - `s`       : outer accumulator
  - `sum_val` : inner Σ over A_i nonzeros
  - `tmp`     : single product v2 * invZ[d,b] * X[a,g]  (chained FMA)

Zero-allocation per call. Aliasing-safe (mp_fma! supports full aliasing).
"""
function cal_f3!(
    s::BigFloat, sum_val::BigFloat, tmp::BigFloat,
    X::AbstractMatrix{BigFloat}, invZ::AbstractMatrix{BigFloat},
    Ai::SDPASparseMatrix{BigFloat}, Aj::SDPASparseMatrix{BigFloat},
)
    mp_zero!(s)
    @inbounds for idx1 in 1:Aj.nonZeroEffect
        alpha = Aj.row_index[idx1]
        beta = Aj.column_index[idx1]
        v1 = Aj.sp_ele[idx1]
        mp_zero!(sum_val)
        for idx2 in 1:Ai.nonZeroEffect
            gamma = Ai.row_index[idx2]
            delta = Ai.column_index[idx2]
            v2 = Ai.sp_ele[idx2]
            # tmp = v2 * invZ[delta, beta]
            mp_mul!(tmp, v2, invZ[delta, beta])
            # tmp = tmp * X[alpha, gamma] + sum_val → sum_val  (FMA chain)
            mp_fma!(sum_val, tmp, X[alpha, gamma], sum_val)
            if gamma != delta
                # tmp = v2 * invZ[gamma, beta]
                mp_mul!(tmp, v2, invZ[gamma, beta])
                # sum_val += tmp * X[alpha, delta]
                mp_fma!(sum_val, tmp, X[alpha, delta], sum_val)
            end
        end
        # s += v1 * sum_val
        mp_fma!(s, v1, sum_val, s)
        alpha == beta && continue
        # Off-diagonal of A_j: symmetric contribution with alpha <-> beta
        mp_zero!(sum_val)
        for idx2 in 1:Ai.nonZeroEffect
            gamma = Ai.row_index[idx2]
            delta = Ai.column_index[idx2]
            v2 = Ai.sp_ele[idx2]
            mp_mul!(tmp, v2, invZ[delta, alpha])
            mp_fma!(sum_val, tmp, X[beta, gamma], sum_val)
            if gamma != delta
                mp_mul!(tmp, v2, invZ[gamma, alpha])
                mp_fma!(sum_val, tmp, X[beta, delta], sum_val)
            end
        end
        # s += v1 * sum_val
        mp_fma!(s, v1, sum_val, s)
    end
    return s
end

"""
    compute_formula!(newton_state, problem, kappa=KAPPA_FORMULA)

Select F1/F2/F3 per (constraint, SDP block) using the C++ cost model.
Mirrors C++ `Newton::computeFormula_SDP` at `sdpa_newton.cpp:694-774`.

Cost model (per constraint i, SDP block l):
  n  = block size of A_i's l-th SDP block
  inz = NonZeroEffect of A_i's l-th SDP block
  up  = inz + Σ_{j: jnz < inz, or (jnz==inz and j<i)} jnz

  f1 = kappa * n * inz + n³ + kappa * up
  f2 = kappa * n * inz + kappa * (n+1) * up
  f3 = (2 * kappa * inz + 1) * up

Selection:
  - If A_i block is DENSE: use F1 if f1 < f2, else F2. NEVER F3.
  - If A_i block is SPARSE: use the min of {f1, f2, f3}, unless its
    triplet count can cause it to own a pair with a dense neighbor.
"""
function compute_formula!(
    newton_state::NewtonState{T}, problem::SDPAProblem{T};
    kappa::T=T(KAPPA_FORMULA),
) where {T<:Real}
    m = problem.m
    fill!(newton_state.useFormula, :F1)

    for l in 1:problem.nBlock
        problem.blockStruct[l] >= 2 || continue
        nconstr_l = problem.SDP_nConstraint[l]
        nconstr_l == 0 && continue

        # Compute upNonZeroCount for each constraint touching block l
        # (reusing the pre-allocated scratch, one array per call).
        up_counts = newton_state._up_counts_scratch
        for k1 in 1:nconstr_l
            i = problem.SDP_constraint[l][k1]
            ib = problem.SDP_blockIndex[l][k1]
            inz = problem.A[i].SDP_sp_block[ib].nonZeroEffect
            up = inz
            for k2 in 1:nconstr_l
                j = problem.SDP_constraint[l][k2]
                jb = problem.SDP_blockIndex[l][k2]
                jnz = problem.A[j].SDP_sp_block[jb].nonZeroEffect
                if jnz < inz || (jnz == inz && j < i)
                    up += jnz
                end
            end
            up_counts[i] = up
        end

        # Select formula per constraint.
        for k1 in 1:nconstr_l
            i = problem.SDP_constraint[l][k1]
            ib = problem.SDP_blockIndex[l][k1]
            Ai_block = problem.A[i].SDP_sp_block[ib]
            inz = Ai_block.nonZeroEffect
            n = Ai_block.nRow
            up = T(up_counts[i])

            f1 = kappa * n * inz + n * n * n + kappa * up
            f2 = kappa * n * inz + kappa * (n + 1) * up
            f3 = (2 * kappa * inz + 1) * up

            if Ai_block.type === :dense || inz >= n * Ai_block.nCol
                # Both operands of F3 must be sparse. Ordinarily a dense
                # block owns every dense/sparse pair because its weight is
                # the full slot count. A manually assembled sparse block
                # with repeated triplets may match or exceed that weight.
                newton_state.useFormula[i, l] = (f1 < f2) ? :F1 : :F2
            else
                # SPARSE: min of {f1, f2, f3}.
                if f1 < f2 && f1 < f3
                    newton_state.useFormula[i, l] = :F1
                elseif f2 < f3
                    newton_state.useFormula[i, l] = :F2
                else
                    newton_state.useFormula[i, l] = :F3
                end
            end
        end
    end
    _check_no_F3_on_dense!(problem, newton_state)
    return newton_state
end

"""
    _check_no_F3_on_dense!(problem, newton_state)

Defense-in-depth assertion: F3 formula must NEVER be selected for
dense SDP blocks (mirrors C++ assertion at sdpa_newton.cpp:768-774).
Exported with underscore prefix so it can be unit-tested directly.
"""
function _check_no_F3_on_dense!(
    problem::SDPAProblem{T}, newton_state::NewtonState{T}
) where {T<:Real}
    for l in 1:problem.nBlock
        problem.blockStruct[l] >= 2 || continue
        for k1 in 1:problem.SDP_nConstraint[l]
            i = problem.SDP_constraint[l][k1]
            ib = problem.SDP_blockIndex[l][k1]
            Ai_block = problem.A[i].SDP_sp_block[ib]
            if Ai_block.type === :dense
                @assert newton_state.useFormula[i, l] !== :F3 (
                    "F3 must never be selected for dense blocks " *
                    "(C++ sdpa_newton.cpp:768-774). Block $l, constraint $i."
                )
            end
        end
    end
    return nothing
end

"""
    compute_bmat_dense!(newton_state, problem, current_pt, work)

Fill the dense Schur complement matrix `bmat` using F1/F2/F3 per
constraint. Only the upper triangle is computed; the lower triangle is
filled by symmetry.

Mirrors C++ `Newton::compute_bMat_dense_SDP` at
`sdpa_newton.cpp:1007-1096`.
"""
function compute_bmat_dense!(
    newton_state::NewtonState{T}, problem::SDPAProblem{T},
    current_pt::SDPASolution{T}, work::SDPAWorkVariables{T},
) where {T<:Real}
    m = problem.m
    # Dense-path-only entry point: the forced-sparse path never reaches
    # here, so this is where the lazily-allocated bmat gets materialized
    # (no-op when the default constructor already allocated it).
    bmat = ensure_dense_bmat!(newton_state).de_ele
    # Zero bmat in-place (not _fill_independent! which allocates m² new BigFloats).
    if T <: BigFloat
        @inbounds for idx in eachindex(bmat)
            mp_zero!(bmat[idx])
        end
    else
        fill!(bmat, zero(T))
    end

    # Pre-allocate BigFloat accumulators ONCE per call.
    # These are reused across all (k1, k2) iterations via in-place mp_fma!
    # / mp_add!. For Float64, they're ignored (the existing `value = zero(T)`
    # path is already allocation-free because Float64 is a stack value).
    # NOTE: `BigFloat()` returns NaN on Julia 1.12 — must use
    # `zero(BigFloat)` for proper zero initialization.
    if T <: BigFloat
        value_buf = zero(BigFloat)      # F1/F2/F3 outer accumulator
        v2_buf = zero(BigFloat)         # F2 inner column-dot accumulator
        sum_val_buf = zero(BigFloat)    # F3 inner sum
        tmp_buf = zero(BigFloat)        # F3 single product
    end

    # Reusable F2 G-cache flag (reset per constraint i).
    hasF2Gcal = Ref(false)

    for l in 1:problem.nBlock
        problem.blockStruct[l] >= 2 || continue
        # Find SDP-block index for problem-block l (shared helper; see
        # `_sdp_blocks_up_to`).
        sdp_l = _sdp_blocks_up_to(l, problem.blockStruct)

        xMat_blk = current_pt.xMat.SDP_block[sdp_l].de_ele
        invZ_blk = current_pt.invzMat.SDP_block[sdp_l].de_ele
        work1 = work.DLS1.SDP_block[sdp_l].de_ele
        work2 = work.DLS2.SDP_block[sdp_l].de_ele

        nconstr_l = problem.SDP_nConstraint[l]
        for k1 in 1:nconstr_l
            i = problem.SDP_constraint[l][k1]
            ib = problem.SDP_blockIndex[l][k1]
            Ai = problem.A[i].SDP_sp_block[ib]
            inz = Ai.nonZeroEffect
            formula = newton_state.useFormula[i, l]

            hasF2Gcal[] = false

            if formula === :F1
                # work1 = Ai * invZ  (A_i · Z⁻¹)
                _sparse_dense_mul!(work1, Ai, invZ_blk)
                # work2 = xMat * work1  (X · A_i · Z⁻¹ = G_i)
                sdpa_mul!(work2, xMat_blk, work1)
            elseif formula === :F2
                # work1 = Ai * invZ  (A_i · Z⁻¹ = F_i)
                _sparse_dense_mul!(work1, Ai, invZ_blk)
            end

            for k2 in 1:nconstr_l
                j = problem.SDP_constraint[l][k2]
                jb = problem.SDP_blockIndex[l][k2]
                Aj = problem.A[j].SDP_sp_block[jb]
                jnz = Aj.nonZeroEffect

                # Upper triangle only: skip if (inz < jnz) or (inz == jnz and i < j).
                (inz < jnz) && continue
                (inz == jnz && i < j) && continue

                if T <: BigFloat
                    # In-place path: F1/F2/F3 inner loops use mp_fma! (zero
                    # allocation). The OUTER accumulation is also in-place:
                    # `bmat[i,j]` getindex returns the MUTABLE BigFloat
                    # object reference (not a copy — see test_aliasing.jl),
                    # and `bmat.de_ele` is allocated via _independent_zeros
                    # so every element is an independent object, so
                    # `mp_add!(bmat[i,j], bmat[i,j], value_buf)` mutates the
                    # array element with zero allocation (single rounding,
                    # numerically identical to `bmat[i,j] += value_buf`).
                    if formula === :F1
                        cal_f1!(value_buf, work2, Aj)
                    elseif formula === :F2
                        cal_f2!(value_buf, v2_buf, work1, work2,
                                xMat_blk, Aj, hasF2Gcal)
                    else  # :F3
                        cal_f3!(value_buf, sum_val_buf, tmp_buf,
                                xMat_blk, invZ_blk, Ai, Aj)
                    end
                    if i != j
                        mp_add!(bmat[i, j], bmat[i, j], value_buf)
                        mp_add!(bmat[j, i], bmat[j, i], value_buf)
                    else
                        mp_add!(bmat[i, i], bmat[i, i], value_buf)
                    end
                else
                    # Float64 path (unchanged): zero-alloc because Float64
                    # is a stack value, and `s += v*G` lowers to a CPU FMA.
                    value = zero(T)
                    if formula === :F1
                        value = cal_f1!(work2, Aj)
                    elseif formula === :F2
                        value = cal_f2!(work1, work2, xMat_blk, Aj, hasF2Gcal)
                    else  # :F3
                        value = cal_f3!(xMat_blk, invZ_blk, Ai, Aj)
                    end
                    if i != j
                        bmat[i, j] += value
                        bmat[j, i] += value
                    else
                        bmat[i, i] += value
                    end
                end
            end
        end
    end

    # LP blocks: B[i,j] += X_z_inv * A_i * A_j (scalar products)
    _compute_bmat_lp!(bmat, newton_state, problem, current_pt)

    return newton_state
end

"""
Helper: compute `work = Ai * B` where Ai is sparse-or-dense and B is dense.
For sparse Ai, iterates over stored nonzeros (upper triangle only,
doubling off-diagonal for symmetric result).

BigFloat path uses mp_fma! with a pre-allocated local accumulator.
Array writes use the replace-reference pattern (work[r,k] = deepcopy(buf))
because BigFloat(buf) returns the SAME object (Julia 1.12 identity convert).
"""
function _sparse_dense_mul!(
    work::AbstractMatrix{T}, Ai::SDPASparseMatrix{T}, B::AbstractMatrix{T}
) where {T<:Real}
    # Zero work in-place (not _fill_independent! which allocates n² new BigFloats).
    if T <: BigFloat
        @inbounds for idx in eachindex(work)
            mp_zero!(work[idx])
        end
    else
        fill!(work, zero(T))
    end
    n = Ai.nRow
    local buf::T
    has_buf = false
    if T <: BigFloat
        buf = zero(BigFloat)
        has_buf = true
    end
    if Ai.type === :sparse
        @inbounds for idx in 1:Ai.nonZeroEffect
            r = Ai.row_index[idx]
            c = Ai.column_index[idx]
            v = Ai.sp_ele[idx]
            if has_buf
                for k in 1:n
                    # work[r, k] += v * B[c, k]
                    mp_fma!(buf, v, B[c, k], work[r, k])
                    mp_set!(work[r, k], buf)
                end
                if r != c
                    for k in 1:n
                        # work[c, k] += v * B[r, k]
                        mp_fma!(buf, v, B[r, k], work[c, k])
                        mp_set!(work[c, k], buf)
                    end
                end
            else
                @simd for k in 1:n
                    work[r, k] += v * B[c, k]
                end
                if r != c
                    @simd for k in 1:n
                        work[c, k] += v * B[r, k]
                    end
                end
            end
        end
    else
        sdpa_mul!(work, Ai.de_ele, B)
    end
    return work
end

"""
Helper: build, per LP scalar, the ascending list of constraints whose
compact LP storage contains that scalar. Equivalent to a dense
`LP_sp_block[lp_k] != 0` scan but walks only the stored nonzeros.
"""
function _lp_constraint_lists(problem::SDPAProblem{T}) where {T<:Real}
    # Problem-level LP scalar count (matches xMat.LP_nBlock built from
    # blockStruct); every compact LP_sp_index entry is in 1:lp_n.
    lp_n = lp_scalar_count(problem.blockStruct)
    clists = [Int[] for _ in 1:lp_n]
    for i in 1:problem.m
        Ai = problem.A[i]
        @inbounds for k in 1:Ai.LP_nnz
            push!(clists[Ai.LP_sp_index[k]], i)
        end
    end
    return clists
end

"""
    _lp_constraint_lists!(newton_state, problem) -> Vector{Vector{Int}}

Cached accessor for `_lp_constraint_lists(problem)`: the per-LP-scalar
constraint lists are value-independent for the whole solve (built from
`A[i].LP_sp_index`, which never changes after parsing). They are
computed ONCE on first use and cached on `newton_state` (same lifecycle
as the location table built in solve.jl: one NewtonState per problem,
never invalidated).

ORDER PRESERVATION: construction order — i ascending, LP_sp_index order
within each A_i — is fixed, and every consumer iterates the lists in the
same lp_k-ascending, (i, j) lexicographic order, so accumulation order
is deterministic.

`isempty` is a safe "not yet computed" sentinel: all three consumers
return early when LP_nBlock == 0 BEFORE calling this accessor, so an
empty cache always means "compute now".
"""
function _lp_constraint_lists!(
    newton_state::NewtonState{T}, problem::SDPAProblem{T},
) where {T<:Real}
    if isempty(newton_state._lp_clists)
        newton_state._lp_clists = _lp_constraint_lists(problem)
    end
    return newton_state._lp_clists
end

"""
Helper: LP block contribution to bmat.
B[i,j] += X_lp * Z⁻¹_lp * A_i_lp * A_j_lp  (scalar products).
"""
function _compute_bmat_lp!(
    bmat::AbstractMatrix{T}, newton_state::NewtonState{T},
    problem::SDPAProblem{T}, current_pt::SDPASolution{T},
) where {T<:Real}
    # Iterate only LP scalars that carry at least one nonzero constraint
    # entry (the compact SparseLinearSpace stores ONLY nonzero LP entries),
    # and only the constraint pairs (i, j) that both touch the scalar.
    # The iteration order (lp_k ascending, then (i, j) lexicographic)
    # matches the dense path, so accumulation is bit-identical.
    lp_n = current_pt.xMat.LP_nBlock
    lp_n == 0 && return bmat
    clists = _lp_constraint_lists!(newton_state, problem)
    # Both sides derive their LP scalar count from the same blockStruct
    # classification (lp_scalar_count): a mismatch here would be a
    # constructor invariant violation — fail loudly and close to the cause
    # instead of bounds-erroring deep in the loop below.
    @assert length(clists) == lp_n
    for lp_k in 1:lp_n
        clist = clists[lp_k]
        isempty(clist) && continue
        x_lp = current_pt.xMat.LP_block[lp_k]
        invz_lp = current_pt.invzMat.LP_block[lp_k]
        xz_inv = x_lp * invz_lp
        for a in 1:length(clist)
            i = clist[a]
            ai = get_lp_value(problem.A[i], lp_k)
            for b in a:length(clist)
                j = clist[b]
                aj = get_lp_value(problem.A[j], lp_k)
                val = xz_inv * ai * aj
                bmat[i, j] += val
                if i != j
                    bmat[j, i] += val
                end
            end
        end
    end
    return bmat
end

"""
    _compute_bmat_sparse_lp!(newton_state, problem, current_pt)

Assemble LP block contributions directly into sparse_bmat.sp_ele.
Matches C++ `compute_bMat_sparse_LP` (sdpa_newton.cpp:1227-1250).

For each LP variable that carries at least one nonzero constraint entry,
iterates the constraint pairs (i,j) that both have a nonzero LP entry,
checks if the permuted entry is within the skyline profile, and if so
adds the contribution. The compact `SparseLinearSpace` stores only
nonzeros, so the pairs come from per-LP-scalar constraint lists rather
than a dense `LP_sp_nBlock × m²` scan. The iteration order (lp_k
ascending, then (i,j) lexicographic) is identical to the dense path, so
accumulation is bit-identical.

Uses the same checked profile addressing as SDP assembly.
"""
function _compute_bmat_sparse_lp!(
    newton_state::NewtonState{T}, problem::SDPAProblem{T},
    current_pt::SDPASolution{T},
) where {T<:Real}
    lp_n = current_pt.xMat.LP_nBlock
    lp_n == 0 && return newton_state
    sparse_bmat = newton_state.sparse_bmat
    sparse_bmat === nothing && error("sparse Schur matrix is not allocated")
    diagonalIndex = _schur_offsets(sparse_bmat, newton_state.diagonalIndex)
    reverse_ordering = newton_state.reverse_ordering
    clists = _lp_constraint_lists!(newton_state, problem)

    for lp_k in 1:lp_n
        clist = clists[lp_k]
        isempty(clist) && continue
        x_lp = current_pt.xMat.LP_block[lp_k]
        invz_lp = current_pt.invzMat.LP_block[lp_k]
        xz_inv = x_lp * invz_lp

        for a in 1:length(clist)
            i = clist[a]
            ai = get_lp_value(problem.A[i], lp_k)
            p_i = reverse_ordering[i]

            for b in a:length(clist)
                j = clist[b]
                aj = get_lp_value(problem.A[j], lp_k)
                p_j = reverse_ordering[j]

                ii = min(p_i, p_j)
                jj = max(p_i, p_j)

                loc = _schur_assembly_slot(sparse_bmat, diagonalIndex, ii, jj, "LP")
                val = xz_inv * ai * aj

                if T <: BigFloat
                    mp_add!(sparse_bmat.sp_ele[loc], sparse_bmat.sp_ele[loc], val)
                else
                    sparse_bmat.sp_ele[loc] += val
                end
            end
        end
    end
    return newton_state
end

"""
    threaded_compute_bmat_sparse_lp!(newton_state, problem, current_pt)

Threaded version of `_compute_bmat_sparse_lp!`. Parallelizes the LP
Schur assembly across LP scalars `lp_k` with OhMyThreads `@tasks`, using
the same per-entry value-buffer + sequential-reduction pattern as
`threaded_compute_bmat_sparse_sdp!`.

Different LP scalars CAN contribute to the same `sp_ele[loc]`: two
constraints A_i, A_j that both carry two LP scalars emit the same
permuted (ii, jj) profile entry from each `lp_k`, and the location table
has ONE sp_ele slot per (i, j) with contributions summed (C++
`make_aggrigateIndex_LP`). A direct parallel scatter-add would race.
Instead each (lp_k, pair) value goes into a DISJOINT slot of a flat value
buffer whose layout IS the serial iteration order (lp_k ascending, then
(i, j) lexicographic); a sequential reduction then adds into sp_ele in
that flat order, so the per-loc accumulation sequence is bit-identical to
`_compute_bmat_sparse_lp!` at any thread count.

Uses the compact LP storage: each task iterates the
per-LP-scalar constraint list from `_lp_constraint_lists` (built from
`A[i].LP_sp_index`/`LP_sp_val`), so only nonzero LP entries produce
pairs — no `LP_sp_nBlock × m²` dense scan.
"""
function threaded_compute_bmat_sparse_lp!(
    newton_state::NewtonState{T}, problem::SDPAProblem{T},
    current_pt::SDPASolution{T},
) where {T<:Real}
    lp_n = current_pt.xMat.LP_nBlock
    lp_n == 0 && return newton_state
    sparse_bmat = newton_state.sparse_bmat
    sparse_bmat === nothing && error("sparse Schur matrix is not allocated")
    diagonalIndex = _schur_offsets(sparse_bmat, newton_state.diagonalIndex)
    reverse_ordering = newton_state.reverse_ordering
    clists = _lp_constraint_lists!(newton_state, problem)

    # Flat (lp_k, pair) layout: offsets[lp_k] is the first flat index
    # owned by lp_k, and lp_k's pairs fill offsets[lp_k]+1 : offsets[lp_k+1]
    # in the serial order (a in 1:len, b in a:len). total_pairs == 0 means
    # no LP scalar has any nonzero entry → nothing to assemble.
    offsets = Vector{Int}(undef, lp_n + 1)
    offsets[1] = 0
    for lp_k in 1:lp_n
        c = length(clists[lp_k])
        offsets[lp_k + 1] = offsets[lp_k] + c * (c + 1) ÷ 2
    end
    total_pairs = offsets[end]
    total_pairs == 0 && return newton_state

    # Per-entry buffers; each task writes a DISJOINT slice (its lp_k's
    # pairs). all_values is cached on newton_state (same pattern as
    # threaded_compute_bmat_sparse_sdp!'s _sdp_all_values): every slot is
    # assigned a fresh value before the sequential reduction reads it, so
    # the buffer is allocated once per problem and reused across make_bmat!
    # calls. For BigFloat the assigned `val` objects are fresh (no shared
    # reference), keeping slot independence (fresh BigFloat per slot).
    if length(newton_state._lp_all_values) != total_pairs
        newton_state._lp_all_values = _independent_zeros(T, total_pairs)
    end
    all_values = newton_state._lp_all_values
    lp_locations = Vector{Int}(undef, total_pairs)

    active = Int[lp_k for lp_k in 1:lp_n if !isempty(clists[lp_k])]

    @tasks for lp_k in active  # threaded_compute_bmat_sparse_lp!
        # Batch the per-lp_k bodies: each is often tiny (a handful of
        # pairs); a task per scalar would pay scheduler overhead for
        # near-empty work.
        @set chunksize = max(1, cld(length(active), 4 * Threads.nthreads()))

        clist = clists[lp_k]
        x_lp = current_pt.xMat.LP_block[lp_k]
        invz_lp = current_pt.invzMat.LP_block[lp_k]
        xz_inv = x_lp * invz_lp

        flat = offsets[lp_k]
        for a in 1:length(clist)
            i = clist[a]
            ai = get_lp_value(problem.A[i], lp_k)
            p_i = reverse_ordering[i]

            for b in a:length(clist)
                j = clist[b]
                aj = get_lp_value(problem.A[j], lp_k)
                p_j = reverse_ordering[j]

                ii = min(p_i, p_j)
                jj = max(p_i, p_j)

                loc = _schur_assembly_slot(sparse_bmat, diagonalIndex, ii, jj, "LP")
                flat += 1
                lp_locations[flat] = loc
                val = xz_inv * ai * aj
                all_values[flat] = val
            end
        end
    end

    # Sequential reduction in flat order == the serial iteration order, so
    # the per-loc accumulation sequence is identical → bit-identical
    # sp_ele (same pattern as threaded_compute_bmat_sparse_sdp!).
    @inbounds for k in 1:total_pairs
        loc = lp_locations[k]
        if T <: BigFloat
            mp_add!(sparse_bmat.sp_ele[loc], sparse_bmat.sp_ele[loc], all_values[k])
        else
            sparse_bmat.sp_ele[loc] += all_values[k]
        end
    end

    return newton_state
end


"""
    _fill_sparse_values!(sparse_mat, dense_bmat, ordering, diagonalIndex)

Refill ONLY the values (`sp_ele`) of an existing profile-form
`sparse_mat` from a dense `dense_bmat`, reusing the existing
`row_index`/`column_index` structure (which is value-independent).

Reads the row index from `column_index` for each entry, so it
automatically respects the skyline profile — only entries that exist
in the structure are written.

No production caller; retained as the canonical profile value-fill used
by tests (`test/_profile_from_dense.jl`,
`test/algorithm/test_cpp_parity_minors.jl`,
`test/algorithm/test_use_formula_per_block.jl`): given a dense matrix,
it fills the profile-form values while reusing the existing symbolic
structure, which is exactly the reference pipeline those tests verify
(profile_from_skyline! + this function). Deleting requires a
replacement value-fill path.
"""
function _fill_sparse_values!(
    sparse_mat::SDPASparseMatrix{T}, dense_bmat::AbstractMatrix{T},
    ordering::AbstractVector{<:Integer}, diagonalIndex::Vector{Int},
) where {T<:Real}
    m = length(diagonalIndex) - 1
    @inbounds for j in 1:m
        orig_j = ordering[j]
        for idx in diagonalIndex[j]:(diagonalIndex[j+1]-1)
            i = sparse_mat.column_index[idx]
            orig_i = ordering[i]
            # Must create a NEW BigFloat (not reference copy which aliases,
            # not mp_set! which fails on undef BigFloats from Vector{BigFloat}(undef,n)).
            # deepcopy creates an independent copy at the correct precision.
            if T <: BigFloat
                sparse_mat.sp_ele[idx] = BigFloat(0)
                mp_set!(sparse_mat.sp_ele[idx], dense_bmat[orig_i, orig_j])
            else
                sparse_mat.sp_ele[idx] = dense_bmat[orig_i, orig_j]
            end
        end
    end
    return sparse_mat
end

"""
    make_bmat!(newton_state, problem, current_pt, work)

Dispatcher for Schur complement matrix assembly.

- Sparse path (`use_sparse=true`):
  - SDP-only (no LP blocks at all): assembles directly into
    `sparse_bmat.sp_ele` via `threaded_compute_bmat_sparse_sdp!` when
    `Threads.nthreads() > 1`, else `compute_bmat_sparse_sdp!`.
  - LP blocks present: `threaded_compute_bmat_sparse_sdp!` +
    `threaded_compute_bmat_sparse_lp!` when `Threads.nthreads() > 1`,
    else serial `compute_bmat_sparse_sdp!` + `_compute_bmat_sparse_lp!`.
    The threaded SDP variant is safe on mixed problems: both sparse SDP
    assemblies map problem blocks to SDP-block positions via
    `_sdp_blocks_up_to`. The threaded LP variant mirrors the serial
    accumulation order via a flat value buffer + sequential reduction,
    so results are bit-identical at any thread count.
- Dense path (`use_sparse=false`): zero bmat, run dense assembly via
  `threaded_compute_bmat_dense!` when `Threads.nthreads() > 1`,
  otherwise `compute_bmat_dense!`.

Mirrors C++ `Newton::Make_bMat` at `sdpa_newton.cpp:1251-1272`.
"""
function make_bmat!(
    newton_state::NewtonState{T}, problem::SDPAProblem{T},
    current_pt::SDPASolution{T}, work::SDPAWorkVariables{T},
) where {T<:Real}
    if !newton_state.use_sparse
        # Pure dense path — choose threaded vs serial by global thread count.
        if Threads.nthreads() > 1
            threaded_compute_bmat_dense!(newton_state, problem, current_pt, work)
        else
            compute_bmat_dense!(newton_state, problem, current_pt, work)
        end
        return newton_state
    end

    # Sparse path. Choose the SDP and LP assembly variants and handle LP.
    #
    # `has_lp` must be EXACT: LP blocks are `blockStruct[k] < 2`
    # (SDPAProblem.jl doc; same test as read_dat_s.jl:119 and
    # _lp_block_position). Both compute_bmat_sparse_sdp! and
    # threaded_compute_bmat_sparse_sdp! map problem blocks to SDP-block
    # positions via _sdp_blocks_up_to, so mixed problems take either
    # variant safely; at nthreads > 1 the threaded variants run the SDP
    # and LP assemblies (bit-identical flat-buffer + sequential
    # reduction). Serial variants remain the nthreads == 1 path.
    has_lp = any(bs -> bs < 2, problem.blockStruct)
    if Threads.nthreads() > 1
        # Sparse path with LP blocks: assemble SDP + LP directly into
        # sp_ele. Matches C++ compute_bMat_sparse_SDP +
        # compute_bMat_sparse_LP.
        threaded_compute_bmat_sparse_sdp!(newton_state, problem, current_pt, work)
        if has_lp
            threaded_compute_bmat_sparse_lp!(newton_state, problem, current_pt)
        end
    else
        # Serial sparse path (any has_lp): assemble SDP + LP directly
        # into sp_ele. Matches C++ compute_bMat_sparse_SDP +
        # compute_bMat_sparse_LP.
        compute_bmat_sparse_sdp!(newton_state, problem, current_pt, work)
        if has_lp
            _compute_bmat_sparse_lp!(newton_state, problem, current_pt)
        end
    end

    return newton_state
end

# ---------------------------------------------------------------------------
# Sparse Schur complement numeric assembly.
# ---------------------------------------------------------------------------
# Mirrors C++ Newton::compute_bMat_sparse_SDP at sdpa_newton.cpp:1098-1170.
# Uses the location table built by make_aggrigate_index_sdp!.
#
# See docs/sparse-path.md for the sparse bmat design.

"""
    _sdp_blocks_up_to(l, blockStruct) -> Int

Map a problem-block index `l` (1..nBlock) to its 1-based position in the
SDP-block arrays (`xMat.SDP_block`, `invzMat.SDP_block`, `work.DLS1/DLS2
.SDP_block`): the count of SDP blocks (`blockStruct[k] >= 2`) in `1:l`.

The SDP-block arrays are indexed by the order of the SDP entries of
`blockStruct` (see the `SDPADenseLinearSpace(blockStruct)` constructor),
while the sparse location table (`SDP_block_table_len`) is PROBLEM-block
indexed (sized nBlock; LP blocks have 0 entries). The two coincide only
when the problem has no LP blocks — hence the explicit mapping.

The ONE shared block-position mapping: the serial `compute_bmat_dense!`,
both sparse SDP assemblies (`compute_bmat_sparse_sdp!` /
`threaded_compute_bmat_sparse_sdp!`) and `threaded_compute_bmat_dense!`
(ThreadedOps.jl) all route through this helper.
"""
function _sdp_blocks_up_to(l::Int, blockStruct::AbstractVector{Int})
    pos = 0
    for k in 1:l
        if blockStruct[k] >= 2
            pos += 1
        end
    end
    return pos
end

"""
    compute_bmat_sparse_sdp!(newton_state, problem, current_pt, work)

Assemble the sparse Schur complement matrix directly from
`currentPt.xMat`, `currentPt.invzMat`, and the constraint matrices
`problem.A[i]`, bypassing the dense `bmat` entirely.

Pre-condition: `sparse_bmat` allocated, `diagonalIndex` set, and the
location-table arrays populated (via `make_aggrigate_index_sdp!`).

Post-condition: `sparse_bmat.sp_ele` holds the values of the
permuted/profile-form Schur complement; the dense `bmat` is NOT touched.

Mirrors C++ `Newton::compute_bMat_sparse_SDP` at
`sdpa_newton.cpp:1098-1170`.
"""
function compute_bmat_sparse_sdp!(
    newton_state::NewtonState{T}, problem::SDPAProblem{T},
    current_pt::SDPASolution{T}, work::SDPAWorkVariables{T},
) where {T<:Real}
    sparse_bmat = newton_state.sparse_bmat
    sparse_bmat === nothing &&
        error("compute_bmat_sparse_sdp!: sparse_bmat not allocated")

    # Zero the values; keep structure (row_index/column_index untouched).
    # Use mp_zero! for BigFloat to preserve element independence (required
    # for in-place mp_add! during assembly below).
    if T <: BigFloat
        @inbounds for idx in eachindex(sparse_bmat.sp_ele)
            mp_zero!(sparse_bmat.sp_ele[idx])
        end
    else
        fill!(sparse_bmat.sp_ele, zero(T))
    end

    # Pre-allocate BigFloat accumulators for the in-place
    # F1/F2/F3 path. Same aliasing rules as compute_bmat_dense!:
    # array-element writes use the replace-reference '+=' pattern.
    if T <: BigFloat
        value_buf = zero(BigFloat)
        v2_buf = zero(BigFloat)
        sum_val_buf = zero(BigFloat)
        tmp_buf = zero(BigFloat)
    end

    sdp_block_count = length(newton_state.SDP_block_table_len)
    offset = 0  # flat index into the 5 parallel arrays

    # Reusable F2 G-cache flag (reset per constraint i).
    hasF2Gcal = Ref(false)

    for l in 1:sdp_block_count
        count_l = newton_state.SDP_block_table_len[l]
        count_l == 0 && continue

        # Map problem-block l to its SDP-block position: the location
        # table is problem-block indexed, but xMat/invzMat/DLS1/DLS2 are
        # SDP-position indexed; they coincide only without LP blocks.
        sdp_l = _sdp_blocks_up_to(l, problem.blockStruct)
        # Invariant: a nonzero location-table entry implies l is an SDP
        # block (LP blocks have SDP_nConstraint[l] == 0, hence count 0),
        # and its SDP position is in range for every SDP-block array.
        # error(), not @assert: the location table is built from the
        # user's problem structure.
        problem.blockStruct[l] >= 2 || error("compute_bmat_sparse_sdp!: problem block $l has a nonzero location table but is not an SDP block (blockStruct[l] = $(problem.blockStruct[l]))")
        1 <= sdp_l <= length(current_pt.xMat.SDP_block) || error("compute_bmat_sparse_sdp!: SDP position $sdp_l of problem block $l out of range 1:$(length(current_pt.xMat.SDP_block))")

        # Iterate sub-matrices for SDP block sdp_l.
        xMat_blk    = current_pt.xMat.SDP_block[sdp_l].de_ele
        invzMat_blk = current_pt.invzMat.SDP_block[sdp_l].de_ele
        work1_blk   = work.DLS1.SDP_block[sdp_l].de_ele
        work2_blk   = work.DLS2.SDP_block[sdp_l].de_ele
        previous_i  = 0

        for k_local in 1:count_l
            k = offset + k_local
            i  = newton_state.SDP_constraint1[k]
            ib = newton_state.blockIndex1[k]
            Ai = problem.A[i].SDP_sp_block[ib]
            formula = newton_state.useFormula[i, l]

            # Cache X · A_i · Z⁻¹ (or A_i · Z⁻¹) when i changes.
            if i != previous_i
                hasF2Gcal[] = false
                if formula === :F1
                    _sparse_dense_mul!(work1_blk, Ai, invzMat_blk)
                    sdpa_mul!(work2_blk, xMat_blk, work1_blk)
                elseif formula === :F2
                    _sparse_dense_mul!(work1_blk, Ai, invzMat_blk)
                end
                # :F3 needs no work matrices — cal_f3! contracts
                # A_i/A_j with X and Z⁻¹ directly.
            end

            j  = newton_state.SDP_constraint2[k]
            jb = newton_state.blockIndex2[k]
            Aj = problem.A[j].SDP_sp_block[jb]

            value = if formula === :F1
                if T <: BigFloat
                    cal_f1!(value_buf, work2_blk, Aj)
                    value_buf
                else
                    cal_f1!(work2_blk, Aj)
                end
            elseif formula === :F2
                if T <: BigFloat
                    cal_f2!(value_buf, v2_buf, work1_blk, work2_blk,
                            xMat_blk, Aj, hasF2Gcal)
                    value_buf
                else
                    cal_f2!(work1_blk, work2_blk, xMat_blk, Aj, hasF2Gcal)
                end
            else  # :F3
                if T <: BigFloat
                    cal_f3!(value_buf, sum_val_buf, tmp_buf,
                            xMat_blk, invzMat_blk, Ai, Aj)
                    value_buf
                else
                    cal_f3!(xMat_blk, invzMat_blk, Ai, Aj)
                end
            end

            loc = newton_state.SDP_location_sparse_bMat[k]
            if T <: BigFloat
                mp_add!(sparse_bmat.sp_ele[loc], sparse_bmat.sp_ele[loc], value)
            else
                sparse_bmat.sp_ele[loc] += value
            end
            previous_i = i
        end
        offset += count_l
    end

    return newton_state
end

"""
    threaded_compute_bmat_sparse_sdp!(newton_state, problem, current_pt, work)

Threaded version of `compute_bmat_sparse_sdp!`.
Parallelizes across SDP blocks using OhMyThreads `@tasks`.

Multiple blocks may contribute to the same `sp_ele[loc]`, so we use
per-block value arrays and a sequential reduction step. The expensive
part (F1/F2/F3 inner products) runs in parallel; the cheap scatter-add
runs sequentially.
"""
function threaded_compute_bmat_sparse_sdp!(
    newton_state::NewtonState{T}, problem::SDPAProblem{T},
    current_pt::SDPASolution{T}, work::SDPAWorkVariables{T},
) where {T<:Real}
    sparse_bmat = newton_state.sparse_bmat
    sparse_bmat === nothing &&
        error("threaded_compute_bmat_sparse_sdp!: sparse_bmat not allocated")

    # Zero the values; keep structure.
    # For BigFloat this MUST be a per-element mp_zero! (NOT
    # fill!(sp_ele, zero(T))): fill! aliases every element to one shared
    # BigFloat, and the sequential reduction below mutates sp_ele[loc]
    # IN PLACE via mp_add! — shared elements would accumulate across
    # locations (wrong values, not bit-identical to the serial path).
    # Same idiom as compute_bmat_sparse_sdp!.
    if T <: BigFloat
        @inbounds for idx in eachindex(sparse_bmat.sp_ele)
            mp_zero!(sparse_bmat.sp_ele[idx])
        end
    else
        fill!(sparse_bmat.sp_ele, zero(T))
    end

    sdp_block_count = length(newton_state.SDP_block_table_len)
    total_entries = sum(newton_state.SDP_block_table_len)

    # Per-entry value buffer. For BigFloat the slots MUST be independent
    # AND valid: the sequential reduction below reads all_values[k] while
    # the compute phase mp_set!s into the same slots, so shared or undef
    # slots would corrupt. The buffer is cached on newton_state and
    # reused across make_bmat! calls: the compute phase overwrites
    # EVERY slot on every call, so no re-zeroing is needed — the cache
    # avoids re-allocating total_entries BigFloats per iteration. Same
    # idiom as _g_perm_scratch/_dy_perm_scratch.
    if length(newton_state._sdp_all_values) != total_entries
        newton_state._sdp_all_values = _independent_zeros(T, total_entries)
    end
    all_values = newton_state._sdp_all_values

    # Compute offsets per block for indexing into all_values.
    offsets = Vector{Int}(undef, sdp_block_count + 1)
    offsets[1] = 0
    for l in 1:sdp_block_count
        offsets[l + 1] = offsets[l] + newton_state.SDP_block_table_len[l]
    end

    # Parallel: each task processes one SDP block. Within a block,
    # the computation is sequential (work-matrix caching depends on
    # previous_i). Different blocks use different work matrices
    # (work.DLS1.SDP_block[sdp_l] etc.) so there is no shared mutable
    # state. Filter out empty blocks (can't use `continue` inside
    # @tasks).
    active_blocks = [l for l in 1:sdp_block_count
                     if newton_state.SDP_block_table_len[l] > 0]

    @tasks for l in active_blocks  # threaded_compute_bmat_sparse_sdp!
        count_l = newton_state.SDP_block_table_len[l]
        offset = offsets[l]

        # Map problem-block l to its SDP-block position — same
        # invariant as the serial path.
        sdp_l = _sdp_blocks_up_to(l, problem.blockStruct)
        # Same invariant as the serial path; error(), not @assert.
        problem.blockStruct[l] >= 2 || error("threaded_compute_bmat_sparse_sdp!: problem block $l has a nonzero location table but is not an SDP block (blockStruct[l] = $(problem.blockStruct[l]))")
        1 <= sdp_l <= length(current_pt.xMat.SDP_block) || error("threaded_compute_bmat_sparse_sdp!: SDP position $sdp_l of problem block $l out of range 1:$(length(current_pt.xMat.SDP_block))")

        xMat_blk    = current_pt.xMat.SDP_block[sdp_l].de_ele
        invzMat_blk = current_pt.invzMat.SDP_block[sdp_l].de_ele
        work1_blk   = work.DLS1.SDP_block[sdp_l].de_ele
        work2_blk   = work.DLS2.SDP_block[sdp_l].de_ele
        hasF2Gcal   = Ref(false)
        previous_i  = 0

        # Per-task BigFloat accumulators (for thread safety).
        if T <: BigFloat
            t_value_buf = zero(BigFloat)
            t_v2_buf = zero(BigFloat)
            t_sum_val_buf = zero(BigFloat)
            t_tmp_buf = zero(BigFloat)
        end

        for k_local in 1:count_l
            k = offset + k_local
            i  = newton_state.SDP_constraint1[k]
            ib = newton_state.blockIndex1[k]
            Ai = problem.A[i].SDP_sp_block[ib]
            formula = newton_state.useFormula[i, l]

            if i != previous_i
                hasF2Gcal[] = false
                if formula === :F1
                    _sparse_dense_mul!(work1_blk, Ai, invzMat_blk)
                    sdpa_mul!(work2_blk, xMat_blk, work1_blk)
                elseif formula === :F2
                    _sparse_dense_mul!(work1_blk, Ai, invzMat_blk)
                end
            end

            j  = newton_state.SDP_constraint2[k]
            jb = newton_state.blockIndex2[k]
            Aj = problem.A[j].SDP_sp_block[jb]

            all_values[k] = if formula === :F1
                if T <: BigFloat
                    cal_f1!(t_value_buf, work2_blk, Aj)
                    # all_values slots are _independent_zeros: each is a
                    # fresh BigFloat, so mp_set! is a value-copy into an
                    # independent element.
                    mp_set!(all_values[k], t_value_buf)
                else
                    cal_f1!(work2_blk, Aj)
                end
            elseif formula === :F2
                if T <: BigFloat
                    cal_f2!(t_value_buf, t_v2_buf, work1_blk, work2_blk,
                            xMat_blk, Aj, hasF2Gcal)
                    mp_set!(all_values[k], t_value_buf)
                else
                    cal_f2!(work1_blk, work2_blk, xMat_blk, Aj, hasF2Gcal)
                end
            else  # :F3
                if T <: BigFloat
                    cal_f3!(t_value_buf, t_sum_val_buf, t_tmp_buf,
                            xMat_blk, invzMat_blk, Ai, Aj)
                    mp_set!(all_values[k], t_value_buf)
                else
                    cal_f3!(xMat_blk, invzMat_blk, Ai, Aj)
                end
            end
            previous_i = i
        end
    end

    # Sequential reduction: accumulate all_values into sp_ele by loc.
    # Multiple entries can map to the same loc (contributions from
    # different blocks); the += is safe because this loop is sequential.
    # Exact-zero values are skipped: `mp_add!(x, x, +0)` is the identity
    # (and sp_ele never holds a signed -0 from the +0-init + additions),
    # so skipping zeros leaves every location bit-identical to the
    # serial path while skipping zero-adds (most entries are
    # structurally zero). Nonzero additions keep their exact serial
    # k-order.
    @inbounds for k in 1:total_entries
        loc = newton_state.SDP_location_sparse_bMat[k]
        if T <: BigFloat
            if !iszero(all_values[k])
                mp_add!(sparse_bmat.sp_ele[loc], sparse_bmat.sp_ele[loc], all_values[k])
            end
        else
            v = all_values[k]
            if !iszero(v)
                sparse_bmat.sp_ele[loc] += v
            end
        end
    end

    return newton_state
end

# ---------------------------------------------------------------------------
# Sparse Schur complement location table.
# ---------------------------------------------------------------------------
# Mirrors C++ Newton::make_aggrigateIndex_SDP at sdpa_newton.cpp:393-490.
# Builds 5 parallel arrays mapping every (i, j, block_l) triple in the
# aggregate sparsity pattern to a position in sparse_bmat.sp_ele.
#
# See docs/sparse-path.md for the location-table design.

"""
    _binary_search_column(sparse_bmat, diagonalIndex, ii, jj) -> Int

Binary-search column `ii` of `sparse_bmat.column_index` for the entry
with column-index value `jj`. Returns the position in sp_ele. Throws
if not found (programmer error — the column structure should always
contain jj by construction).
"""
function _binary_search_column(
    sparse_bmat::SDPASparseMatrix{T}, diagonalIndex::Vector{Int},
    ii::Int, jj::Int,
) where {T<:Real}
    lo = diagonalIndex[ii]
    hi = diagonalIndex[ii + 1] - 1
    while lo <= hi
        mid = (lo + hi) ÷ 2
        v = sparse_bmat.column_index[mid]
        if v < jj
            lo = mid + 1
        elseif v > jj
            hi = mid - 1
        else
            return mid
        end
    end
    error("_binary_search_column: column $jj not found in column $ii (called from make_aggrigate_index_sdp!, _compute_bmat_sparse_lp!, threaded_compute_bmat_sparse_lp!)")
end

# The explicit-coordinate compatibility path must still check actual
# coordinates: a manually supplied matrix can have holes inside its range.
@inline function _schur_assembly_slot(
    mat::SDPASparseMatrix, offsets::Vector{Int}, i::Int, j::Int, context::String,
)
    # Same row-index guard as the compact form: a descriptive error, not
    # a BoundsError from the offset reads below.
    1 <= i <= length(offsets) - 1 ||
        error("$context row index ii=$i outside block-level profile")
    height = offsets[i + 1] - offsets[i]
    i <= j && j - i < height ||
        error("$context pair (ii=$i, jj=$j) outside block-level profile")
    return _binary_search_column(mat, offsets, i, j)
end

"""
    make_aggrigate_index_sdp!(newton_state, problem, ordering)

Build the sparse Schur complement location table for SDP blocks.

Pre-condition: `newton_state.sparse_bmat` contains a closed profile from
the block-level skyline. With an explicit-coordinate test or compatibility
matrix, the caller must also set `newton_state.diagonalIndex`. Symbolic
locations do not depend on the numeric values.

Post-condition: the 5 location-table arrays are populated; the table
covers every (i, j, block_l) triple where both i and j have nonzeros in
block l (with `inz >= jnz` upper-triangle skip).

Mirrors C++ `Newton::make_aggrigateIndex_SDP` at
`sdpa_newton.cpp:393-490`.
"""
function make_aggrigate_index_sdp!(
    newton_state::NewtonState{T}, problem::SDPAProblem{T},
    ordering::AbstractVector{<:Integer},
) where {T<:Real}
    sparse_bmat = newton_state.sparse_bmat
    sparse_bmat === nothing &&
        error("make_aggrigate_index_sdp!: sparse_bmat must be populated first")
    diagonalIndex = _schur_offsets(sparse_bmat, newton_state.diagonalIndex)
    length(diagonalIndex) == problem.m + 1 ||
        error("make_aggrigate_index_sdp!: diagonalIndex length mismatch")
    reverse_ordering = newton_state.reverse_ordering
    length(reverse_ordering) == problem.m ||
        error("make_aggrigate_index_sdp!: reverse_ordering length mismatch")

    # Reset the 5 arrays (in case this is a re-run on the same state).
    empty!(newton_state.SDP_constraint1)
    empty!(newton_state.SDP_constraint2)
    empty!(newton_state.blockIndex1)
    empty!(newton_state.blockIndex2)
    empty!(newton_state.SDP_location_sparse_bMat)
    # SDP_block_table_len is sized to problem.nBlock and PROBLEM-block
    # indexed (SDP blocks get their entry counts, LP blocks stay 0) —
    # SDP_nConstraint/SDP_constraint are likewise problem-block indexed
    # (SDPAProblem.jl constructor sizes them to nBlock). The consumers
    # (compute_bmat_sparse_sdp! / threaded_compute_bmat_sparse_sdp!)
    # map each problem block to its SDP-block position via
    # _sdp_blocks_up_to when indexing xMat.SDP_block etc.
    nblocks = length(problem.SDP_nConstraint)
    resize!(newton_state.SDP_block_table_len, nblocks)
    fill!(newton_state.SDP_block_table_len, 0)

    # Per C++: iterate over SDP blocks. For each block, iterate the
    # (k1, k2) pairs in the upper triangle (skip when inz<jnz or tie-i<j).
    # PRE-SIZE the 5 arrays to the EXACT kept-pair count: exact
    # pre-sizing avoids the 2×-growth reallocations of `push!`
    # (C++ pre-allocates (n+1)*n/2 per block, sdpa_newton.cpp:415).
    for l in 1:nblocks
        nC_l = problem.SDP_nConstraint[l]
        nC_l == 0 && continue
        # Exact kept-pair count for this block: pairs (k1,k2) passing the
        # (inz<jnz) || (inz==jnz && i<j) skip. For uniform nonZeroEffect
        # this is exactly nC_l*(nC_l+1)/2; computed exactly rather than
        # assumed.
        cs_l = problem.SDP_constraint[l]
        kept = 0
        for k1 in 1:nC_l
            i = cs_l[k1]
            ib = problem.SDP_blockIndex[l][k1]
            inz = problem.A[i].SDP_sp_block[ib].nonZeroEffect
            for k2 in 1:nC_l
                j = cs_l[k2]
                jb = problem.SDP_blockIndex[l][k2]
                jnz = problem.A[j].SDP_sp_block[jb].nonZeroEffect
                (inz < jnz) && continue
                (inz == jnz && i < j) && continue
                kept += 1
            end
        end
        n_start = length(newton_state.SDP_constraint1)
        n_end = n_start + kept
        # Exact pre-size (the 5 arrays all grow in lockstep).
        sizehint!(newton_state.SDP_constraint1, n_end)
        sizehint!(newton_state.SDP_constraint2, n_end)
        sizehint!(newton_state.blockIndex1, n_end)
        sizehint!(newton_state.blockIndex2, n_end)
        sizehint!(newton_state.SDP_location_sparse_bMat, n_end)

        count_this_block = 0
        for k1 in 1:nC_l
            i = problem.SDP_constraint[l][k1]
            ib = problem.SDP_blockIndex[l][k1]
            inz = problem.A[i].SDP_sp_block[ib].nonZeroEffect
            for k2 in 1:nC_l
                j = problem.SDP_constraint[l][k2]
                jb = problem.SDP_blockIndex[l][k2]
                jnz = problem.A[j].SDP_sp_block[jb].nonZeroEffect

                # Upper-triangle skip (mirror C++ line 441).
                (inz < jnz) && continue
                (inz == jnz && i < j) && continue

                # Compute permuted indices.
                p_i = reverse_ordering[i]
                p_j = reverse_ordering[j]
                ii, jj = p_i <= p_j ? (p_i, p_j) : (p_j, p_i)

                t = _schur_assembly_slot(sparse_bmat, diagonalIndex, ii, jj, "block-level")

                push!(newton_state.SDP_constraint1, i)
                push!(newton_state.SDP_constraint2, j)
                push!(newton_state.blockIndex1, ib)
                push!(newton_state.blockIndex2, jb)
                push!(newton_state.SDP_location_sparse_bMat, t)
                count_this_block += 1
            end
        end
        newton_state.SDP_block_table_len[l] = count_this_block
    end

    return newton_state
end
