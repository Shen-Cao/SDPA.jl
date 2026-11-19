"""
    SDPAResiduals{T<:Real}

Primal and dual residuals at the current iterate. Mirrors C++
`sdpa::Residuals` at `sdpa_dataset.h:124-145` and
`compute` at `sdpa_dataset.cpp:625-646`.

The residual formulas are:
  - `primalVec = b - ⟨A_i, xMat⟩_i`  (length m)
  - `dualMat   = C - Σ_i yVec[i] · A_i - zMat`
  - `normPrimalVec = max(|primalVec[i]|)`         (max-abs norm,
    matching C++ `computeMaxNorm` at `sdpa_dataset.cpp:570-585`)
  - `normDualMat   = max(|dualMat.SDP_block[k][i,j]|, |dualMat.LP_block[k]|)`
    (max-abs norm over every entry of every block)
  - `centerNorm` = 0 (computed elsewhere — set by `compute!` to 0 for
    consistency with C++; the actual centering measure is the
    complementarity `μ = ⟨X,Z⟩/nDim` which lives in
    `AverageComplementarity`)

**IMPORTANT**: the norm is MAX-ABS, not the 2-norm. Matching C++ is
load-bearing: the convergence thresholds (`epsilonStar`, `epsilonDash`)
are calibrated to max-abs in the SDPA paper. Using the 2-norm would
change the effective tolerance by a factor of `sqrt(n)`.
"""
mutable struct SDPAResiduals{T<:Real}
    primalVec::SDPAVector{T}
    dualMat::SDPADenseLinearSpace{T}
    normPrimalVec::T
    normDualMat::T
    centerNorm::T
end

function SDPAResiduals{T}(
    problem::SDPAProblem{T}, ::SDPASolution{T}
) where {T<:Real}
    primalVec = SDPAVector{T}(problem.m)
    dualMat = SDPADenseLinearSpace{T}(problem.blockStruct)
    return SDPAResiduals{T}(
        primalVec, dualMat, zero(T), zero(T), zero(T)
    )
end

"""
    _max_abs(v)

Max-abs of a `Vector{T}`. Returns `zero(T)` for an empty vector.
Mirrors C++ `Residuals::computeMaxNorm(Vector)` at
`sdpa_dataset.cpp:570-585` (uses the explicit loop, not `gmp_idamax`,
to keep the result deterministic across precisions and thread counts).
"""
function _max_abs(v::AbstractVector{T}) where {T<:Real}
    ret = zero(T)
    @inbounds for k in eachindex(v)
        a = abs(v[k])
        if a > ret
            ret = a
        end
    end
    return ret
end

"""
    _max_abs_dls(dls)

Max-abs over every entry of every SDP block and every LP entry.
Mirrors C++ `Residuals::computeMaxNorm(DenseLinearSpace)` at
`sdpa_dataset.cpp:586-620`.
"""
function _max_abs_dls(dls::SDPADenseLinearSpace{T}) where {T<:Real}
    ret = zero(T)
    for k in 1:dls.SDP_nBlock
        if isassigned(dls.SDP_block, k)
            m = dls.SDP_block[k]
            @inbounds for idx in eachindex(m.de_ele)
                a = abs(m.de_ele[idx])
                if a > ret
                    ret = a
                end
            end
        end
    end
    @inbounds for k in eachindex(dls.LP_block)
        a = abs(dls.LP_block[k])
        if a > ret
            ret = a
        end
    end
    return ret
end

"""
    compute!(residuals, problem, current_pt)

Recompute the primal and dual residuals in place. Mirrors C++
`Residuals::compute` at `sdpa_dataset.cpp:625-646`.

Formula:
  - primalVec = b - ⟨A_i, xMat⟩_i   (computed as -A·xMat + b for sign)
  - dualMat   = C - Σ_i yVec[i] · A_i - zMat
  - norms via `_max_abs`

`centerNorm` is set to 0 here; the actual complementarity measure is
maintained separately by `AverageComplementarity`.
"""
function compute!(
    residuals::SDPAResiduals{T},
    problem::SDPAProblem{T},
    current_pt::SDPASolution{T},
) where {T<:Real}
    # primalVec = -<A, xMat> + b
    multi_inner_product_to_a!(residuals.primalVec, problem, current_pt.xMat)
    @inbounds for i in 1:problem.m
        residuals.primalVec.ele[i] = problem.b.ele[i] - residuals.primalVec.ele[i]
    end

    # dualMat = C - sum_i yVec[i] * A_i - zMat
    multi_plus_to_a!(residuals.dualMat, problem, current_pt.yVec)
    # Negate: dualMat = -sum_i y_i * A_i (the C++ does this via
    # Lal::let(dualMat,'=',dualMat,'*",&MMONE) before adding C).
    for k in 1:residuals.dualMat.SDP_nBlock
        if isassigned(residuals.dualMat.SDP_block, k)
            blk = residuals.dualMat.SDP_block[k].de_ele
            @inbounds @simd for idx in eachindex(blk)
                blk[idx] = -blk[idx]
            end
        end
    end
    @inbounds for k in eachindex(residuals.dualMat.LP_block)
        residuals.dualMat.LP_block[k] = -residuals.dualMat.LP_block[k]
    end
    # Add C and subtract zMat block-by-block. C's SDP part is sparse and may
    # have UNASSIGNED blocks (a zero-objective block that never received an
    # entry), so the C loop skips them — the zMat subtraction therefore must
    # NOT be nested inside it (see the zMat loops below).
    for sdp_k in 1:problem.C.SDP_sp_nBlock
        isassigned(problem.C.SDP_sp_block, sdp_k) || continue
        global_k = problem.C.SDP_sp_index[sdp_k]
        C_block = problem.C.SDP_sp_block[sdp_k]
        d_block = _lookup_sdp_block(residuals.dualMat, global_k, problem.blockStruct)
        # dualMat += C_block contribution (sign: -A·y + C)
        if C_block.type === :sparse
            for nz in 1:C_block.nonZeroEffect
                r = C_block.row_index[nz]
                c = C_block.column_index[nz]
                v = C_block.sp_ele[nz]
                d_block.de_ele[r, c] += v
                if r != c
                    d_block.de_ele[c, r] += v
                end
            end
        else
            @inbounds for k in eachindex(C_block.de_ele)
                d_block.de_ele[k] += C_block.de_ele[k]
            end
        end
    end
    # dualMat -= zMat for EVERY SDP block, independent of the C loop
    # above: C's SDP part may have unassigned blocks (e.g. a
    # zero-objective block in a manually built problem), and with the
    # subtraction nested in the C loop such blocks would keep a dual
    # residual of -ΣyA instead of C-ΣyA-Z = -ΣyA-Z; at the default start
    # point (Z = λI) normDualMat would silently omit λ, corrupting the
    # phase classification from iteration 0.
    for l in 1:length(problem.blockStruct)
        problem.blockStruct[l] >= 2 || continue
        z_block = _lookup_sdp_block(current_pt.zMat, l, problem.blockStruct)
        d_block = _lookup_sdp_block(residuals.dualMat, l, problem.blockStruct)
        @inbounds for k in eachindex(d_block.de_ele)
            d_block.de_ele[k] -= z_block.de_ele[k]
        end
    end
    # LP blocks: add C's compact nonzero list, then subtract zMat for
    # EVERY LP scalar. The zMat subtraction must NOT be nested inside the
    # C-nonzero loop: it applies to all LP scalars, including those where
    # C has no entry (C typically touches only a small fraction of LP
    # scalars — nesting would leave the untouched dual residuals at 0
    # instead of -zMat, corrupting the Schur RHS gvec and diverging the
    # trajectory).
    for lp_pos in 1:problem.C.LP_nnz
        lp_k = problem.C.LP_sp_index[lp_pos]
        residuals.dualMat.LP_block[lp_k] += problem.C.LP_sp_val[lp_pos]
    end
    @inbounds for lp_k in eachindex(residuals.dualMat.LP_block)
        residuals.dualMat.LP_block[lp_k] -= current_pt.zMat.LP_block[lp_k]
    end

    residuals.normPrimalVec = _max_abs(residuals.primalVec.ele)
    residuals.normDualMat = _max_abs_dls(residuals.dualMat)
    residuals.centerNorm = zero(T)
    return residuals
end
