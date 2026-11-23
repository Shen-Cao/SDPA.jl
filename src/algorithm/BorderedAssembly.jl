# A streamed counterpart to the explicit sparse location table. Use the
# exact same SDP formula and pair ownership rule, then the same LP-scalar
# order. No per-pair index or value buffer is built.
function _assemble_bordered!(
    schur::_BorderedSchur{T}, newton::NewtonState{T}, problem::SDPAProblem{T},
    pt::SDPASolution{T}, work::SDPAWorkVariables{T},
) where {T}
    F, inc = schur.factors, schur.incidence
    _reset_bordered!(F)
    v = zero(T)
    if T === BigFloat
        v2 = zero(T)
        sumval = zero(T)
        tmp = zero(T)
    end
    has_cache = Ref(false)
    sdp_l = 0
    for (block, entries) in enumerate(inc.sdp)
        problem.blockStruct[block] >= 2 || continue
        sdp_l += 1
        isempty(entries) && continue
        X = pt.xMat.SDP_block[sdp_l].de_ele
        Zi = pt.invzMat.SDP_block[sdp_l].de_ele
        W1 = work.DLS1.SDP_block[sdp_l].de_ele
        W2 = work.DLS2.SDP_block[sdp_l].de_ele
        for e in entries
            i = e.row
            Ai = problem.A[i].SDP_sp_block[e.block_index]
            formula = newton.useFormula[i, block]
            has_cache[] = false
            if formula === :F1
                _sparse_dense_mul!(W1, Ai, Zi)
                sdpa_mul!(W2, X, W1)
            elseif formula === :F2
                _sparse_dense_mul!(W1, Ai, Zi)
            end
            for f in entries
                j = f.row
                Aj = problem.A[j].SDP_sp_block[f.block_index]
                wi, wj = Ai.nonZeroEffect, Aj.nonZeroEffect
                (wi < wj || (wi == wj && i < j)) && continue
                value = if T === BigFloat
                    if formula === :F1
                        cal_f1!(v, W2, Aj)
                    elseif formula === :F2
                        cal_f2!(v, v2, W1, W2, X, Aj, has_cache)
                    else
                        cal_f3!(v, sumval, tmp, X, Zi, Ai, Aj)
                    end
                    v
                elseif formula === :F1
                    cal_f1!(W2, Aj)
                elseif formula === :F2
                    cal_f2!(W1, W2, X, Aj, has_cache)
                else
                    cal_f3!(X, Zi, Ai, Aj)
                end
                _bordered_add!(F, i, j, value)
            end
        end
    end
    for (k, entries) in enumerate(inc.lp)
        isempty(entries) && continue
        scale = if T === BigFloat
            mp_mul!(v2, pt.xMat.LP_block[k], pt.invzMat.LP_block[k])
        else
            pt.xMat.LP_block[k] * pt.invzMat.LP_block[k]
        end
        for (a, e) in enumerate(entries)
            x = problem.A[e.row].LP_sp_val[e.coeff_index]
            for b in a:length(entries)
                f = entries[b]
                y = problem.A[f.row].LP_sp_val[f.coeff_index]
                value = if T === BigFloat
                    mp_mul!(v, scale, x)
                    mp_mul!(v, v, y)
                else
                    scale * x * y
                end
                _bordered_add!(F, e.row, f.row, value)
            end
        end
    end
    F.state = _BORDERED_ASSEMBLED
    return nothing
end
