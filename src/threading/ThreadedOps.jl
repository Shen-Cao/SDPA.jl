# Threaded versions of the hot-loop operations.
#
# Uses OhMyThreads.@tasks with per-task buffers (invariant:
# no thread-id queries for buffer indexing).
#
# Correctness invariant: each output element is written by exactly one
# task iteration. No reductions needed (embarrassingly parallel).

"""
    threaded_compute_bmat_dense!(newton_state, problem, current_pt, work)

Threaded version of `compute_bmat_dense!`. Parallelizes the outer
constraint loop (k1) across threads. Each thread gets its own work1/work2
matrices via a per-task pool.

Bit-identical to the serial version because each B[i,j] element is
computed by exactly one (k1, k2) pair per block (enforced by the skip
condition), and cross-block contributions to the same element accumulate
in serial block order (see the LOAD-BEARING comment below).
"""
function threaded_compute_bmat_dense!(
    newton_state::NewtonState{T}, problem::SDPAProblem{T},
    current_pt::SDPASolution{T}, work::SDPAWorkVariables{T},
) where {T<:Real}
    m = problem.m
    # Dense-path-only entry point (see compute_bmat_dense!): materialize
    # the lazily-allocated bmat (no-op when already present).
    bmat = ensure_dense_bmat!(newton_state).de_ele
    # Zero bmat in-place (not _fill_independent! which allocates m² new BigFloats).
    if T <: BigFloat
        @inbounds for idx in eachindex(bmat)
            mp_zero!(bmat[idx])
        end
    else
        fill!(bmat, zero(T))
    end

    # LOAD-BEARING: per-block `@tasks` regions inside a SERIAL `for l`
    # loop over SDP blocks. Constraints spanning multiple blocks (the
    # block_sparse / mixed-problem regime) contribute to the SAME
    # bmat[i,j] from DIFFERENT blocks; those contributions must
    # accumulate in serial block order (bit-stability + no cross-task
    # writes to one MPFR cell). Do NOT flatten into one region:
    # cross-block contributions to one bmat element would race and
    # reorder accumulation. The alternative (one region + TLV keyed by
    # (task, bs)) is a possible future design; see docs/roadmap.md.
    for l in 1:problem.nBlock
        problem.blockStruct[l] >= 2 || continue
        sdp_l = _sdp_blocks_up_to(l, problem.blockStruct)
        nconstr_l = problem.SDP_nConstraint[l]
        nconstr_l == 0 && continue

        xMat_blk = current_pt.xMat.SDP_block[sdp_l].de_ele
        invZ_blk = current_pt.invzMat.SDP_block[sdp_l].de_ele
        bs = problem.blockStruct[l]

        @tasks for k1 in 1:nconstr_l
            # Per-task workspace (OhMyThreads @tasks + @local). The
            # pool's work1/work2 (and the BigFloat scalar accumulators) are
            # acquired ONCE per task via a single typed @local TLV
            # (OhMyThreads TaskLocalValue: the initializer closure runs once
            # per task on first access and is cached in that task's local
            # storage; TLVs cannot reference each other, so one TLV carries
            # the whole tuple) and reused across ALL k1 iterations of the
            # task's chunk. Reuse is bit-identical because every read of a
            # scratch buffer follows a full overwrite within the same
            # iteration: _sparse_dense_mul! zeroes work1 in-place;
            # sdpa_mul!/mul! fully overwrite work2 (beta=0); cal_f1!/cal_f2!/
            # cal_f3! zero their scalar accumulators on entry (mp_zero!).
            # No cross-task sharing: each task touches only its own TLV, so
            # no locking is needed.
            @local ws::Tuple{BigFloatPool{T}, Matrix{T}, Matrix{T},
                             BigFloat, BigFloat, BigFloat, BigFloat, Ref{Bool}} = let
                pool = BigFloatPool{T}(;
                    precision=precision(BigFloat), max_mats=2, max_vecs=0,
                    mat_size=(bs, bs), vec_size=0,
                )
                (pool,
                 acquire_mat!(pool, bs, bs), acquire_mat!(pool, bs, bs),
                 zero(BigFloat), zero(BigFloat), zero(BigFloat), zero(BigFloat),
                 Ref(false))
            end
            work1 = ws[2]
            work2 = ws[3]

            hasF2Gcal = ws[8]
            hasF2Gcal[] = false

            # Per-task BigFloat accumulators (zero-alloc path).
            # For Float64 these are unused (existing path is already alloc-free).
            # NOTE: `BigFloat()` returns NaN on Julia 1.12 — must use
            # `zero(BigFloat)` for proper zero initialization.
            if T <: BigFloat
                value_buf = ws[4]
                v2_buf = ws[5]
                sum_val_buf = ws[6]
                tmp_buf = ws[7]
            end

            i = problem.SDP_constraint[l][k1]
            ib = problem.SDP_blockIndex[l][k1]
            Ai = problem.A[i].SDP_sp_block[ib]
            inz = Ai.nonZeroEffect
            formula = newton_state.useFormula[i, l]

            if formula === :F1
                _sparse_dense_mul!(work1, Ai, invZ_blk)
                sdpa_mul!(work2, xMat_blk, work1)
            elseif formula === :F2
                _sparse_dense_mul!(work1, Ai, invZ_blk)
            end

            for k2 in 1:nconstr_l
                j = problem.SDP_constraint[l][k2]
                jb = problem.SDP_blockIndex[l][k2]
                Aj = problem.A[j].SDP_sp_block[jb]
                jnz = Aj.nonZeroEffect

                (inz < jnz) && continue
                (inz == jnz && i < j) && continue

                if T <: BigFloat
                    # In-place path (see SchurComplement.jl for full notes):
                    # `bmat[i,j]` getindex returns the MUTABLE BigFloat
                    # reference (not a copy — BigFloat getindex returns the
                    # mutable element), so
                    # `mp_add!(bmat[i,j], bmat[i,j], value_buf)` mutates the
                    # array element in place with zero allocation.
                    if formula === :F1
                        cal_f1!(value_buf, work2, Aj)
                    elseif formula === :F2
                        cal_f2!(value_buf, v2_buf, work1, work2,
                                xMat_blk, Aj, hasF2Gcal)
                    else
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
                    # Float64 path (unchanged).
                    value = zero(T)
                    if formula === :F1
                        value = cal_f1!(work2, Aj)
                    elseif formula === :F2
                        value = cal_f2!(work1, work2, xMat_blk, Aj, hasF2Gcal)
                    else
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

    _compute_bmat_lp!(bmat, newton_state, problem, current_pt)
    return newton_state
end

"""
    threaded_get_inv_chol!(inv_chol_mat, a_mat, work_mat) -> Bool

Threaded version of `get_inv_chol!`. Parallelizes Cholesky across SDP
blocks. Each block's Cholesky is independent.
"""
function threaded_get_inv_chol!(
    inv_chol_mat::SDPADenseLinearSpace{T},
    a_mat::SDPADenseLinearSpace{T},
    work_mat::SDPADenseLinearSpace{T},
) where {T<:Real}
    inv_chol_mat.SDP_nBlock == a_mat.SDP_nBlock ||
        error("threaded_get_inv_chol!: SDP_nBlock mismatch")
    inv_chol_mat.LP_nBlock == a_mat.LP_nBlock ||
        error("threaded_get_inv_chol!: LP_nBlock mismatch")

    ok = fill(true, a_mat.SDP_nBlock)
    @tasks for k in 1:a_mat.SDP_nBlock
        # @tasks doesn't support continue; gate the body with if/else.
        # Per-task zero-alloc accumulator, same idiom as the serial
        # get_inv_chol! (JordanAlgebra.jl:280-285). Assigned only inside
        # this @tasks body, so it is a genuine per-call local.
        local s_buf::T
        has_s_buf = false
        if T <: BigFloat
            s_buf = zero(BigFloat)
            has_s_buf = true
        end
        if isassigned(a_mat.SDP_block, k)
            src_blk = a_mat.SDP_block[k].de_ele
            n = size(src_blk, 1)
            work_blk = work_mat.SDP_block[k].de_ele
            _copyto_independent!(work_blk, src_blk)
            F = cholesky!(Hermitian(work_blk, :U); check=false)
            if issuccess(F)
                L = LowerTriangular(F.factors')
                out_blk = inv_chol_mat.SDP_block[k].de_ele
                # Zero out_blk in-place
                if T <: BigFloat
                    @inbounds for idx in eachindex(out_blk)
                        mp_zero!(out_blk[idx])
                    end
                else
                    fill!(out_blk, zero(T))
                end
                @inbounds for j in 1:n
                    out_blk[j, j] = one(T) / L[j, j]
                    for i in j+1:n
                        if has_s_buf
                            # BigFloat in-place path: MUST match the serial
                            # get_inv_chol! term-by-term. mp_fma! rounds
                            # once per term; `s += a*b` rounds twice, so
                            # the threaded result would differ from serial
                            # by ulps for multi-term sums (block >= 3) and
                            # break cross-thread-count bit-identity.
                            mp_zero!(s_buf)
                            for kk in j:i-1
                                mp_fma!(s_buf, L[i, kk], out_blk[kk, j], s_buf)
                            end
                            mp_div!(s_buf, s_buf, L[i, i])
                            mp_neg!(s_buf, s_buf)
                            # deepcopy (NOT BigFloat(s_buf)): breaks the
                            # BigFloat(buf) aliasing on Julia 1.12.
                            out_blk[i, j] = deepcopy(s_buf)
                        else
                            s = zero(T)
                            for kk in j:i-1
                                s += L[i, kk] * out_blk[kk, j]
                            end
                            out_blk[i, j] = -s / L[i, i]
                        end
                    end
                end
            else
                ok[k] = false
            end
        end
    end
    all(ok) || (return false)

    # LP blocks: MUST match the serial get_inv_chol! (JordanAlgebra.jl)
    # term-by-term. For BigFloat the serial path uses mp_rec_sqrt!
    # (single rounding); `one(T) / sqrt(x)` rounds twice and would
    # differ by ulps, breaking cross-thread-count bit-identity (the
    # LP entries feed the step-length min-eigen, which binds for
    # LP-heavy problems like arch0).
    lp_buf = zero(BigFloat)
    @inbounds for k in eachindex(a_mat.LP_block)
        if a_mat.LP_block[k] <= 0
            return false
        end
        if T <: BigFloat
            mp_rec_sqrt!(lp_buf, a_mat.LP_block[k])
            inv_chol_mat.LP_block[k] = deepcopy(lp_buf)
        else
            inv_chol_mat.LP_block[k] = one(T) / sqrt(a_mat.LP_block[k])
        end
    end
    return true
end

"""
    threaded_get_inv_chol_and_inv!(inv_chol_mat, inverse_mat, a_mat, work_mat) -> Bool

Threaded version of `get_inv_chol_and_inv!`. Parallelizes across SDP blocks.
"""
function threaded_get_inv_chol_and_inv!(
    inv_chol_mat::SDPADenseLinearSpace{T},
    inverse_mat::SDPADenseLinearSpace{T},
    a_mat::SDPADenseLinearSpace{T},
    work_mat::SDPADenseLinearSpace{T},
) where {T<:Real}
    ok = threaded_get_inv_chol!(inv_chol_mat, a_mat, work_mat)
    if !ok
        return false
    end
    @tasks for k in 1:a_mat.SDP_nBlock
        # Per-task symmetrization scratch — ONE TLV carrying the avg
        # buffer plus a pre-allocated exact `two`, created once per task
        # (not per element). Same rounding chain as the serial
        # get_inv_chol_and_inv! (JordanAlgebra.jl): mp_add! = the single
        # `+` rounding; mp_div! by 2 is exact in binary FP (pure exponent
        # decrement — same as Base's `x / 2`), so cross-thread-count
        # bit-identity is preserved by construction.
        @local sym_scratch::Tuple{BigFloat, BigFloat} =
            (zero(BigFloat), BigFloat(2))
        # @tasks doesn't support continue; gate with if.
        if isassigned(a_mat.SDP_block, k)
            L_inv = inv_chol_mat.SDP_block[k].de_ele
            inv_blk = inverse_mat.SDP_block[k].de_ele
            n = size(L_inv, 1)
            # Direct Rgemm! product on the raw matrix: inv_blk = (L⁻¹)ᵀ·L⁻¹.
            # The direct call is op-for-op identical to the former manual
            # kernel in both the serial and @tasks branches of Rgemm!, so
            # the bit-pattern is preserved by construction. Inside Rgemm!,
            # distinct per-task accumulator names prevent @tasks closure
            # capture (shared-accumulator race), so the nested @tasks
            # region is safe.
            Rgemm!("T", "N", n, n, n, one(T), L_inv, L_inv, zero(T), inv_blk)
            if T <: BigFloat
                # Value-copy: `inv_blk[i, j] = avg` would store the
                # SAME BigFloat object in both transposed slots, so a
                # later in-place MPFR op on one would corrupt the
                # mirrored entry. Float64 is value-semantic.
                # Scratch-based chain, bit-identical to
                # `avg = (a + b) / 2` (one `+` rounding + exact halving).
                avg_buf = sym_scratch[1]
                two = sym_scratch[2]
                @inbounds for j in 1:n, i in 1:j
                    mp_add!(avg_buf, inv_blk[i, j], inv_blk[j, i])
                    mp_div!(avg_buf, avg_buf, two)
                    mp_set!(inv_blk[i, j], avg_buf)
                    mp_set!(inv_blk[j, i], avg_buf)
                end
            else
                @inbounds for j in 1:n, i in 1:j
                    avg = (inv_blk[i, j] + inv_blk[j, i]) / 2
                    inv_blk[i, j] = avg
                    inv_blk[j, i] = avg
                end
            end
        end
    end
    @inbounds for k in eachindex(a_mat.LP_block)
        # Guard MUST match the serial get_inv_chol_and_inv!
        # (JordanAlgebra.jl): an LP entry <= 0 returns false (loud failure
        # over a silent Inf).
        if a_mat.LP_block[k] <= 0
            return false
        end
        inverse_mat.LP_block[k] = one(T) / a_mat.LP_block[k]
    end
    return true
end

"""
    threaded_min_eigen_dls(l_mat, x_mat, work) -> T

Threaded version of `_min_eigen_dls`. Parallelizes eigenvalue computation
across SDP blocks. Returns the minimum across all blocks.
"""
function threaded_min_eigen_dls(
    l_mat::SDPADenseLinearSpace{T}, x_mat::SDPADenseLinearSpace{T},
    work::SDPAWorkVariables{T},
) where {T<:Real}
    nblocks = x_mat.SDP_nBlock
    # Sentinel T(1e50), C++ parity — see _min_eigen_dls for the
    # Float32→Inf note; max(nblocks, 1) only avoids minimum(∅).
    vals = fill(T(1e50), max(nblocks, 1))
    @tasks for k in 1:nblocks
        # Per-task scratch pool: block sizes vary, so each task
        # acquires/releases its scratch out-buffers per block; equal-sized blocks within
        # a task's chunk reuse the same buffers (free-list hit). The pool is
        # a per-task @local TLV — never shared across tasks — so acquire/
        # release need no locking.
        @local pool::BigFloatPool{T} = BigFloatPool{T}(;
            precision=precision(BigFloat), max_mats=2, max_vecs=0,
            mat_size=(0, 0), vec_size=0,
        )
        if isassigned(x_mat.SDP_block, k) && isassigned(l_mat.SDP_block, k)
            L = l_mat.SDP_block[k].de_ele
            X = x_mat.SDP_block[k].de_ele
            n = size(X, 1)
            if n > 20
                vals[k] = min_eigen_lanczos(L, X)
            else
                # Per-task scratch buffers (OhMyThreads @local pool).
                # Exception safety: min_eigen_qr (GenericLinearAlgebra
                # eigen for BigFloat) can throw on numerical failure.
                # Nested try/finally releases BOTH mats on every
                # path: if the SECOND acquire (for `out`) throws, the outer
                # finally releases `tmp` before rethrowing; if the kernels
                # or min_eigen_qr throw, the inner finally releases `out`
                # and the outer releases `tmp`.
                tmp = acquire_mat!(pool, n, n)
                try
                    out = acquire_mat!(pool, n, n)
                    try
                        # Direct Rgemm! products on the raw matrices:
                        # tmp = X * L', out = L * tmp = L * X * L'.
                        # The direct calls are op-for-op identical to the
                        # former manual kernels in both the serial and
                        # @tasks branches of Rgemm!, so the bit-pattern is
                        # preserved by construction. Inside Rgemm!, distinct
                        # per-task accumulator names prevent @tasks closure
                        # capture (shared-accumulator race), so the nested
                        # @tasks region is safe.
                        Rgemm!("N", "T", n, n, n, one(T), X, L, zero(T), tmp)
                        Rgemm!("N", "N", n, n, n, one(T), L, tmp, zero(T), out)
                        vals[k] = min_eigen_qr(out)
                    finally
                        release_mat!(pool, out)
                    end
                finally
                    release_mat!(pool, tmp)
                end
            end
        end
    end
    min_val = minimum(vals)
    @inbounds for k in 1:x_mat.LP_nBlock
        if k <= length(l_mat.LP_block) && k <= length(x_mat.LP_block)
            val = l_mat.LP_block[k]^2 * x_mat.LP_block[k]
            # LP twin of the serial sticky-NaN guard in
            # _min_eigen_dls (StepLength.jl) — a NaN LP entry must
            # PROPAGATE to the loud guard in compute_step_length!, not be
            # dropped by the comparison. Strict no-op for finite val.
            if isnan(val) || val < min_val
                min_val = val
            end
        end
    end
    return min_val
end
