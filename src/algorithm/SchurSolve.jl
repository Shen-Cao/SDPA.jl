"""
Schur solve: Cholesky factorization of B, then Δy/Δz/Δx recovery.

Mirrors C++ `Newton::compute_DyVec` at `sdpa_newton.cpp:1308-1344`,
`Newton::compute_DzMat` at `:1346-1356`, and `Newton::compute_DxMat`
at `:1358-1373`.

After `make_bmat!` produces B and `make_gvec!` produces g, this module
solves the Schur system `B · Δy = g` and recovers the full Newton
direction (Δy, Δz, Δx) via:

  Δy = B⁻¹ · g              (triangular solve from cached Cholesky)
  Δz = -Σ_i Δy_i · A_i + dual_residual   (if dual-infeasible)
     = -Σ_i Δy_i · A_i                   (if dual-feasible)
  Δx = R·Z⁻¹ - X·Δz·Z⁻¹   (then symmetrize to guarantee Δx = Δx')
"""

"""
    cholesky_schur!(newton_state) -> Bool

Cholesky-factorize the Schur matrix in-place.  Uses the dense path
(`cholesky!`) by default; when `newton_state.use_sparse` is true,
uses `sparse_cholesky!` on the profile-form `sparse_bmat` instead.

Returns `true` on success, `false` if the matrix is not positive-
definite (triggers main-loop break in the caller).
Mirrors C++ `Lal::choleskyFactorWithAdjust` at `sdpa_linear.cpp:608-681`.
"""
function cholesky_schur!(newton_state::NewtonState{T}) where {T<:Real}
    if newton_state.use_sparse
        sparse_bmat = newton_state.sparse_bmat
        sparse_bmat === nothing && error("cholesky_schur!: sparse_bmat is nothing but use_sparse=true")
        ok = sparse_cholesky!(sparse_bmat, newton_state.diagonalIndex)
        # The sparse Cholesky stores the factorisation in-place in
        # sparse_bmat; no separate cache object is needed.
        newton_state.cholesky_factorization = nothing
        return ok
    else
        # Dense branch (only when `!use_sparse`): materialize the
        # lazily-allocated bmat (no-op when already present).
        bmat = ensure_dense_bmat!(newton_state).de_ele
        if T <: BigFloat
            # Use blocked Cholesky (Rpotrf!) for large matrices, unblocked
            # (Rpotf2!) for small ones. Rpotrf! delegates to Rpotf2! when
            # n <= nb (block size = 64). Use Rpotrf_with_adjust! for
            # diagonal pivot adjustment fallback (ported from C++
            # Lal::choleskyFactorWithAdjust).
            info = Rpotrf_with_adjust!("U", size(bmat, 1), bmat; nb=64)
            if info != 0
                newton_state.cholesky_factorization = nothing
                return false
            end
            # Construct a Cholesky object wrapping the U factor.
            newton_state.cholesky_factorization = Cholesky(bmat, :U, 0)
            return true
        else
            H = Hermitian(bmat, :U)
            F = cholesky!(H; check=false)
            if !issuccess(F)
                newton_state.cholesky_factorization = nothing
                return false
            end
            newton_state.cholesky_factorization = F
            return true
        end
    end
end

"""
    compute_dyvec!(newton_state, is_predictor) -> Bool

Solve `B · Δy = g` using the cached factorization.

Dense path: uses Julia's `ldiv!` with the cached `Cholesky` object.
Sparse path: uses `solve_systems_sparse!` on the in-place factorized
`sparse_bmat`.

Returns `false` on factorization failure (predictor only).
Mirrors C++ `Newton::compute_DyVec` at `sdpa_newton.cpp:1308-1344`.
"""
function compute_dyvec!(
    newton_state::NewtonState{T}, is_predictor::Bool
) where {T<:Real}
    if is_predictor
        ok = cholesky_schur!(newton_state)
        if !ok
            return false
        end
    end

    if newton_state.use_sparse
        sparse_bmat = newton_state.sparse_bmat
        sparse_bmat === nothing && error("compute_dyvec!: sparse_bmat is nothing")
        # Permute gvec before solve, reverse-permute DyVec after.
        # The bmat was assembled/factorized in permuted order (AMD ordering).
        # Without this, non-identity orderings produce silently wrong Δy.
        m = length(newton_state.gvec.ele)
        # Reuse the per-problem scratch buffers (sized to m in NewtonState)
        # instead of re-allocating two length-m vectors on every call.
        g_perm = newton_state._g_perm_scratch
        # x_perm is the in-place OUTPUT of solve_systems_sparse! (which now
        # writes via mp_set! for BigFloat). It MUST have independent,
        # initialized elements — undef BigFloat elements throw UndefRefError
        # on mp_set!, and shared (aliased) elements would corrupt each other.
        # The scratch starts #undef (Vector{T}(undef, m)); initialize each
        # BigFloat slot ONCE to an independent zero — later calls only
        # mp_set! in place, so no re-allocation is needed (same idiom as the
        # _sdp_all_values cache). Float64 needs no init: the first statement
        # of solve_systems_sparse! overwrites every slot via copyto!.
        x_perm = newton_state._dy_perm_scratch
        if T <: BigFloat
            @inbounds for i in 1:m
                if !isassigned(x_perm, i)
                    x_perm[i] = BigFloat(0)
                end
            end
        end
        permute_vec!(g_perm, newton_state.gvec.ele, newton_state.ordering)
        solve_systems_sparse!(
            x_perm,
            sparse_bmat,
            g_perm,
            newton_state.diagonalIndex,
        )
        # Un-permute using reverse_ordering (reverse_ordering is READ
        # here, not just written).
        @inbounds for i in eachindex(newton_state.DyVec.ele)
            newton_state.DyVec.ele[i] = x_perm[newton_state.reverse_ordering[i]]
        end
    else
        F = newton_state.cholesky_factorization
        F === nothing && error("compute_dyvec!: no cached Cholesky factorization")
        copyto!(newton_state.DyVec.ele, newton_state.gvec.ele)
        ldiv!(F, newton_state.DyVec.ele)
    end
    return true
end

"""
    compute_dzmat!(newton_state, problem, current_res, phase)

Recover Δz from Δy:
  - DzMat = -Σ_i Δy_i · A_i              (always)
  - If dual-infeasible (pFEAS = primal feasible / dual infeasible,
    or noINFO): DzMat += dual_residual

The dual_residual term is the "C - Σ y·A - Z" residual; adding it
when the dual is infeasible accounts for the Newton correction
needed to restore dual feasibility.

Mirrors C++ `Newton::compute_DzMat` at `sdpa_newton.cpp:1346-1356`.
"""
function compute_dzmat!(
    newton_state::NewtonState{T}, problem::SDPAProblem{T},
    current_res::SDPAResiduals{T}, phase::PhaseValue,
) where {T<:Real}
    # DzMat = Σ_i Δy_i · A_i (via multi_plus_to_a!)
    multi_plus_to_a!(newton_state.DzMat, problem, newton_state.DyVec)
    # Negate: DzMat = -Σ_i Δy_i · A_i
    for k in 1:newton_state.DzMat.SDP_nBlock
        if isassigned(newton_state.DzMat.SDP_block, k)
            blk = newton_state.DzMat.SDP_block[k].de_ele
            @inbounds @simd for idx in eachindex(blk)
                blk[idx] = -blk[idx]
            end
        end
    end
    @inbounds for k in eachindex(newton_state.DzMat.LP_block)
        newton_state.DzMat.LP_block[k] = -newton_state.DzMat.LP_block[k]
    end
    # If the dual is infeasible (pFEAS = primal feasible / dual
    # infeasible, or noINFO), add the dual residual.
    if phase === pFEAS || phase === noINFO
        for k in 1:newton_state.DzMat.SDP_nBlock
            if isassigned(newton_state.DzMat.SDP_block, k) &&
               isassigned(current_res.dualMat.SDP_block, k)
                blk = newton_state.DzMat.SDP_block[k].de_ele
                rblk = current_res.dualMat.SDP_block[k].de_ele
                size(blk) == size(rblk) || continue
                @inbounds @simd for idx in eachindex(blk)
                    blk[idx] += rblk[idx]
                end
            end
        end
        @inbounds for k in eachindex(newton_state.DzMat.LP_block)
            if k <= length(current_res.dualMat.LP_block)
                newton_state.DzMat.LP_block[k] += current_res.dualMat.LP_block[k]
            end
        end
    end
    return newton_state
end

"""
    compute_dxmat!(newton_state, current_pt, work)

Recover Δx from the Newton system:
  Δx = R·Z⁻¹ - X·Δz·Z⁻¹

where R = βμI − X·Z, so R·Z⁻¹ = βμ·Z⁻¹ − X (the corrector's R is
βμI − X·Z − Δx·Δz). R·Z⁻¹ is already stored in `r_zinvMat`
(from `compute_rmat!`).

The `X·Δz·Z⁻¹` term uses the non-symmetric Jordan triple product
(`ns_jordan_triple_product!`), which produces an UNSYMMETRIC result.
The final Δx MUST be symmetrized (via `(Δx + Δx') / 2`) because the
IPM convergence requires Δx to be exactly symmetric — any asymmetry
compounds across iterations and destroys convergence (a known
convergence-destroying failure mode of unsymmetrized implementations).

Mirrors C++ `Newton::compute_DxMat` at `sdpa_newton.cpp:1358-1373`.
"""
function compute_dxmat!(
    newton_state::NewtonState{T}, current_pt::SDPASolution{T},
    work::SDPAWorkVariables{T},
) where {T<:Real}
    # work.DLS1 = X · Δz · Z⁻¹ (non-symmetric triple product)
    ns_jordan_triple_product!(
        work.DLS1, current_pt.xMat, newton_state.DzMat,
        current_pt.invzMat, work.DLS2,
    )
    # Δx = r_zinvMat - work.DLS1
    for k in 1:newton_state.DxMat.SDP_nBlock
        if isassigned(newton_state.DxMat.SDP_block, k) &&
           isassigned(newton_state.r_zinvMat.SDP_block, k)
            blk = newton_state.DxMat.SDP_block[k].de_ele
            rblk = newton_state.r_zinvMat.SDP_block[k].de_ele
            wblk = work.DLS1.SDP_block[k].de_ele
            size(blk) == size(rblk) == size(wblk) || continue
            @inbounds @simd for idx in eachindex(blk)
                blk[idx] = rblk[idx] - wblk[idx]
            end
        end
    end
    @inbounds for k in eachindex(newton_state.DxMat.LP_block)
        if k <= length(newton_state.r_zinvMat.LP_block) &&
           k <= length(work.DLS1.LP_block)
            newton_state.DxMat.LP_block[k] =
                newton_state.r_zinvMat.LP_block[k] - work.DLS1.LP_block[k]
        end
    end
    # Symmetrize: Δx = (Δx + Δx') / 2 for each SDP block.
    #
    # The BigFloat path uses ONE scratch BigFloat (plus a
    # pre-allocated exact `two`) for the whole symmetrization instead of
    # two fresh temporaries per element. The rounding chain is reproduced
    # op-for-op: `(a + b) / 2` = one rounded mpfr_add (mp_add!) then an
    # EXACT halving — division by 2 is exact in binary floating point
    # (pure exponent decrement, no rounding at any precision), and Base's
    # `x / 2` is the same mpfr_div by the exactly-representable
    # BigFloat(2) — so the scratch path is bit-identical by construction.
    if T <: BigFloat
        avg_buf = zero(BigFloat)
        two = BigFloat(2)
    end
    for k in 1:newton_state.DxMat.SDP_nBlock
        if isassigned(newton_state.DxMat.SDP_block, k)
            blk = newton_state.DxMat.SDP_block[k].de_ele
            n = size(blk, 1)
            if T <: BigFloat
                @inbounds for j in 1:n
                    for i in 1:j
                        mp_add!(avg_buf, blk[i, j], blk[j, i])
                        mp_div!(avg_buf, avg_buf, two)
                        # Value-copy: `blk[i, j] = avg` would store the SAME
                        # BigFloat object in both transposed slots, so an
                        # in-place MPFR op on one would corrupt the mirror
                        # (same pattern as the ThreadedOps.jl symmetrize).
                        mp_set!(blk[i, j], avg_buf)
                        mp_set!(blk[j, i], avg_buf)
                    end
                end
            else
                @inbounds for j in 1:n
                    for i in 1:j
                        avg = (blk[i, j] + blk[j, i]) / 2
                        blk[i, j] = avg
                        blk[j, i] = avg
                    end
                end
            end
        end
    end
    return newton_state
end
