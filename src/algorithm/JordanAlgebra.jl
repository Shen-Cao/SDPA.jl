"""
Jordan algebra operations on `SDPADenseLinearSpace{T}`.

Mirrors C++ `sdpa::Jal` (Jordan algebra) class at
`sdpa_jordan.h:33-77` and the implementations at
`sdpa_jordan.cpp:1-342`.

The "Jordan product" on the symmetric-matrix vector space is

    A ∘ B = (A·B + B·A) / 2

where `·` is the usual matrix product. This symmetrization is what
defines the SDP search direction: the complementarity condition
`X·Z = μ·I` cannot hold exactly for a primal-dual pair unless we
symmetrize, and the choice of symmetrization defines the search
direction (Nesterov-Todd, HKM, Packard-Chen, etc.). SDPA-GMP uses
the **Jordan-product symmetrization**, which corresponds to the A5
direction in the Monteiro-Zhang 1998 taxonomy.

Reference: Monteiro-Zhang 1998, "A unified analysis for a class of
long-step primal-dual path-following interior-point algorithms for
semidefinite programming", Math Programming 84:3.
"""

"""
    multiply!(ret::SDPADenseLinearSpace{T}, a::SDPADenseLinearSpace{T},
              b::SDPADenseLinearSpace{T}, scalar=nothing) where T

Block-wise dense matrix product `ret = scalar * (a · b)` (or just
`a · b` when `scalar === nothing`). For LP blocks, `ret = scalar * a * b`.

PRECONDITION: `ret` must not alias `a` or `b`; blocks unassigned in
`a`/`b` leave `ret`'s corresponding block stale (callers must pre-zero
or ensure full assignment).

**Important**: the per-block matrix product is NOT symmetric in
general — `a · b ≠ b · a` even when both `a` and `b` are symmetric.
The unsymmetric `multiply!` is the production kernel used by
`ns_jordan_triple_product!` and the Schur-complement assembly.

Mirrors C++ `Jal::multiply` at `sdpa_jordan.cpp:174-200`.
"""
function multiply!(
    ret::SDPADenseLinearSpace{T},
    a::SDPADenseLinearSpace{T},
    b::SDPADenseLinearSpace{T},
    scalar::Union{Nothing,T}=nothing,
) where {T<:Real}
    ret.SDP_nBlock == a.SDP_nBlock == b.SDP_nBlock ||
        error("multiply!: SDP_nBlock mismatch")
    ret.LP_nBlock == a.LP_nBlock == b.LP_nBlock ||
        error("multiply!: LP_nBlock mismatch")
    for k in 1:ret.SDP_nBlock
        if isassigned(a.SDP_block, k) && isassigned(b.SDP_block, k)
            # Use sdpa_mul! (dispatches to Rgemm! for BigFloat,
            # LinearAlgebra.mul! for Float64, where LAPACK is faster).
            sdpa_mul!(ret.SDP_block[k].de_ele,
                      a.SDP_block[k].de_ele, b.SDP_block[k].de_ele)
            if scalar !== nothing
                ret.SDP_block[k].de_ele .*= scalar
            end
        end
    end
    @inbounds for k in eachindex(ret.LP_block)
        prod = a.LP_block[k] * b.LP_block[k]
        ret.LP_block[k] = scalar === nothing ? prod : scalar * prod
    end
    return ret
end

"""
    ns_jordan_triple_product!(ret, a, b, c, work)

Compute the non-symmetric triple product `ret = A · B · C`.

**Critical**: the result is NOT symmetric in general. The caller
MUST symmetrize the result separately (typically via `(M + M') / 2`)
before using it in a symmetric context — unsymmetrized `Δx` will
compound across iterations and destroy convergence (a known
convergence-destroying failure mode of unsymmetrized implementations).

Used in `compute_dxmat!` for the `X · Δz · Z⁻¹` term.

Mirrors C++ `Jal::ns_jordan_triple_product` at `sdpa_jordan.cpp:336-340`
(which chains `Jal::multiply` at `sdpa_jordan.cpp:173-215` twice).
"""
function ns_jordan_triple_product!(
    ret::SDPADenseLinearSpace{T},
    a::SDPADenseLinearSpace{T},
    b::SDPADenseLinearSpace{T},
    c::SDPADenseLinearSpace{T},
    work::SDPADenseLinearSpace{T},
) where {T<:Real}
    multiply!(work, a, b)        # work = A·B
    multiply!(ret, work, c)      # ret = A·B·C
    return ret
end

"""
    get_inv_chol!(inv_chol_mat, a_mat, work_mat) -> Bool

Compute the inverse of the lower-triangular Cholesky factor of `a_mat`.

For each SDP block:
  1. Cholesky-factorize `a_mat.SDP_block[k] = L · L'`
  2. Compute `inv_chol_mat.SDP_block[k] = L⁻¹` (lower-triangular inverse).

For each LP block:
  - `inv_chol_mat.LP_block[k] = 1 / sqrt(a_mat.LP_block[k])` if positive
  - Returns `false` if any LP block is non-positive (`<= 0`).

Returns `false` on Cholesky failure (e.g. `a_mat` is not positive-definite);
caller must handle this case (typically by triggering a step-length reduction
or breaking the main loop).

Mirrors C++ `Jal::getInvChol` at `sdpa_jordan.cpp:123-152`.

!!! note "Deliberate divergence from C++"
    C++ computes `1/sqrt(x)` for any `x >= 0` LP entry, silently producing
    `Inf` when `x == 0` (sdpa_jordan.cpp:145-148). We guard `x <= 0` and
    return `false` instead — a loud failure over silent `Inf`.
"""
function get_inv_chol!(
    inv_chol_mat::SDPADenseLinearSpace{T},
    a_mat::SDPADenseLinearSpace{T},
    work_mat::SDPADenseLinearSpace{T},
) where {T<:Real}
    inv_chol_mat.SDP_nBlock == a_mat.SDP_nBlock ||
        error("get_inv_chol!: SDP_nBlock mismatch")
    inv_chol_mat.LP_nBlock == a_mat.LP_nBlock ||
        error("get_inv_chol!: LP_nBlock mismatch")
    # Pre-allocate local accumulator for BigFloat.
    # The inner triangular-solve loop's `s += L[i,kk]*out[kk,j]` allocates
    # ~2 BigFloats per inner iteration; mp_fma! eliminates this.
    # NOTE: We can ONLY optimize the LOCAL accumulator `s_buf`, not the
    # array writes. The write must store an INDEPENDENT copy of the
    # accumulator (`out[i,j] = deepcopy(s_buf)` below): on Julia 1.12
    # `BigFloat(buf)` is an identity conversion (same object), and the
    # accumulator is reused across iterations, so a plain assignment
    # would make the new slot alias the accumulator and corrupt it on
    # the next inner-loop update.
    local s_buf::T
    has_s_buf = false
    if T <: BigFloat
        s_buf = zero(BigFloat)
        has_s_buf = true
    end
    for k in 1:a_mat.SDP_nBlock
        if !isassigned(a_mat.SDP_block, k)
            continue
        end
        src_blk = a_mat.SDP_block[k].de_ele
        n = size(src_blk, 1)
        # Copy src_blk into work_mat before Cholesky so we don't mutate
        # the input (mirrors C++ Lal::getCholesky(workMat, aMat) which
        # writes the factor into workMat, leaving aMat untouched).
        work_blk = work_mat.SDP_block[k].de_ele
        _copyto_independent!(work_blk, src_blk)
        F = cholesky!(Hermitian(work_blk, :U); check=false)
        if !issuccess(F)
            return false
        end
        # F.factors is upper-triangular U with M = U' * U.
        # The lower-triangular Cholesky factor is L = U'.
        L = LowerTriangular(F.factors')
        # Compute L⁻¹ by forward substitution; store in inv_chol_mat.
        out_blk = inv_chol_mat.SDP_block[k].de_ele
        _fill_independent!(out_blk, zero(T))
        @inbounds for j in 1:n
            out_blk[j, j] = one(T) / L[j, j]
            for i in j+1:n
                if has_s_buf
                    # BigFloat in-place path: zero-allocation via mpfr_fma! mpfr_mul! mpfr_div! mpfr_neg!.
                    mp_zero!(s_buf)  # mpfr_set_si!
                    for kk in j:i-1
                        # s_buf += L[i, kk] * out_blk[kk, j]
                        mp_fma!(s_buf, L[i, kk], out_blk[kk, j], s_buf)  # mpfr_fma!
                    end
                    # out_blk[i, j] = -s_buf / L[i, i]
                    mp_div!(s_buf, s_buf, L[i, i])  # mpfr_div!
                    mp_neg!(s_buf, s_buf)  # mpfr_neg!
                    # CRITICAL: use deepcopy, NOT BigFloat(s_buf).
                    # BigFloat(buf) returns the SAME object on Julia 1.12
                    # (identity convert), so out_blk[i,j] would alias s_buf
                    # and be corrupted by any subsequent s_buf mutation
                    # (e.g. in the LP block loop below).
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
    end
    @inbounds for k in eachindex(a_mat.LP_block)
        if a_mat.LP_block[k] <= 0
            return false
        end
        if has_s_buf
            # BigFloat in-place: inv_chol_mat.LP_block[k] = 1 / sqrt(x)
            mp_rec_sqrt!(s_buf, a_mat.LP_block[k])  # mpfr_rec_sqrt!
            inv_chol_mat.LP_block[k] = deepcopy(s_buf)
        else
            inv_chol_mat.LP_block[k] = one(T) / sqrt(a_mat.LP_block[k])
        end
    end
    return true
end

"""
    get_inv_chol_and_inv!(inv_chol_mat, inverse_mat, a_mat, work_mat) -> Bool

Same as `get_inv_chol!`, but also computes the full inverse
`inverse_mat = (L · L')⁻¹ = L'⁻¹ · L⁻¹`.

For each SDP block:
  1. Calls `get_inv_chol!` to get `inv_chol_mat.SDP_block[k] = L⁻¹`
     (lower-triangular).
  2. Computes `inverse_mat.SDP_block[k] = L'⁻¹ · L⁻¹` via triangular
     matmul: `inverseMat = L⁻¹' * L⁻¹`.

For each LP block:
  - `inverse_mat.LP_block[k] = 1 / a_mat.LP_block[k]`.

Returns `false` on Cholesky failure. Mirrors C++ `Jal::getInvCholAndInv`
at `sdpa_jordan.cpp:154-171`.

!!! note "Deliberate divergence from C++"
    C++ computes `inverse = 1/x` for any `x >= 0` LP entry, silently
    producing `Inf` when `x == 0`. We guard `x <= 0` (returning `false`)
    before the division — a loud failure over silent `Inf`.
"""
function get_inv_chol_and_inv!(
    inv_chol_mat::SDPADenseLinearSpace{T},
    inverse_mat::SDPADenseLinearSpace{T},
    a_mat::SDPADenseLinearSpace{T},
    work_mat::SDPADenseLinearSpace{T},
) where {T<:Real}
    ok = get_inv_chol!(inv_chol_mat, a_mat, work_mat)
    if !ok
        return false
    end
    for k in 1:a_mat.SDP_nBlock
        if !isassigned(a_mat.SDP_block, k)
            continue
        end
        L_inv = inv_chol_mat.SDP_block[k].de_ele   # lower-triangular L⁻¹
        inv_blk = inverse_mat.SDP_block[k].de_ele
        # inverse = L'⁻¹ · L⁻¹ = (L⁻¹)' · L⁻¹ (since L is lower-triangular).
        n = size(L_inv, 1)
        # Explicit Rgemm! on the raw matrix: GenericLinearAlgebra's
        # scalar mul! is slow/allocating for BigFloat; Rgemm! uses
        # zero-allocation mp_fma! loops for BigFloat and
        # LinearAlgebra.mul! (LAPACK) for Float64. Accumulation order
        # differs from the wrapper path (allowed: tolerance-equivalent,
        # not bit-identical). MUST stay kernel-identical with the
        # threaded variant (bit-identical across thread counts).
        Rgemm!("T", "N", n, n, n, one(T), L_inv, L_inv, zero(T), inv_blk)
        # Force symmetry (numerical cleanup). BigFloat: value-copy via
        # mp_set! — `inv_blk[i, j] = avg` would store the SAME BigFloat
        # object in both transposed slots, so a later in-place MPFR op on
        # one would corrupt the mirrored entry. Float64 assignment is
        # already value-semantic.
        #
        # The BigFloat path uses ONE scratch BigFloat (plus a
        # pre-allocated exact `two`) allocated once per block instead of
        # two fresh temporaries per element. The rounding chain is
        # reproduced op-for-op: `(a + b) / 2` = one rounded mpfr_add
        # (mp_add!) then an EXACT halving — division by 2 is exact in
        # binary floating point (pure exponent decrement, no rounding at
        # any precision), and Base's `x / 2` is the same mpfr_div by the
        # exactly-representable BigFloat(2) — so the scratch path is
        # bit-identical by construction. MUST stay kernel-identical with
        # the threaded variant (bit-identical across thread counts).
        if T <: BigFloat
            avg_buf = zero(BigFloat)
            two = BigFloat(2)
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
    @inbounds for k in eachindex(a_mat.LP_block)
        if a_mat.LP_block[k] <= 0
            return false
        end
        inverse_mat.LP_block[k] = one(T) / a_mat.LP_block[k]
    end
    return true
end
