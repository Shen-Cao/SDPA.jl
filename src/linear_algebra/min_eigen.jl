"""
Minimum-eigenvalue solvers: Lanczos (large blocks) + QR (small blocks).

Mirrors C++ `Lal::getMinEigenValue` (QR path, `sdpa_linear.cpp:126-150`)
and `Lal::getMinEigen` (Lanczos path, `sdpa_linear.cpp:27-121`), plus
the dispatcher `Jal::getMinEigen` (`sdpa_jordan.cpp:55-88`).

**Lanczos** is iterative and approximate but fast for large matrices
with few needed eigenvalues. **QR** is exact but O(n³). The dispatcher
uses Lanczos when `blockSize > 20`, QR otherwise — matching C++.

The Lanczos starting vector is **deterministic** (all-ones normalized
to length `sqrt(n)`) so results are bit-identical across runs and
thread counts.
"""

"""
    min_eigen_qr(mat::AbstractMatrix{T}) where T

Compute the minimum eigenvalue of the symmetric matrix `mat` via full
symmetric eigensolve. Dispatches to LAPACK `syev!` for `BlasFloat` and
to `GenericLinearAlgebra.eigen` for `BigFloat` (via `update_eigen!`).

Returns `λ_min::T`. The input matrix is **overwritten** with
eigenvectors (matching the C++ convention).

Mirrors C++ `Lal::getMinEigenValue` at `sdpa_linear.cpp:126-150`.
"""
function min_eigen_qr(mat::AbstractMatrix{T}) where {T<:Real}
    vals = update_eigen!(mat)
    return vals[1]
end

"""
    min_eigen_qr_block(mat::AbstractMatrix{T}) where T

Wrapper that copies `mat` before eigensolve so the input is not
mutated. Used by `min_eigen` dispatcher when we need to preserve the
original matrix.
"""
function min_eigen_qr_block(mat::AbstractMatrix{T}) where {T<:Real}
    # Independent copy: for BigFloat, `copy` shares element objects (reference
    # copy), and `update_eigen!` writes eigenvectors via in-place mp_set! —
    # a reference copy would mutate the CALLER's elements. The helper
    # dispatches to plain `copy` for isbits T.
    mat_copy = _copy_independent(mat)
    return min_eigen_qr(mat_copy)
end

"""
    lanczos_restart_threshold(n) -> BigFloat

Restart threshold for the periodic eigen-restart in `min_eigen_lanczos`:
the restart fires when `k > sqrt((mpf_class)nDim + 9)` — sqrt of the SUM,
evaluated in arbitrary precision (C++ `sdpa_linear.cpp:69`). This is NOT
`floor(sqrt(nDim)) + 9`, and it is distinct from the OUTER iteration cap
`k < sqrt(nDim) + 10` (C++ `sdpa_linear.cpp:43`). Julia's `iter_cap` uses
the floored `floor(Int, sqrt(n)) + 10` — an intentional, documented
deviation; see `min_eigen_lanczos`.
"""
@inline lanczos_restart_threshold(n::Int) = sqrt(BigFloat(n + 9))

"""
    min_eigen_lanczos(l_mat::AbstractMatrix{T}, x_mat::AbstractMatrix{T},
                      max_iter::Int) where T

Native Julia implementation of the Lanczos iteration for
`min λ(L · X · L')` where `L` is lower-triangular and `X` is symmetric.

Mirrors C++ `Lal::getMinEigen` at `sdpa_linear.cpp:27-121`.

Algorithm:
  1. Deterministic start: `r = ones(n)` (norm `sqrt(n)`).
  2. Iteratively build the Krylov basis of `M = L · X · L'`:
       - `q_{k+1} = r / beta`
       - `w = M · q_{k+1}` (computed as `L · (X · (L' · q_{k+1}))`)
       - `alpha = q_{k+1}' · w`
       - `r = w - alpha * q_{k+1} - beta * qold`  (3-term recurrence)
       - `beta = ||r||`
  3. Periodically (at `k == 0` and then every `sqrt(k)` steps, at
     `k == n - 1`, and at the restart cap) solve the tridiagonal
     eigenproblem on `(alpha_0, ..., alpha_k)` and
     `(beta_0, ..., beta_{k-1})`.
  4. Convergence: stop when the relative change in `λ_min` drops below
     `1e-5` (with an absolute floor of `1e-8`) AND the error estimate
     `|err_est * beta|` is below `1e-2` relative to `|λ_min|` (with an
     absolute floor of `1e-4`), OR when `k` reaches the cap
     `min(n, floor(Int, sqrt(n)) + 10)`.

**Documented deviation from C++**: the iteration cap here is
`floor(Int, sqrt(n)) + 10`, vs the un-floored loop bound
`k < sqrt(nDim) + 10` in C++ (`sdpa_linear.cpp:43`) — so for `n` that
is not a perfect square the Julia cap admits up to one iteration fewer.
Both caps are valid Lanczos heuristics; the deviation is intentionally
kept because changing it would alter the converged numerics with no
correctness or performance gain.

Returns `λ_min - |err_est * beta|` (the Ritz value corrected by the
sensitivity estimate, matching C++).
"""
function min_eigen_lanczos(
    l_mat::AbstractMatrix{T}, x_mat::AbstractMatrix{T}, max_iter::Int=0
) where {T<:Real}
    n = size(l_mat, 1)
    size(l_mat, 2) == n || error("l_mat must be square")
    size(x_mat) == (n, n) || error("x_mat must match l_mat size")

    # Cap on iterations. DOCUMENTED DEVIATION from C++: the C++ loop
    # bound is the un-floored `k < sqrt(nDim) + 10` (sdpa_linear.cpp:43);
    # Julia floors the sqrt, admitting up to one iteration fewer when n
    # is not a perfect square. Both caps are valid heuristics; kept as-is
    # because changing it would alter numerics with no correctness or
    # performance gain (see the docstring for the full rationale).
    iter_cap = max_iter > 0 ? min(max_iter, n) : min(n, floor(Int, sqrt(n)) + 10)

    # Deterministic start: r = ones(n), beta = sqrt(n) (norm of r).
    # Work vectors must have INDEPENDENT BigFloat elements: `zeros`/`ones`
    # fill every slot with ONE shared BigFloat object (a latent aliasing
    # hazard for any in-place MPFR op). The helpers dispatch to plain
    # `zeros`/`fill!` for isbits T.
    q = _independent_zeros(T, n)
    q_old = _independent_zeros(T, n)
    r = _independent_zeros(T, n)
    _fill_independent!(r, one(T))
    beta = sqrt(T(n))

    # Tri-diagonal accumulator: alphas (diagonal), betas (off-diagonal).
    alphas = _independent_zeros(T, n)
    betas = _independent_zeros(T, n)

    # Work vectors.
    w = _independent_zeros(T, n)
    tmp = _independent_zeros(T, n)

    min_val = T(1e50)
    min_old = T(1e52)
    # Named err_est, NOT error — a local named error shadows the
    # Base.error function and breaks the dimension-check error() calls
    # at the top of this function.
    err_est = T(1e10)

    L_tri = LowerTriangular(l_mat)

    kk = 0  # counter for periodic eigen-restart
    k = 0
    # Hoisted out of the loop: `lanczos_restart_threshold(n)` allocates a
    # BigFloat on every call (`sqrt(BigFloat(n + 9))`).
    restart_thresh = lanczos_restart_threshold(n)

    while k < iter_cap
        if beta <= T(1e-16)
            break
        end

        # Value copy (BigFloat): `copyto!` would alias q_old[i] === q[i].
        # The helper dispatches to plain `copyto!` for isbits T.
        _copyto_independent!(q_old, q)
        inv_beta = one(T) / beta
        if T <: BigFloat
            # in-place MPFR normalization (one rounding, zero alloc).
            @inbounds for i in 1:n
                mp_mul!(q[i], r[i], inv_beta)
            end
        else
            @inbounds @simd for i in 1:n
                q[i] = r[i] * inv_beta
            end
        end

        # Krylov matvec w = L · X · L' · q, computed as L · (X · (L' · q)).
        if T <: BigFloat
            # Explicit mp_fma! matvecs on the RAW matrices. The
            # LowerTriangular/Adjoint wrapper mul!s would dispatch BigFloat
            # to GenericLinearAlgebra's scalar path (slow, allocating);
            # these loops are zero-allocation (one rounding per term, mp_fma!).
            # Accumulation order differs from the wrapper path — allowed
            # (tolerance-equivalent, not bit-identical).
            # w = L' * q (L' upper-triangular: w[i] = Σ_{j≥i} L[j,i]·q[j])
            @inbounds for i in 1:n
                mp_zero!(w[i])
                for j in i:n
                    mp_fma!(w[i], l_mat[j, i], q[j], w[i])
                end
            end
            # tmp = X * w
            @inbounds for i in 1:n
                mp_zero!(tmp[i])
                for j in 1:n
                    mp_fma!(tmp[i], x_mat[i, j], w[j], tmp[i])
                end
            end
            # w = L * tmp (L lower-triangular: w[i] = Σ_{j≤i} L[i,j]·tmp[j])
            @inbounds for i in 1:n
                mp_zero!(w[i])
                for j in 1:i
                    mp_fma!(w[i], l_mat[i, j], tmp[j], w[i])
                end
            end
        else
            # Float64: BLAS gemv via the wrappers (LAPACK is faster than a
            # scalar loop for isbits types).
            mul!(w, LinearAlgebra.adjoint(L_tri), q)
            mul!(tmp, x_mat, w)
            mul!(w, L_tri, tmp)
        end
        # Now w = L * X * L' * q (the Krylov matrix-vector product).

        alpha = zero(T)
        if T <: BigFloat
            # in-place dot product (single rounding per term, zero alloc).
            mp_zero!(alpha)
            @inbounds for i in 1:n
                mp_add_mul!(alpha, q[i], w[i])
            end
        else
            @inbounds @simd for i in 1:n
                alpha += q[i] * w[i]
            end
        end
        alphas[k + 1] = alpha

        # r = w - alpha*q - beta*qold
        if T <: BigFloat
            # in-place 3-term recurrence (single rounding per term, zero alloc).
            @inbounds for i in 1:n
                mp_set!(r[i], w[i])
                mp_sub_mul!(r[i], alpha, q[i])
                mp_sub_mul!(r[i], beta, q_old[i])
            end
        else
            @inbounds @simd for i in 1:n
                r[i] = w[i] - alpha * q[i] - beta * q_old[i]
            end
        end

        # Periodic eigen-restart: solve tridiagonal eigenproblem.
        # C++ parity (Lal::getMinEigen, sdpa_linear.cpp): the restart test
        # is `kk >= floor(sqrt(k))`, which at k=0 reads 0 >= 0 and is
        # ALWAYS TRUE — the very first iteration computes a 1×1 Ritz
        # value. The former `max(1, k)` suppressed that k=0 restart, so a
        # Lanczos run whose starting vector is already an exact
        # eigenvector (residual ≤ 1e-16 at k=1) broke out with the 1e50
        # sentinel still in min_val, which downstream reads as "no
        # negative curvature" and awards a full alphaBD step.
        if kk >= floor(Int, sqrt(k)) || k == n - 1 || k > restart_thresh
            kk = 0
            # Build the (k+1) x (k+1) tridiagonal submatrix; k starts at 0
            # and only increments, so kp1 = k + 1 >= 1 always holds.
            kp1 = k + 1
            diag = alphas[1:kp1]
            offdiag = kp1 >= 2 ? betas[1:kp1-1] : T[]
            # SymTridiagonal expects off-diagonal of length n-1.
            if length(offdiag) == kp1 - 1
                T_tri = SymTridiagonal(diag, offdiag)
                F = eigen(T_tri)
                min_val_new = F.values[1]
                err_est = F.vectors[kp1, 1]  # sensitivity of last basis vector
                min_old = min_val
                min_val = min_val_new
            end
        end

        # beta = ||r||
        beta_sq = zero(T)
        if T <: BigFloat
            # in-place norm accumulation (single rounding per term, zero alloc).
            mp_zero!(beta_sq)
            @inbounds for i in 1:n
                mp_add_mul!(beta_sq, r[i], r[i])
            end
        else
            @inbounds @simd for i in 1:n
                beta_sq += r[i] * r[i]
            end
        end
        beta = sqrt(beta_sq)
        betas[k + 1] = beta  # off-diagonal for next iteration's tridiagonal

        # Convergence test (matches C++ sdpa_linear.cpp:43-46).
        if abs(min_val - min_old) <= T(1e-5) * abs(min_val) + T(1e-8) &&
           abs(err_est * beta) <= T(1e-2) * abs(min_val) + T(1e-4)
            break
        end

        k += 1
        kk += 1
    end

    return min_val - abs(err_est * beta)
end

"""
    min_eigen_dispatch(l_mat, x_mat) where T

Block-size dispatcher. Uses Lanczos when `size > 20`, QR otherwise.
Mirrors C++ `Jal::getMinEigen` at `sdpa_jordan.cpp:55-88`.
"""
function min_eigen_dispatch(
    l_mat::AbstractMatrix{T}, x_mat::AbstractMatrix{T}
) where {T<:Real}
    n = size(x_mat, 1)
    # Validate BOTH matrices fully: a non-square l_mat (or an x_mat
    # with a mismatched column count) must fail loudly instead of
    # silently dropping rows/columns and returning a wrong eigenvalue.
    size(l_mat, 1) == n && size(l_mat, 2) == n || throw(DimensionMismatch(
        "min_eigen_dispatch: l_mat is $(size(l_mat)); expected ($n, $n)"))
    size(x_mat, 2) == n || throw(DimensionMismatch(
        "min_eigen_dispatch: x_mat is $(size(x_mat)); expected ($n, $n)"))
    if n > 20
        return min_eigen_lanczos(l_mat, x_mat)
    else
        # QR path: compute L * X * L' explicitly, then full eigensolve.
        # (dimension checks done once at the top of min_eigen_dispatch)
        # Rgemm! MUTATES C in place (mp_zero!/mp_fma! per element), so the
        # outputs must have INDEPENDENT, DEFINED BigFloat elements —
        # `similar` would leave BigFloat slots #undef; the helper
        # dispatches to `similar` for isbits T.
        work = _independent_zeros(T, n, n)
        out = _independent_zeros(T, n, n)
        # Explicit Rgemm! on the raw matrices: the
        # LowerTriangular/Adjoint wrapper mul!s would dispatch BigFloat
        # to GenericLinearAlgebra's scalar path (slow, allocating);
        # Rgemm! uses zero-allocation mp_fma! loops for BigFloat and
        # LinearAlgebra.mul! (LAPACK) for Float64. Accumulation order
        # differs from the wrapper path — tolerance-equivalent
        # (rtol 1e-30/1e-15), not bit-identical.
        # work = X * L'
        Rgemm!("N", "T", n, n, n, one(T), x_mat, l_mat, zero(T), work)
        # out = L * work = L * X * L'
        Rgemm!("N", "N", n, n, n, one(T), l_mat, work, zero(T), out)
        return min_eigen_qr_block(out)
    end
end

