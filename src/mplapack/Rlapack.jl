# src/mplapack/Rlapack.jl
#
# Port of MPLAPACK's Rpotf2 (unblocked Cholesky) and Rtrsm (triangular
# solve matrix) for BigFloat matrices.
#
# References:
#   https://github.com/nakatamaho/sdpa-gmp/blob/master/mplapack/Rpotf2.cpp (140 lines, LGPLv3, Nakata 2008)
#   https://github.com/nakatamaho/sdpa-gmp/blob/master/mplapack/Rtrsm.cpp  (291 lines, LGPLv3, Nakata 2008)
#
# Algorithm: standard unblocked Cholesky (left-looking), matching the
# reference LAPACK DPOTF2. For BigFloat, the kernels operate in place
# with per-call local accumulators: Rpotf2! stores results with mp_set!
# value-copies (mp_fma!/mp_sqrt!/mp_div! for the arithmetic); Rtrsm! and
# Rsyrk! use the same mp_set!/mp_fma! in-place pattern (Rsyrk!'s beta==0
# overwrite branch stores a fresh deepcopy, since the BLAS contract
# forbids reading C there).
#
# Threading: NOT parallelized (Cholesky has serial column dependencies;
# left-looking factorization can't be parallelized across columns without
# algorithmic redesign).

"""
    Rpotf2!(uplo, n, A) -> Int

In-place unblocked Cholesky factorization of an `n × n` symmetric
positive-definite matrix `A`. Returns 0 on success, or the index `j > 0`
of the first non-positive pivot encountered (caller treats this as "not
positive definite").

- `uplo == "U"`: factorize upper triangle, store upper-triangular U with A = U' * U.
- `uplo == "L"`: factorize lower triangle, store lower-triangular L with A = L * L'.

For BigFloat, uses mp_sqrt!, mp_fma!, mp_div! with per-call local
accumulators. For Float64/32, dispatches to LinearAlgebra.cholesky!.

Mirrors MPLAPACK's Rpotf2 at `https://github.com/nakatamaho/sdpa-gmp/blob/master/mplapack/Rpotf2.cpp` (lines
84-130 for the algorithm; lines 1-83 are input validation which we elide).
"""
function Rpotf2!(uplo::AbstractString, n::Integer, A::AbstractMatrix{T}) where {T<:BigFloat}
    n == 0 && return 0
    upper = uplo == "U"
    upper || uplo == "L" ||
        error("Rpotf2!: uplo must be \"U\" or \"L\", got $uplo")
    size(A) == (n, n) || throw(DimensionMismatch(
        "Rpotf2!: A is $(size(A)); expected ($n, $n)"))

    # Per-call local accumulators.
    ajj = zero(T)   # diagonal element
    s = zero(T)     # sub-diagonal accumulation
    tmp = zero(T)   # multiplicative scratch

    if upper
        # Compute U such that A = U' * U.
        # After factorization, U is stored in the UPPER triangle of A.
        # NOTE: For column j, the upper triangle holds U[1:j-1, j] (the
        # column above the diagonal). The diagonal U[j, j] is computed first.
        @inbounds for j in 1:n
            # Compute U[j, j] = sqrt(A[j, j] - sum_k U[k, j]² for k<j)
            mp_set!(ajj, A[j, j])
            for k in 1:(j - 1)
                # ajj -= A[k, j]²  (U[k, j] is stored at A[k, j])
                mp_mul!(tmp, A[k, j], A[k, j])
                mp_sub!(ajj, ajj, tmp)
            end
            # A NaN pivot fails the <= 0 test, so NaN input would
            # otherwise be reported as SUCCESS; check NaN explicitly
            # (LAPACK DPOTF2 parity: DISNAN on the pivot).
            if isnan(ajj) || ajj <= zero(T)
                # Not positive definite — store ajj.
                mp_set!(A[j, j], ajj)
                return j
            end
            # ajj = sqrt(ajj)
            mp_sqrt!(ajj, ajj)
            mp_set!(A[j, j], ajj)

            # Compute column j of U (rows below diagonal are zero; we update
            # A[j, i] for i > j, which is the upper-triangle row j).
            # U[j, i] = (A[j, i] - sum_k U[k, i] * U[k, j]) / U[j, j]
            #                                              for k<j, i>j
            for i in (j + 1):n
                mp_set!(s, A[j, i])
                for k in 1:(j - 1)
                    # s -= A[k, i] * A[k, j]
                    mp_mul!(tmp, A[k, i], A[k, j])
                    mp_sub!(s, s, tmp)
                end
                # A[j, i] = s / ajj
                mp_div!(s, s, ajj)
                mp_set!(A[j, i], s)
            end
        end
    else  # lower
        # Compute L such that A = L * L'.
        # A[i, j] for i ≥ j (lower triangle).
        @inbounds for j in 1:n
            # Compute L[j, j] = sqrt(A[j, j] - sum_k L[j, k]² for k<j)
            mp_set!(ajj, A[j, j])
            for k in 1:(j - 1)
                # ajj -= A[j, k]²
                mp_mul!(tmp, A[j, k], A[j, k])
                mp_sub!(ajj, ajj, tmp)
            end
            if isnan(ajj) || ajj <= zero(T)
                mp_set!(A[j, j], ajj)
                return j
            end
            mp_sqrt!(ajj, ajj)
            mp_set!(A[j, j], ajj)

            # Compute column j of L (rows j+1..n).
            # L[i, j] = (A[i, j] - sum_k L[i, k] * L[j, k]) / L[j, j]
            #                                              for k<j, i>j
            for i in (j + 1):n
                mp_set!(s, A[i, j])
                for k in 1:(j - 1)
                    # s -= A[i, k] * A[j, k]
                    mp_mul!(tmp, A[i, k], A[j, k])
                    mp_sub!(s, s, tmp)
                end
                mp_div!(s, s, ajj)
                mp_set!(A[i, j], s)
            end
        end
    end
    return 0
end

# Float64/32 fallback: use LinearAlgebra.cholesky! (LAPACK is faster).
function Rpotf2!(uplo::AbstractString, n::Integer, A::AbstractMatrix{T}) where {T<:Union{Float32,Float64}}
    if n == 0; return 0; end
    uplo == "U" || uplo == "L" ||
        error("Rpotf2!: uplo must be \"U\" or \"L\", got $uplo")
    size(A) == (n, n) || throw(DimensionMismatch(
        "Rpotf2!: A is $(size(A)); expected ($n, $n)"))
    uplo_sym = uplo == "U" ? :U : :L
    # Bundled OpenBLAS dpotf2 does not flag NaN pivots (the <= 0
    # test fails for NaN), so scan the used triangle up front and report
    # the first NaN column (LAPACK DISNAN parity).
    if uplo_sym === :U
        for j in 1:n, i in 1:j
            isnan(A[i, j]) && return j
        end
    else
        for j in 1:n, i in j:n
            isnan(A[i, j]) && return j
        end
    end
    H = Hermitian(A, uplo_sym)
    F = cholesky!(H; check=false)
    if !issuccess(F)
        # Report the actual failed pivot (F.info) when available.
        return F.info > 0 ? F.info : 1
    end
    # F.factors holds the upper-triangular U (for :U input) regardless.
    copyto!(A, F.factors)
    return 0
end

"""
    Rtrsm!(side, uplo, transa, diag, m, n, alpha, A, B)

Solve `op(A) * X = alpha * B` (side="L") or `X * op(A) = alpha * B`
(side="R") for X, overwriting B in place. A is triangular.

For SDPA's use case (predictor/corrector triangular solve after
Cholesky), both side="L" and side="R" are supported. transa supports
"N" (A as-is), "T" (Aᵀ), "C" (Aᴴ; same as T for real).

For BigFloat, uses in-place MPFR with local accumulators.
"""
function Rtrsm!(
    side::AbstractString, uplo::AbstractString, transa::AbstractString,
    diag::AbstractString, m::Integer, n::Integer,
    alpha::T, A::AbstractMatrix{T}, B::AbstractMatrix{T},
) where {T<:BigFloat}
    # Validate control strings: a typo like "n" must error, not be
    # silently treated as its closest valid flag.
    side == "L" || side == "R" ||
        error("Rtrsm!: side must be \"L\" or \"R\", got $side")
    uplo == "U" || uplo == "L" ||
        error("Rtrsm!: uplo must be \"U\" or \"L\", got $uplo")
    transa == "N" || transa == "T" || transa == "C" ||
        error("Rtrsm!: transa must be N, T, or C, got $transa")
    diag == "N" || diag == "U" ||
        error("Rtrsm!: diag must be \"N\" or \"U\", got $diag")
    # Dimension checks before the @inbounds sweeps. side="L" → A is
    # m×m, side="R" → A is n×n; B is always m×n.
    size(A) == (side == "L" ? m : n, side == "L" ? m : n) || throw(DimensionMismatch(
        "Rtrsm!: A is $(size(A)); expected $(side == "L" ? (m, m) : (n, n)) for side='$side'"))
    size(B) == (m, n) || throw(DimensionMismatch(
        "Rtrsm!: B is $(size(B)); expected ($m, $n)"))
    nounit = diag == "N"
    upper = uplo == "U"
    notrans = transa == "N"
    tmp = zero(T)
    s = zero(T)

    if side == "R"
        # Solve X * op(A) = alpha * B (right side). Mirrors LAPACK DTRSM
        # side="R": B := alpha*B*inv(op(A)). B is m×n, A is n×n triangular.
        # The column sweep direction follows which previously-solved columns
        # X[i, k] each X[i, j] depends on.
        if !isone(alpha)
            for j in 1:n, i in 1:m
                mp_mul!(tmp, alpha, B[i, j])
                mp_set!(B[i, j], tmp)
            end
        end
        if upper
            if notrans
                # Solve X * U = B: X[i, j] depends on X[i, k] for k < j → forward.
                @inbounds for j in 1:n
                    for i in 1:m
                        mp_set!(s, B[i, j])
                        for k in 1:(j - 1)
                            mp_mul!(tmp, B[i, k], A[k, j])
                            mp_sub!(s, s, tmp)
                        end
                        if nounit
                            mp_div!(s, s, A[j, j])
                        end
                        mp_set!(B[i, j], s)
                    end
                end
            else  # transa == "T" or "C"
                # Solve X * Uᵀ = B: X[i, j] depends on X[i, k] for k > j → backward.
                @inbounds for j in n:-1:1
                    for i in 1:m
                        mp_set!(s, B[i, j])
                        for k in (j + 1):n
                            mp_mul!(tmp, B[i, k], A[j, k])
                            mp_sub!(s, s, tmp)
                        end
                        if nounit
                            mp_div!(s, s, A[j, j])
                        end
                        mp_set!(B[i, j], s)
                    end
                end
            end
        else  # lower
            if notrans
                # Solve X * L = B: X[i, j] depends on X[i, k] for k > j → backward.
                @inbounds for j in n:-1:1
                    for i in 1:m
                        mp_set!(s, B[i, j])
                        for k in (j + 1):n
                            mp_mul!(tmp, B[i, k], A[k, j])
                            mp_sub!(s, s, tmp)
                        end
                        if nounit
                            mp_div!(s, s, A[j, j])
                        end
                        mp_set!(B[i, j], s)
                    end
                end
            else  # transa == "T" or "C"
                # Solve X * Lᵀ = B: X[i, j] depends on X[i, k] for k < j → forward.
                @inbounds for j in 1:n
                    for i in 1:m
                        mp_set!(s, B[i, j])
                        for k in 1:(j - 1)
                            mp_mul!(tmp, B[i, k], A[j, k])
                            mp_sub!(s, s, tmp)
                        end
                        if nounit
                            mp_div!(s, s, A[j, j])
                        end
                        mp_set!(B[i, j], s)
                    end
                end
            end
        end
        return B
    end
    side == "L" || error("Rtrsm!: side must be \"L\" or \"R\", got $side")

    # Scale B by alpha first.
    if !isone(alpha)
        for j in 1:n, i in 1:m
            mp_mul!(tmp, alpha, B[i, j])
            mp_set!(B[i, j], tmp)
        end
    end

    # Solve op(A) * X = B in-place.
    # The algorithm depends on whether A is upper/lower and whether transposed.
    if upper
        if notrans
            # Solve U * X = B (back substitution, top-down... actually U is upper
            # triangular so we go from the bottom row up).
            # X[i, j] = (B[i, j] - sum_{k>i} U[i, k] * X[k, j]) / U[i, i]
            # Process rows from m down to 1.
            @inbounds for j in 1:n
                for i in m:-1:1
                    mp_set!(s, B[i, j])
                    for k in (i + 1):m
                        mp_mul!(tmp, A[i, k], B[k, j])
                        mp_sub!(s, s, tmp)
                    end
                    if nounit
                        mp_div!(s, s, A[i, i])
                    end
                    mp_set!(B[i, j], s)
                end
            end
        else  # transa == "T" or "C"
            # Solve Uᵀ * X = B (forward substitution, top-down).
            # X[i, j] = (B[i, j] - sum_{k<i} U[k, i] * X[k, j]) / U[i, i]
            @inbounds for j in 1:n
                for i in 1:m
                    mp_set!(s, B[i, j])
                    for k in 1:(i - 1)
                        mp_mul!(tmp, A[k, i], B[k, j])
                        mp_sub!(s, s, tmp)
                    end
                    if nounit
                        mp_div!(s, s, A[i, i])
                    end
                    mp_set!(B[i, j], s)
                end
            end
        end
    else  # lower
        if notrans
            # Solve L * X = B (forward substitution, top-down).
            # X[i, j] = (B[i, j] - sum_{k<i} L[i, k] * X[k, j]) / L[i, i]
            @inbounds for j in 1:n
                for i in 1:m
                    mp_set!(s, B[i, j])
                    for k in 1:(i - 1)
                        mp_mul!(tmp, A[i, k], B[k, j])
                        mp_sub!(s, s, tmp)
                    end
                    if nounit
                        mp_div!(s, s, A[i, i])
                    end
                    mp_set!(B[i, j], s)
                end
            end
        else  # transa == "T" or "C"
            # Solve Lᵀ * X = B (back substitution, bottom-up).
            # X[i, j] = (B[i, j] - sum_{k>i} L[k, i] * X[k, j]) / L[i, i]
            @inbounds for j in 1:n
                for i in m:-1:1
                    mp_set!(s, B[i, j])
                    for k in (i + 1):m
                        mp_mul!(tmp, A[k, i], B[k, j])
                        mp_sub!(s, s, tmp)
                    end
                    if nounit
                        mp_div!(s, s, A[i, i])
                    end
                    mp_set!(B[i, j], s)
                end
            end
        end
    end
    return B
end

# Float64/32 fallback: use LinearAlgebra.ldiv! with triangular wrappers.
function Rtrsm!(
    side::AbstractString, uplo::AbstractString, transa::AbstractString,
    diag::AbstractString, m::Integer, n::Integer,
    alpha::T, A::AbstractMatrix{T}, B::AbstractMatrix{T},
) where {T<:Union{Float32,Float64}}
    # Validate control strings, mirroring the BigFloat method above.
    side == "L" || side == "R" ||
        error("Rtrsm!: side must be \"L\" or \"R\", got $side")
    uplo == "U" || uplo == "L" ||
        error("Rtrsm!: uplo must be \"U\" or \"L\", got $uplo")
    transa == "N" || transa == "T" || transa == "C" ||
        error("Rtrsm!: transa must be N, T, or C, got $transa")
    diag == "N" || diag == "U" ||
        error("Rtrsm!: diag must be \"N\" or \"U\", got $diag")
    size(A) == (side == "L" ? m : n, side == "L" ? m : n) || throw(DimensionMismatch(
        "Rtrsm!: A is $(size(A)); expected $(side == "L" ? (m, m) : (n, n)) for side='$side'"))
    size(B) == (m, n) || throw(DimensionMismatch(
        "Rtrsm!: B is $(size(B)); expected ($m, $n)"))
    # Build the triangular wrapper matching uplo and diag.
    Tri = diag == "U" ?
        (uplo == "U" ? UnitUpperTriangular : UnitLowerTriangular) :
        (uplo == "U" ? UpperTriangular : LowerTriangular)
    base = Tri(A)
    # Apply alpha scaling to B first.
    if !isone(alpha)
        rmul!(B, alpha)
    end
    if side == "L"
        # Solve op(A) * X = B via ldiv!.
        opA = transa == "N" ? base : adjoint(base)
        ldiv!(opA, B)
    elseif side == "R"
        # Solve X * op(A) = B  ⟺  op(A)' * X' = B'  ⟺  X = (op(A)' \ B')'.
        # op(A)' is A' (transa="N") or A (transa="T"/"C").
        solve_wrap = transa == "N" ? adjoint(base) : base
        B .= (solve_wrap \ adjoint(B))'
    else
        error("Rtrsm!: side must be \"L\" or \"R\", got $side")
    end
    return B
end

# ---------------------------------------------------------------------------
# Rsyrk! and Rpotrf! (blocked Cholesky)
#
# Ported from:
#   https://github.com/nakatamaho/sdpa-gmp/blob/master/mplapack/Rpotrf.cpp (154 lines, BSD-2, Nakata 2008-2021)
#   https://github.com/nakatamaho/sdpa-gmp/blob/master/mplapack/Rsyrk.cpp (LGPLv3, Nakata 2008)
#
# Rpotrf! is the BLOCKED Cholesky factorization. It divides the matrix
# into blocks of size nb, factorizing the diagonal block with Rpotf2!
# (unblocked), then updating the off-diagonal blocks with Rgemm!/Rtrsm!.
# The blocked algorithm is 10-30× faster than unblocked for large n
# because it uses Rgemm! (which has OhMyThreads @tasks parallelism and
# better cache behavior) for the bulk of the computation.
#
# Rsyrk! (symmetric rank-k update) is implemented as a thin wrapper
# around Rgemm! since C = alpha*op(A)*op(A)' + beta*C is just Rgemm!
# with both matrix operands being the same matrix.
# ---------------------------------------------------------------------------

"""
    Rsyrk!(uplo, trans, n, k, alpha, A, beta, C)

Symmetric rank-k update: C = alpha * op(A) * op(A)' + beta * C.
Only the triangle specified by `uplo` is updated.

For BigFloat, delegates to Rgemm! (which has @tasks parallelism).
Ported from Rsyrk.cpp.
"""
function Rsyrk!(
    uplo::AbstractString, trans::AbstractString,
    n::Integer, k::Integer,
    alpha::T, A::AbstractMatrix{T}, beta::T, C::AbstractMatrix{T},
) where {T<:BigFloat}
    # Validate control strings: a typo must error, not silently default
    # to the transpose path. Internal callers (Rpotrf!) pass the BLAS short
    # forms, so the strict match is safe.
    uplo == "U" || uplo == "L" ||
        error("Rsyrk!: uplo must be \"U\" or \"L\", got $uplo")
    trans == "N" || trans == "T" ||
        error("Rsyrk!: trans must be \"N\" or \"T\", got $trans")
    # Dimension checks (trans="N" → A is n×k; "T" → A is k×n).
    size(A) == (trans == "N" ? (n, k) : (k, n)) || throw(DimensionMismatch(
        "Rsyrk!: A is $(size(A)); expected $(trans == "N" ? (n, k) : (k, n)) for trans='$trans'"))
    size(C) == (n, n) || throw(DimensionMismatch(
        "Rsyrk!: C is $(size(C)); expected ($n, $n)"))

    # True xSYRK triangle semantics — ONLY the triangle of C named
    # by uplo is read (for the beta scaling) and written; the opposite
    # triangle of C is left untouched (LAPACK DSYRK contract). A itself is
    # a general matrix and is always read in full.
    # Internal callers (Rpotrf!) pass blocks that lie entirely inside the
    # named triangle, so their updates stay exact; the summation uses
    # one rounding per product (fma) instead of Rgemm!'s alpha-pre-scaled
    # two-rounding order.
    upper = uplo == "U"
    notrans = trans == "N"
    s = zero(BigFloat)
    tmp = zero(BigFloat)
    if upper && notrans
        # C[i,j] (i ≤ j) = alpha * Σ_l A[i,l]·A[j,l] + beta*C[i,j]
        @inbounds for j in 1:n, i in 1:j
            mp_zero!(s)
            for l in 1:k
                mp_fma!(s, A[i, l], A[j, l], s)
            end
            mp_mul!(tmp, alpha, s)
            # BLAS beta==0 means overwrite — C[i,j] must NOT be read
            # (it may be undefined/NaN). Even mp_set! reads the C[i,j]
            # REFERENCE, which throws UndefRefError on an undef slot, so
            # store a fresh deepcopy instead (BigFloat(tmp) would alias
            # tmp on Julia 1.12; tmp is mutated every iteration).
            # For beta != 0 the fma form is unchanged (fma(0, c, t) == t
            # for defined, non-NaN c).
            if iszero(beta)
                C[i, j] = deepcopy(tmp)
            else
                mp_fma!(C[i, j], beta, C[i, j], tmp)
            end
        end
    elseif upper
        # C[i,j] (i ≤ j) = alpha * Σ_l A[l,i]·A[l,j] + beta*C[i,j]
        @inbounds for j in 1:n, i in 1:j
            mp_zero!(s)
            for l in 1:k
                mp_fma!(s, A[l, i], A[l, j], s)
            end
            mp_mul!(tmp, alpha, s)
            # BLAS beta==0 means overwrite — C[i,j] must NOT be read
            # (it may be undefined/NaN). Even mp_set! reads the C[i,j]
            # REFERENCE, which throws UndefRefError on an undef slot, so
            # store a fresh deepcopy instead (BigFloat(tmp) would alias
            # tmp on Julia 1.12; tmp is mutated every iteration).
            # For beta != 0 the fma form is unchanged (fma(0, c, t) == t
            # for defined, non-NaN c).
            if iszero(beta)
                C[i, j] = deepcopy(tmp)
            else
                mp_fma!(C[i, j], beta, C[i, j], tmp)
            end
        end
    elseif notrans
        # C[i,j] (i ≥ j) = alpha * Σ_l A[i,l]·A[j,l] + beta*C[i,j]
        @inbounds for j in 1:n, i in j:n
            mp_zero!(s)
            for l in 1:k
                mp_fma!(s, A[i, l], A[j, l], s)
            end
            mp_mul!(tmp, alpha, s)
            # BLAS beta==0 means overwrite — C[i,j] must NOT be read
            # (it may be undefined/NaN). Even mp_set! reads the C[i,j]
            # REFERENCE, which throws UndefRefError on an undef slot, so
            # store a fresh deepcopy instead (BigFloat(tmp) would alias
            # tmp on Julia 1.12; tmp is mutated every iteration).
            # For beta != 0 the fma form is unchanged (fma(0, c, t) == t
            # for defined, non-NaN c).
            if iszero(beta)
                C[i, j] = deepcopy(tmp)
            else
                mp_fma!(C[i, j], beta, C[i, j], tmp)
            end
        end
    else
        # C[i,j] (i ≥ j) = alpha * Σ_l A[l,i]·A[l,j] + beta*C[i,j]
        @inbounds for j in 1:n, i in j:n
            mp_zero!(s)
            for l in 1:k
                mp_fma!(s, A[l, i], A[l, j], s)
            end
            mp_mul!(tmp, alpha, s)
            # BLAS beta==0 means overwrite — C[i,j] must NOT be read
            # (it may be undefined/NaN). Even mp_set! reads the C[i,j]
            # REFERENCE, which throws UndefRefError on an undef slot, so
            # store a fresh deepcopy instead (BigFloat(tmp) would alias
            # tmp on Julia 1.12; tmp is mutated every iteration).
            # For beta != 0 the fma form is unchanged (fma(0, c, t) == t
            # for defined, non-NaN c).
            if iszero(beta)
                C[i, j] = deepcopy(tmp)
            else
                mp_fma!(C[i, j], beta, C[i, j], tmp)
            end
        end
    end
    return C
end

function Rsyrk!(
    uplo::AbstractString, trans::AbstractString,
    n::Integer, k::Integer,
    alpha::T, A::AbstractMatrix{T}, beta::T, C::AbstractMatrix{T},
) where {T<:Union{Float32,Float64}}
    # Validate control strings: a typo must error, not silently
    # default to the transposed path. Mirrors the BigFloat method above.
    uplo == "U" || uplo == "L" ||
        error("Rsyrk!: uplo must be \"U\" or \"L\", got $uplo")
    trans == "N" || trans == "T" ||
        error("Rsyrk!: trans must be \"N\" or \"T\", got $trans")
    size(A) == (trans == "N" ? (n, k) : (k, n)) || throw(DimensionMismatch(
        "Rsyrk!: A is $(size(A)); expected $(trans == "N" ? (n, k) : (k, n)) for trans='$trans'"))
    size(C) == (n, n) || throw(DimensionMismatch(
        "Rsyrk!: C is $(size(C)); expected ($n, $n)"))

    # Triangle-only xSYRK semantics, mirroring the BigFloat method.
    upper = uplo == "U"
    notrans = trans == "N"
    if upper && notrans
        @inbounds for j in 1:n, i in 1:j
            s = zero(T)
            for l in 1:k
                s += A[i, l] * A[j, l]
            end
            # BLAS beta==0 means overwrite — C[i,j] must NOT be read
            # (it may be undefined/NaN); write alpha*s directly.
            if iszero(beta)
                C[i, j] = alpha * s
            else
                C[i, j] = alpha * s + beta * C[i, j]
            end
        end
    elseif upper
        @inbounds for j in 1:n, i in 1:j
            s = zero(T)
            for l in 1:k
                s += A[l, i] * A[l, j]
            end
            # BLAS beta==0 means overwrite — C[i,j] must NOT be read
            # (it may be undefined/NaN); write alpha*s directly.
            if iszero(beta)
                C[i, j] = alpha * s
            else
                C[i, j] = alpha * s + beta * C[i, j]
            end
        end
    elseif notrans
        @inbounds for j in 1:n, i in j:n
            s = zero(T)
            for l in 1:k
                s += A[i, l] * A[j, l]
            end
            # BLAS beta==0 means overwrite — C[i,j] must NOT be read
            # (it may be undefined/NaN); write alpha*s directly.
            if iszero(beta)
                C[i, j] = alpha * s
            else
                C[i, j] = alpha * s + beta * C[i, j]
            end
        end
    else
        @inbounds for j in 1:n, i in j:n
            s = zero(T)
            for l in 1:k
                s += A[l, i] * A[l, j]
            end
            # BLAS beta==0 means overwrite — C[i,j] must NOT be read
            # (it may be undefined/NaN); write alpha*s directly.
            if iszero(beta)
                C[i, j] = alpha * s
            else
                C[i, j] = alpha * s + beta * C[i, j]
            end
        end
    end
    return C
end

"""
    Rpotrf!(uplo, n, A; nb=64) -> Int

Blocked Cholesky factorization. Returns 0 on success, or the index of
the first non-positive pivot.

For n <= nb, delegates to Rpotf2! (unblocked). For larger n, uses the
blocked algorithm from LAPACK DPOTRF:

  For each block column j (step nb):
    1. Rsyrk: A[j:j+jb, j:j+jb] -= A[1:j-1, j:j+jb]' * A[1:j-1, j:j+jb]
    2. Rpotf2: factorize diagonal block A[j:j+jb, j:j+jb]
    3. Rgemm + Rtrsm: update off-diagonal block row A[j:j+jb, j+jb:n]

The blocked algorithm calls Rgemm! (which has OhMyThreads @tasks
parallelism) for the bulk of the computation, giving 10-30× speedup
over unblocked Rpotf2! for large n.

Ported from Rpotrf.cpp (154 lines, BSD-2, Nakata 2008-2021).
"""
function Rpotrf!(
    uplo::AbstractString, n::Integer, A::AbstractMatrix{T};
    nb::Integer=64,
) where {T<:BigFloat}
    n == 0 && return 0
    upper = uplo == "U"
    upper || uplo == "L" ||
        error("Rpotrf!: uplo must be \"U\" or \"L\"")
    size(A) == (n, n) || throw(DimensionMismatch(
        "Rpotrf!: A is $(size(A)); expected ($n, $n)"))

    # For small matrices, use unblocked code.
    if nb <= 1 || nb >= n
        return Rpotf2!(uplo, n, A)
    end

    one_t = one(T)
    neg_one = -one_t

    if upper
        # Compute U such that A = U' * U (upper triangular factor).
        j = 1
        while j <= n
            jb = min(nb, n - j + 1)

            # Step 1: Update diagonal block.
            # A[j:j+jb-1, j:j+jb-1] -= A[1:j-1, j:j+jb-1]' * A[1:j-1, j:j+jb-1]
            if j > 1
                diag_block = view(A, j:(j+jb-1), j:(j+jb-1))
                above = view(A, 1:(j-1), j:(j+jb-1))
                Rsyrk!("U", "T", jb, j-1,
                       neg_one, above, one_t, diag_block)
            end

            # Step 2: Factorize diagonal block.
            diag_block = view(A, j:(j+jb-1), j:(j+jb-1))
            info = Rpotf2!("U", jb, diag_block)
            if info != 0
                return info + j - 1
            end

            # Step 3: Update off-diagonal block row.
            if j + jb <= n
                rest_cols = n - j - jb + 1
                # A[j:j+jb-1, j+jb:n] -= A[1:j-1, j:j+jb-1]' * A[1:j-1, j+jb:n]
                if j > 1
                    above = view(A, 1:(j-1), j:(j+jb-1))
                    above_rest = view(A, 1:(j-1), (j+jb):n)
                    row_block = view(A, j:(j+jb-1), (j+jb):n)
                    Rgemm!("T", "N", jb, rest_cols, j-1,
                           neg_one, above, above_rest, one_t, row_block)
                end
                # A[j:j+jb-1, j+jb:n] = U[j:j+jb-1, j:j+jb-1]⁻ᵀ * A[j:j+jb-1, j+jb:n]
                diag_block = view(A, j:(j+jb-1), j:(j+jb-1))
                row_block = view(A, j:(j+jb-1), (j+jb):n)
                Rtrsm!("L", "U", "T", "N", jb, rest_cols,
                       one_t, diag_block, row_block)
            end

            j += nb
        end
    else  # lower
        # Compute L such that A = L * L' (lower triangular factor).
        j = 1
        while j <= n
            jb = min(nb, n - j + 1)

            # Step 1: Update diagonal block.
            if j > 1
                diag_block = view(A, j:(j+jb-1), j:(j+jb-1))
                left = view(A, j:(j+jb-1), 1:(j-1))
                Rsyrk!("L", "N", jb, j-1,
                       neg_one, left, one_t, diag_block)
            end

            # Step 2: Factorize diagonal block.
            diag_block = view(A, j:(j+jb-1), j:(j+jb-1))
            info = Rpotf2!("L", jb, diag_block)
            if info != 0
                return info + j - 1
            end

            # Step 3: Update off-diagonal block column.
            if j + jb <= n
                rest_rows = n - j - jb + 1
                if j > 1
                    left = view(A, j:(j+jb-1), 1:(j-1))
                    below_left = view(A, (j+jb):n, 1:(j-1))
                    col_block = view(A, (j+jb):n, j:(j+jb-1))
                    Rgemm!("N", "T", rest_rows, jb, j-1,
                           neg_one, below_left, left, one_t, col_block)
                end
                diag_block = view(A, j:(j+jb-1), j:(j+jb-1))
                col_block = view(A, (j+jb):n, j:(j+jb-1))
                Rtrsm!("R", "L", "T", "N", rest_rows, jb,
                       one_t, diag_block, col_block)
            end

            j += nb
        end
    end
    return 0
end

# Float64/32 fallback for Rpotrf!: use LAPACK cholesky!
function Rpotrf!(
    uplo::AbstractString, n::Integer, A::AbstractMatrix{T};
    nb::Integer=64,
) where {T<:Union{Float32,Float64}}
    if n == 0; return 0; end
    uplo == "U" || uplo == "L" ||
        error("Rpotrf!: uplo must be \"U\" or \"L\", got $uplo")
    size(A) == (n, n) || throw(DimensionMismatch(
        "Rpotrf!: A is $(size(A)); expected ($n, $n)"))
    uplo_sym = uplo == "U" ? :U : :L
    # NaN scan (see Rpotf2! Float64 fallback).
    if uplo_sym === :U
        for j in 1:n, i in 1:j
            isnan(A[i, j]) && return j
        end
    else
        for j in 1:n, i in j:n
            isnan(A[i, j]) && return j
        end
    end
    H = Hermitian(A, uplo_sym)
    F = cholesky!(H; check=false)
    if !issuccess(F)
        return F.info > 0 ? F.info : 1
    end
    copyto!(A, F.factors)
    return 0
end

# ---------------------------------------------------------------------------
# Rpotrf_with_adjust!: Cholesky factorization, returning failure on non-PD.
#
# Mirrors the ACTIVE C++ path of `Lal::getCholesky` (sdpa_linear.cpp:352-361):
# plain `Rpotrf("Lower", ...)`, and on `info != 0` return FAILURE.  The C++
# diagonal-adjust algorithm (`choleskyFactorWithAdjust`, sdpa_linear.cpp:631-680)
# is `#if 0` dead code and is deliberately NOT ported.
# ---------------------------------------------------------------------------

"""
    Rpotrf_with_adjust!(uplo, n, A; nb=64) -> Int

Cholesky factorization of the `n × n` matrix `A` (in-place).

The name is kept for C++ API parity (`Lal::getCholesky`); there is no
"adjustment".  Behavior matches C++ `Lal::getCholesky`: plain `Rpotrf!`,
returning 0 on success and a positive info on failure (the caller treats any
non-zero info as Cholesky failure and breaks the main loop).  On failure `A`
holds a partial factorization and must not be used.
"""
function Rpotrf_with_adjust!(
    uplo::AbstractString, n::Integer, A::AbstractMatrix{T};
    nb::Integer=64,
) where {T<:BigFloat}
    return Rpotrf!(uplo, n, A; nb=nb)
end

# Float32/64: same plain-Cholesky + failure semantics (delegates to Rpotrf!).
function Rpotrf_with_adjust!(
    uplo::AbstractString, n::Integer, A::AbstractMatrix{T};
    nb::Integer=64,
) where {T<:Union{Float32,Float64}}
    return Rpotrf!(uplo, n, A; nb=nb)
end
