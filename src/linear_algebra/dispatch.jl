# Linear-algebra dispatch helpers — BlasFloat vs generic paths.
#
# Mirrors Hypatia's `dense.jl` pattern at
# Hypatia.jl dense linear algebra (adapted).
#
# Julia's stdlib `LinearAlgebra` only ships LAPACK factorizations for
# `BlasFloat` (Float32/Float64/ComplexF32/ComplexF64). The
# `GenericLinearAlgebra` package (loaded via `using` at module level)
# extends `cholesky!`, `eigen`, `lu`, etc. to `BigFloat`. This file
# provides thin dispatch wrappers so the rest of the solver can call
# the same function name regardless of `T`.
#
# Load-bearing: every BigFloat IPM operation goes through these helpers.
# Without `using GenericLinearAlgebra`, the BigFloat paths would
# `MethodError` because Julia would try to call LAPACK on a `Matrix{BigFloat}`.
using LinearAlgebra: BlasFloat, BlasReal, LAPACK
using GenericLinearAlgebra

const RealOrComplex{T<:Real} = Union{T, Complex{T}}

"""
    update_eigen!(X::Matrix{T}) where T

In-place symmetric eigendecomposition. The upper triangle of `X` is
overwritten with the eigenvectors; the eigenvalues are returned.

For `BlasReal` (Float32/Float64): dispatches to LAPACK `syev!`. The
signature is deliberately `BlasReal`, not `BlasFloat`: `syev!` is
real-only, so a `Complex` matrix would hit this method and `MethodError`
inside LAPACK at runtime — narrowing makes complex input fail cleanly
at dispatch. For other real types (e.g. `BigFloat`): dispatches to
`GenericLinearAlgebra.eigen(Hermitian(X, :U))`.

Mirrors Hypatia `update_eigen!` at `dense.jl:59-67`.
"""
function update_eigen!(X::Matrix{<:BlasReal})
    return LAPACK.syev!('V', 'U', X)[1]
end

function update_eigen!(X::Matrix{T}) where {T<:Real}
    F = eigen(Hermitian(X, :U))
    # GLA's `eigen` for BigFloat reference-copies X's elements into its
    # workspace (`eigencopy_oftype`), so `F.values` may ALIAS X's own element
    # objects (observed for already-tridiagonal inputs). Snapshot the values
    # BEFORE the in-place value-copy below, which mutates those objects.
    vals = _copy_independent(F.values)
    # Value copy (BigFloat): default `copyto!` copies REFERENCES
    # (X[i] === F.vectors[i]), aliasing X's elements to a dying workspace.
    # `_copyto_independent!` mutates X's existing elements via mp_set!
    # (in-place value copy); it dispatches to `copyto!` for isbits T.
    _copyto_independent!(X, F.vectors)
    return vals
end

"""
    posdef_fact!(A::Symmetric{T, Matrix{T}}) where T

Symmetric positive-definite Cholesky factorization, in-place. Returns
a `Cholesky{T}` object; check `issuccess` to detect non-PD inputs.

For all real `T` (including `BigFloat`), dispatches to `cholesky!(A,
check=false)`. The `GenericLinearAlgebra` package's `cholesky!` for
`Hermitian{BigFloat,Matrix{BigFloat}}` is what makes the BigFloat path
work — it's a pure-Julia implementation that does NOT call LAPACK.

Mirrors Hypatia `posdef_fact!` at `dense.jl:226`.
"""
function posdef_fact!(A::Symmetric{T,Matrix{T}}) where {T<:Real}
    return cholesky!(A; check=false)
end

"""
    symm_fact!(A::Symmetric{T, Matrix{T}}) where T

Symmetric indefinite factorization. Returns a `BunchKaufman` (for
`BlasReal`) or `LU` (for other real types like `BigFloat`).

The LU fallback for `BigFloat` exists because `bunchkaufman!` is
LAPACK-only. Use this only on matrices that may be indefinite; for
known-PD matrices use `posdef_fact!` (faster and more numerically
stable).

Mirrors Hypatia `symm_fact!` at `dense.jl:213-216`.
"""
function symm_fact!(A::Symmetric{T,Matrix{T}}) where {T<:BlasReal}
    return bunchkaufman!(A, true; check=false)
end

function symm_fact!(A::Symmetric{T,Matrix{T}}) where {T<:Real}
    return lu!(A; check=false)
end

"""
    increase_diag!(A::Matrix{T}) where T

Multiply the diagonal of `A` by `1 + 1e-5` (with a floor at
`1000 * eps(T)`). Used to recover from non-positive-definite inputs
in late IPM iterations where intermediate matrices can lose
positive-definiteness due to round-off.

Mirrors Hypatia `increase_diag!` at `dense.jl:218-226`.
"""
function increase_diag!(A::Matrix{T}) where {T<:Real}
    diag_pert = 1 + T(1e-5)
    diag_min = 1000 * eps(T)
    @inbounds for j in 1:size(A, 1)
        A[j, j] = diag_pert * max(A[j, j], diag_min)
    end
    return A
end

