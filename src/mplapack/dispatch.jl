# src/mplapack/dispatch.jl
#
# Dispatching matmul that picks Rgemm! for BigFloat (where Rgemm is
# ~10x faster than LinearAlgebra.mul!) and LinearAlgebra.mul! for
# Float64 (where LAPACK is faster).
#
# Only the plain matrix-matrix case is dispatched. Specialized forms
# (transpose, triangular, adjoint) still use LinearAlgebra.mul! because
# Rgemm! handles them but the call sites pass SubArray/Adjoint wrappers
# that would need explicit transposition flags.

"""
    sdpa_mul!(C, A, B)
    sdpa_mul!(C, A, B, alpha, beta)

In-place matrix multiply: `C = A * B` (3-arg) or `C = alpha * A * B +
beta * C` (5-arg), with type-based dispatch:
- BigFloat: dispatches to `Rgemm!("N", "N", ...)` (zero-allocation inner
  loop via mp_fma!, OhMyThreads parallelization).
- Other (Float64, Float32): dispatches to `LinearAlgebra.mul!(C, A, B, alpha, beta)`
  (LAPACK is faster than anything we can write in Julia for Float64).
"""
@inline function sdpa_mul!(
    C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T},
) where {T<:BigFloat}
    m, n = size(C)
    k = size(A, 2)
    Rgemm!("N", "N", m, n, k, one(T), A, B, zero(T), C)
    return C
end

@inline function sdpa_mul!(
    C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T},
) where {T<:Union{Float32,Float64}}
    mul!(C, A, B)
    return C
end

# The 5-argument form is retained for explicit alpha/beta calls.
@inline function sdpa_mul!(
    C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T},
    alpha::T, beta::T,
) where {T<:BigFloat}
    m, n = size(C)
    k = size(A, 2)
    Rgemm!("N", "N", m, n, k, alpha, A, B, beta, C)
    return C
end

@inline function sdpa_mul!(
    C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T},
    alpha::T, beta::T,
) where {T<:Union{Float32,Float64}}
    mul!(C, A, B, alpha, beta)
    return C
end
