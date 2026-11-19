# src/mplapack/Rgemm.jl
#
# Port of MPLAPACK's Rgemm (matrix-matrix multiply for
# arbitrary-precision floating-point).
#
# Reference: https://github.com/nakatamaho/sdpa-gmp/blob/master/mplapack/Rgemm.cpp (224 lines, LGPLv3)
#            https://github.com/nakatamaho/sdpa-gmp/blob/master/mplapack/Rgemm_NN_omp.cpp (74 lines, BSD-2)
#
# Algorithm (NN case, from Rgemm_NN_omp.cpp):
#   for j in 1:n
#     for l in 1:k
#       temp = alpha * B[l, j]
#       if temp != 0
#         for i in 1:m
#           C[i, j] += temp * A[i, l]   # FMA
#
# The inner loop `C[i,j] += temp * A[i,l]` allocates 2 BigFloats per iter
# with plain operators (the multiplication result and the addition
# result). All four kernels below instead update C in place — the
# rank-1 kernels via `mp_fma!(C[i,j], temp, A[i,l], C[i,j])`, the
# inner-product kernels via `mp_add!(C[i,j], C[i,j], mul_result)` with
# per-task local accumulators — so the BigFloat loops allocate nothing
# per element (MPFR supports full aliasing; no deepcopy is needed).
#
# Threading: OhMyThreads.@tasks for the outer j loop (matching OMP's
# `#pragma omp parallel for`). Each task gets its own accumulators.

using OhMyThreads
using LinearAlgebra: Adjoint, Transpose, UpperTriangular, LowerTriangular

"""
    Rgemm!(transa, transb, m, n, k, alpha, A, B, beta, C)

Perform `C := alpha*op(A)*op(B) + beta*C` where:
- `transa` ∈ {"N", "T", "C"}: op(A) = A (N), Aᵀ (T), or Aᴴ (C; same as T for real)
- `transb` ∈ {"N", "T", "C"}: similar for B
- `A` is `m × k` if transa=="N", else `k × m`
- `B` is `k × n` if transb=="N", else `n × k`
- `C` is `m × n`

For BigFloat, uses zero-allocation inner loop via mp_fma! with per-task
local accumulators. For Float64, dispatches to LinearAlgebra.mul! (LAPACK
is faster than anything we can write in Julia).

Mirrors MPLAPACK's Rgemm at `https://github.com/nakatamaho/sdpa-gmp/blob/master/mplapack/Rgemm.cpp`.
"""
function Rgemm!(
    transa::AbstractString, transb::AbstractString,
    m::Integer, n::Integer, k::Integer,
    alpha::T, A::AbstractMatrix{T}, B::AbstractMatrix{T},
    beta::T, C::AbstractMatrix{T},
) where {T<:BigFloat}
    # Validate control strings: a typo like "n" must error, not be
    # silently treated as transpose. Mirrors Rgemv! (Rblas.jl).
    nota = transa == "N"
    nota || transa == "T" || transa == "C" ||
        error("Rgemm!: transa must be N, T, or C, got $transa")
    notb = transb == "N"
    notb || transb == "T" || transb == "C" ||
        error("Rgemm!: transb must be N, T, or C, got $transb")
    _check_gemm_dims!(transa, transb, m, n, k, A, B, C,
                      !(m == 0 || n == 0 || k == 0 || iszero(alpha)))

    # Quick-return paths (mirror Rgemm.cpp:127-146). On these paths the
    # operands A and B are never referenced (BLAS leaves C := beta*C),
    # so an aliased C is SAFE here — the alias guard below must not fire.
    if m == 0 || n == 0 || ((iszero(alpha) || k == 0) && isone(beta))
        return C
    end
    if iszero(alpha) || k == 0
        if iszero(beta)
            # Zero each element in place: `fill!(C, zero(T))` would store the
            # SAME BigFloat object in every slot, so a later in-place MPFR op
            # (e.g. mp_add!) on one element would corrupt all the others.
            for j in 1:n, i in 1:m
                mp_zero!(C[i, j])
            end
        elseif !isone(beta)
            for j in 1:n, i in 1:m
                C[i, j] = beta * C[i, j]
            end
        end
        return C
    end

    # The in-place kernels overwrite operands mid-computation when
    # C shares memory with A or B (silent corruption). BLAS has the same
    # contract: C must not alias A or B. mightalias also catches
    # view-aliasing, not just ===. The guard sits AFTER the quick-return
    # paths above: those never reference the operands, so an aliased C
    # must be accepted there.
    (Base.mightalias(C, A) || Base.mightalias(C, B)) &&
        throw(ArgumentError("Rgemm!: C must not alias A or B (BLAS convention)"))

    # Dispatch on transposition flags.
    if notb
        if nota
            _Rgemm_NN!(m, n, k, alpha, A, B, beta, C)
        else
            _Rgemm_TN!(m, n, k, alpha, A, B, beta, C)
        end
    else
        if nota
            _Rgemm_NT!(m, n, k, alpha, A, B, beta, C)
        else
            _Rgemm_TT!(m, n, k, alpha, A, B, beta, C)
        end
    end
    return C
end

# Float64 fallback: just use LinearAlgebra.mul! (LAPACK is faster).
function Rgemm!(
    transa::AbstractString, transb::AbstractString,
    m::Integer, n::Integer, k::Integer,
    alpha::T, A::AbstractMatrix{T}, B::AbstractMatrix{T},
    beta::T, C::AbstractMatrix{T},
) where {T<:Union{Float32,Float64}}
    # Validate control strings on the Float64 path too (a typo like
    # "n" must error, not silently select the adjoint branch).
    transa == "N" || transa == "T" || transa == "C" ||
        error("Rgemm!: transa must be N, T, or C, got $transa")
    transb == "N" || transb == "T" || transb == "C" ||
        error("Rgemm!: transb must be N, T, or C, got $transb")
    # Same aliasing contract as the BigFloat method.
    (Base.mightalias(C, A) || Base.mightalias(C, B)) &&
        throw(ArgumentError("Rgemm!: C must not alias A or B (BLAS convention)"))
    _check_gemm_dims!(transa, transb, m, n, k, A, B, C,
                      !(m == 0 || n == 0 || k == 0 || iszero(alpha)))
    opA = transa == "N" ? A : (transa == "T" ? transpose(A) : adjoint(A))
    opB = transb == "N" ? B : (transb == "T" ? transpose(B) : adjoint(B))
    mul!(C, opA, opB, alpha, beta)
    return C
end

# BLAS-storage dimension validation shared by both Rgemm! methods.
# Convention: transa="N" → A is m×k, else A is k×m; transb="N" → B is k×n,
# else B is n×k; C is always m×n.
function _check_gemm_dims!(
    transa::AbstractString, transb::AbstractString,
    m::Integer, n::Integer, k::Integer,
    A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix,
    check_ab::Bool,
)
    size(C) == (m, n) || throw(DimensionMismatch(
        "Rgemm!: C is $(size(C)); expected ($m, $n)"))
    # A and B are only referenced when the kernels actually run: the BLAS
    # quick-return cases (m=0, n=0, k=0, alpha=0) never touch them, so
    # their shapes are unconstrained there.
    check_ab || return nothing
    size_a = transa == "N" ? (m, k) : (k, m)
    size_b = transb == "N" ? (k, n) : (n, k)
    size(A) == size_a || throw(DimensionMismatch(
        "Rgemm!: A is $(size(A)); expected $size_a for transa='$transa'"))
    size(B) == size_b || throw(DimensionMismatch(
        "Rgemm!: B is $(size(B)); expected $size_b for transb='$transb'"))
    return nothing
end

# ---------------------------------------------------------------------------
# NN: C = alpha * A * B + beta * C
# Ported from Rgemm_NN_omp.cpp (BSD-2 license, Nakata 2010-2012).
#
# Algorithm:
#   for j in 1:n:
#     for l in 1:k:
#       temp = alpha * B[l, j]
#       if temp != 0:
#         for i in 1:m:
#           C[i, j] += temp * A[i, l]
#
# Julia port notes:
#   - Inner loop updates C in place: mp_fma!(C[i,j], temp, A[i,l], C[i,j])
#     (MPFR supports full aliasing; no deepcopy is needed)
#   - OhMyThreads.@tasks for outer j loop parallelization
# ---------------------------------------------------------------------------
# Cutoff below which the serial loop beats @tasks (empirically calibrated:
# under this m*n*k volume the @tasks scheduling overhead outweighs the
# kernel work).
const _RGEMM_SERIAL_THRESHOLD = 4096  # m*n*k below this → serial

function _Rgemm_NN!(
    m::Integer, n::Integer, k::Integer,
    alpha::T, A::AbstractMatrix{T}, B::AbstractMatrix{T},
    beta::T, C::AbstractMatrix{T},
) where {T<:BigFloat}
    # Zero-alloc direct-write. For large matrices, parallelize over j columns.
    # Small matrices use serial loop (avoids @tasks overhead).
    if m * n * k ≤ _RGEMM_SERIAL_THRESHOLD
        # NOTE: serial-branch accumulators are named *_{serial} on purpose:
        # the parallel @tasks branch below assigns `temp`/`s`/`mul_result`,
        # and a same-name assignment in the enclosing function scope would
        # be CAPTURED (shared) by every task closure, racing on one
        # BigFloat object. Distinct names keep the @tasks-body locals
        # genuinely per-task.
        temp_serial = zero(BigFloat)
        @inbounds for j in 1:n
            if iszero(beta)
                for i in 1:m; mp_zero!(C[i, j]); end
            elseif !isone(beta)
                for i in 1:m; mp_mul!(C[i, j], beta, C[i, j]); end
            end
            for l in 1:k
                mp_mul!(temp_serial, alpha, B[l, j])
                iszero(temp_serial) && continue
                for i in 1:m; mp_fma!(C[i, j], temp_serial, A[i, l], C[i, j]); end
            end
        end
    else
        # Parallel path: each task handles a different column j.
        # NOTE: @inbounds must go INSIDE the @tasks body: the macro
        # hoists the body into a separate mapping_function, so an outer
        # `@inbounds @tasks` would not propagate into it.
        @tasks for j in 1:n
            temp = zero(BigFloat)
            @inbounds begin
                if iszero(beta)
                    for i in 1:m; mp_zero!(C[i, j]); end
                elseif !isone(beta)
                    for i in 1:m; mp_mul!(C[i, j], beta, C[i, j]); end
                end
                for l in 1:k
                    mp_mul!(temp, alpha, B[l, j])
                    iszero(temp) && continue
                    for i in 1:m; mp_fma!(C[i, j], temp, A[i, l], C[i, j]); end
                end
            end
        end
    end
    return C
end

# ---------------------------------------------------------------------------
# TN: C = alpha * Aᵀ * B + beta * C
# Uses inner-product formulation (no rank-1 update):
#   for j in 1:n, for i in 1:m:
#     temp = sum_l A[l, i] * B[l, j]
#     C[i, j] = alpha * temp + beta * C[i, j]
# ---------------------------------------------------------------------------
function _Rgemm_TN!(
    m::Integer, n::Integer, k::Integer,
    alpha::T, A::AbstractMatrix{T}, B::AbstractMatrix{T},
    beta::T, C::AbstractMatrix{T},
) where {T<:BigFloat}
    if m * n * k ≤ _RGEMM_SERIAL_THRESHOLD
        # NOTE: `s`/`mul_result` are only assigned inside the @tasks body
        # below; naming the serial-branch accumulators *_{serial} prevents
        # them from being captured (shared) by the task closures.
        s_serial = zero(BigFloat)
        mul_result_serial = zero(BigFloat)
        for j in 1:n
            if iszero(beta)
                for i in 1:m; mp_zero!(C[i, j]); end
            elseif !isone(beta)
                for i in 1:m; mp_mul!(C[i, j], beta, C[i, j]); end
            end
            for i in 1:m
                mp_zero!(s_serial)
                for l in 1:k; mp_fma!(s_serial, A[l, i], B[l, j], s_serial); end
                mp_mul!(mul_result_serial, alpha, s_serial)
                mp_add!(C[i, j], C[i, j], mul_result_serial)
            end
        end
    else
        @tasks for j in 1:n
            s = zero(BigFloat)
            mul_result = zero(BigFloat)
            @inbounds begin
                if iszero(beta)
                    for i in 1:m; mp_zero!(C[i, j]); end
                elseif !isone(beta)
                    for i in 1:m; mp_mul!(C[i, j], beta, C[i, j]); end
                end
                for i in 1:m
                    mp_zero!(s)
                    for l in 1:k; mp_fma!(s, A[l, i], B[l, j], s); end
                    mp_mul!(mul_result, alpha, s)
                    mp_add!(C[i, j], C[i, j], mul_result)
                end
            end
        end
    end
    return C
end

# ---------------------------------------------------------------------------
# NT: C = alpha * A * Bᵀ + beta * C
# Rank-1 update: for j, for l: temp = alpha * B[j, l]; for i: C[i,j] += temp * A[i, l]
# ---------------------------------------------------------------------------
function _Rgemm_NT!(
    m::Integer, n::Integer, k::Integer,
    alpha::T, A::AbstractMatrix{T}, B::AbstractMatrix{T},
    beta::T, C::AbstractMatrix{T},
) where {T<:BigFloat}
    if m * n * k ≤ _RGEMM_SERIAL_THRESHOLD
        # See _Rgemm_NN!: *_{serial} names keep the @tasks body's `temp`
        # a genuine per-task local.
        temp_serial = zero(BigFloat)
        @inbounds for j in 1:n
            if iszero(beta)
                for i in 1:m; mp_zero!(C[i, j]); end
            elseif !isone(beta)
                for i in 1:m; mp_mul!(C[i, j], beta, C[i, j]); end
            end
            for l in 1:k
                mp_mul!(temp_serial, alpha, B[j, l])
                iszero(temp_serial) && continue
                for i in 1:m; mp_fma!(C[i, j], temp_serial, A[i, l], C[i, j]); end
            end
        end
    else
        @tasks for j in 1:n
            temp = zero(BigFloat)
            @inbounds begin
                if iszero(beta)
                    for i in 1:m; mp_zero!(C[i, j]); end
                elseif !isone(beta)
                    for i in 1:m; mp_mul!(C[i, j], beta, C[i, j]); end
                end
                for l in 1:k
                    mp_mul!(temp, alpha, B[j, l])
                    iszero(temp) && continue
                    for i in 1:m; mp_fma!(C[i, j], temp, A[i, l], C[i, j]); end
                end
            end
        end
    end
    return C
end

# ---------------------------------------------------------------------------
# TT: C = alpha * Aᵀ * Bᵀ + beta * C
# Inner-product formulation.
# ---------------------------------------------------------------------------
function _Rgemm_TT!(
    m::Integer, n::Integer, k::Integer,
    alpha::T, A::AbstractMatrix{T}, B::AbstractMatrix{T},
    beta::T, C::AbstractMatrix{T},
) where {T<:BigFloat}
    if m * n * k ≤ _RGEMM_SERIAL_THRESHOLD
        # See _Rgemm_TN!: *_{serial} names keep the @tasks body's
        # `s`/`mul_result` genuine per-task locals.
        s_serial = zero(BigFloat)
        mul_result_serial = zero(BigFloat)
        for j in 1:n
            if iszero(beta)
                for i in 1:m; mp_zero!(C[i, j]); end
            elseif !isone(beta)
                for i in 1:m; mp_mul!(C[i, j], beta, C[i, j]); end
            end
            for i in 1:m
                mp_zero!(s_serial)
                for l in 1:k; mp_fma!(s_serial, A[l, i], B[j, l], s_serial); end
                mp_mul!(mul_result_serial, alpha, s_serial)
                mp_add!(C[i, j], C[i, j], mul_result_serial)
            end
        end
    else
        @tasks for j in 1:n
            s = zero(BigFloat)
            mul_result = zero(BigFloat)
            @inbounds begin
                if iszero(beta)
                    for i in 1:m; mp_zero!(C[i, j]); end
                elseif !isone(beta)
                    for i in 1:m; mp_mul!(C[i, j], beta, C[i, j]); end
                end
                for i in 1:m
                    mp_zero!(s)
                    for l in 1:k; mp_fma!(s, A[l, i], B[j, l], s); end
                    mp_mul!(mul_result, alpha, s)
                    mp_add!(C[i, j], C[i, j], mul_result)
                end
            end
        end
    end
    return C
end
