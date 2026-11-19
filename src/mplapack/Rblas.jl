# src/mplapack/Rblas.jl
#
# Port of MPLAPACK BLAS Level 1 operations for BigFloat.
#
# References:
#   https://github.com/nakatamaho/sdpa-gmp/blob/master/mplapack/Raxpy_omp.cpp (77 lines, BSD-2, Nakata 2010-2012)
#   https://github.com/nakatamaho/sdpa-gmp/blob/master/mplapack/Rdot_serial.cpp (63 lines, BSD-2, Nakata)
#   https://github.com/nakatamaho/sdpa-gmp/blob/master/mplapack/Rgemv.cpp (204 lines, LGPLv3, Nakata 2008)
#   https://github.com/nakatamaho/sdpa-gmp/blob/master/mplapack/Rscal.cpp (67 lines, LGPLv3, Nakata 2008)
#
# These are standalone BLAS Level 1 building blocks. They are NOT on the
# critical hot path of the current solver (Rpotf2! inlines its own dot
# products).
#
# For BigFloat, the inner loops use mp_fma!/mp_mul! for zero-allocation
# accumulation. Array writes use the replace-reference pattern (=) which
# is allocation-safe. For Float64/32, Julia's native arithmetic is used
# (already FMA-optimized by LLVM).

"""
    Raxpy!(n, da, dx, incx, dy, incy)

BLAS AXPY: `dy[i] += da * dx[i]` for i=1..n with strides.
Ported from Raxpy_omp.cpp.
"""
function Raxpy!(
    n::Integer, da::T, dx::AbstractVector{T}, incx::Integer,
    dy::AbstractVector{T}, incy::Integer,
) where {T<:BigFloat}
    n <= 0 && return dy
    iszero(da) && return dy
    ix = incx < 0 ? (-n + 1) * incx + 1 : 1
    iy = incy < 0 ? (-n + 1) * incy + 1 : 1
    tmp = zero(BigFloat)
    @inbounds for _ in 1:n
        # dy[iy] += da * dx[ix]
        mp_mul!(tmp, da, dx[ix])
        dy[iy] += tmp  # replace-reference (1 alloc, safe)
        ix += incx
        iy += incy
    end
    return dy
end

function Raxpy!(
    n::Integer, da::T, dx::AbstractVector{T}, incx::Integer,
    dy::AbstractVector{T}, incy::Integer,
) where {T<:Union{Float32,Float64}}
    n <= 0 && return dy
    iszero(da) && return dy
    ix = incx < 0 ? (-n + 1) * incx + 1 : 1
    iy = incy < 0 ? (-n + 1) * incy + 1 : 1
    @inbounds for _ in 1:n
        dy[iy] += da * dx[ix]
        ix += incx
        iy += incy
    end
    return dy
end

"""
    Rdot(n, dx, incx, dy, incy) -> T

BLAS DOT: `sum_i dx[i] * dy[i]` for i=1..n with strides.
Uses mp_fma! for zero-allocation accumulation (BigFloat).
Ported from Rdot_serial.cpp.
"""
function Rdot(
    n::Integer, dx::AbstractVector{T}, incx::Integer,
    dy::AbstractVector{T}, incy::Integer,
) where {T<:BigFloat}
    n <= 0 && return zero(T)
    ix = incx < 0 ? (-n + 1) * incx + 1 : 1
    iy = incy < 0 ? (-n + 1) * incy + 1 : 1
    s = zero(BigFloat)
    @inbounds for _ in 1:n
        # s += dx[ix] * dy[iy]
        mp_fma!(s, dx[ix], dy[iy], s)
        ix += incx
        iy += incy
    end
    return s
end

function Rdot(
    n::Integer, dx::AbstractVector{T}, incx::Integer,
    dy::AbstractVector{T}, incy::Integer,
) where {T<:Union{Float32,Float64}}
    n <= 0 && return zero(T)
    ix = incx < 0 ? (-n + 1) * incx + 1 : 1
    iy = incy < 0 ? (-n + 1) * incy + 1 : 1
    s = zero(T)
    @inbounds for _ in 1:n
        s += dx[ix] * dy[iy]
        ix += incx
        iy += incy
    end
    return s
end

# Shape/stride validation shared by both Rgemv! methods. BLAS
# convention: with stride s the accessed indices span 1 .. 1+(len-1)*abs(s),
# starting at the far end for negative strides. This port's convention is
# that A is always stored m×n and transa transposes it logically.
function _check_gemv_dims!(
    transa::AbstractString, m::Integer, n::Integer,
    A::AbstractMatrix, lda::Integer,
    x::AbstractVector, incx::Integer, y::AbstractVector, incy::Integer,
)
    nota = transa == "N"
    len_in = nota ? n : m
    len_out = nota ? m : n
    size(A) == (m, n) ||
        throw(DimensionMismatch(
            "Rgemv!: A is $(size(A)); expected ($m, $n) for transa='$transa'"))
    lda >= max(1, m) ||
        throw(ArgumentError("Rgemv!: lda=$lda must be >= $(max(1, m))"))
    incx == 0 && throw(ArgumentError("Rgemv!: incx must be nonzero"))
    incy == 0 && throw(ArgumentError("Rgemv!: incy must be nonzero"))
    length(x) >= 1 + (len_in - 1) * abs(incx) ||
        throw(DimensionMismatch(
            "Rgemv!: x length $(length(x)) is too short for incx=$incx (need >= $(1 + (len_in - 1) * abs(incx)))"))
    length(y) >= 1 + (len_out - 1) * abs(incy) ||
        throw(DimensionMismatch(
            "Rgemv!: y length $(length(y)) is too short for incy=$incy (need >= $(1 + (len_out - 1) * abs(incy)))"))
    return nothing
end

"""
    Rgemv!(transa, m, n, alpha, A, lda, x, incx, beta, y, incy)

BLAS GEMV: `y = alpha * op(A) * x + beta * y` where op(A) is A (transa="N"),
Aᵀ (transa="T"), or Aᴴ (transa="C"; same as T for real).
In this port A is always stored m×n and transa transposes it logically
(validated by `_check_gemv_dims!`).
Ported from Rgemv.cpp.

`lda` is accepted for BLAS signature parity only: the kernels index `A`
directly (`A[i, j]`), so the effective leading dimension is always
`size(A, 1)` — the size check in `_check_gemv_dims!` enforces
`size(A) == (m, n)`, so `lda` must equal `max(1, m)`; passing a larger
`lda` (padded BLAS column storage) is not supported.
"""
function Rgemv!(
    transa::AbstractString, m::Integer, n::Integer,
    alpha::T, A::AbstractMatrix{T}, lda::Integer,
    x::AbstractVector{T}, incx::Integer,
    beta::T, y::AbstractVector{T}, incy::Integer,
) where {T<:BigFloat}
    nota = transa == "N"
    nota || transa == "T" || transa == "C" ||
        error("Rgemv!: transa must be N, T, or C")
    _check_gemv_dims!(transa, m, n, A, lda, x, incx, y, incy)
    # Same aliasing contract as Rgemm!: BLAS GEMV requires
    # y not to overlap x or A; the in-place loops above silently corrupt
    # results otherwise. mightalias also catches view-aliasing.
    (Base.mightalias(y, x) || Base.mightalias(y, A)) &&
        throw(ArgumentError("Rgemv!: y must not alias x or A (BLAS convention)"))

    # Output length (BLAS GEMV): m for "N", n for "T"/"C". The negative-
    # stride START index must be derived from THIS length, not max(m, n):
    # on a non-square system (m != n) max() starts the reversed write at
    # the wrong (out-of-range) slot.
    len_out = nota ? m : n

    # Scale y by beta.
    if iszero(beta)
        iy = incy < 0 ? (1 - len_out) * incy + 1 : 1
        @inbounds for _ in 1:len_out
            y[iy] = zero(T)
            iy += incy
        end
    elseif !isone(beta)
        iy = incy < 0 ? (1 - len_out) * incy + 1 : 1
        @inbounds for _ in 1:len_out
            # Use operate_to! (mpfr_mul ccall) for in-place beta scaling.
            # operate_to! supports aliasing (output == input).
            MutableArithmetics.operate_to!(y[iy], *, beta, y[iy])
            iy += incy
        end
    end

    s = zero(BigFloat)
    tmp = zero(BigFloat)

    if nota
        # Form y = alpha * A * x + beta * y
        # y[i] = alpha * sum_j A[i,j] * x[1+(j-1)*incx] + beta * y[i]
        iy = incy < 0 ? (1 - len_out) * incy + 1 : 1
        # Reference BLAS: for negative incx the source x is read from
        # the far end — start index kx = 1 - (n-1)*incx — NOT from index 1,
        # which walked out of bounds (index 0 and below) from the second
        # element on. For positive incx, kx = 1 and the formula reduces to
        # the original 1 + (j-1)*incx.
        kx = incx < 0 ? (1 - n) * incx + 1 : 1
        @inbounds for i in 1:m
            mp_zero!(s)
            for j in 1:n
                mp_fma!(s, A[i, j], x[kx + (j - 1) * incx], s)
            end
            mp_mul!(tmp, alpha, s)
            y[iy] += tmp
            iy += incy
        end
    else
        # Form y = alpha * Aᵀ * x + beta * y
        # y[j] = alpha * sum_i A[i,j] * x[kx+(i-1)*incx] + beta * y[j]
        ix = incx < 0 ? (1 - m) * incx + 1 : 1
        tmp_a = zero(BigFloat)
        @inbounds for i in 1:m
            mp_mul!(tmp_a, alpha, x[ix])
            iszero(tmp_a) || begin
                iy = incy < 0 ? (1 - len_out) * incy + 1 : 1
                for j in 1:n
                    mp_mul!(tmp, tmp_a, A[i, j])
                    y[iy] += tmp
                    iy += incy
                end
            end
            ix += incx
        end
    end
    return y
end

function Rgemv!(
    transa::AbstractString, m::Integer, n::Integer,
    alpha::T, A::AbstractMatrix{T}, lda::Integer,
    x::AbstractVector{T}, incx::Integer,
    beta::T, y::AbstractVector{T}, incy::Integer,
) where {T<:Union{Float32,Float64}}
    nota = transa == "N"
    nota || transa == "T" || transa == "C" ||
        error("Rgemv!: transa must be N, T, or C")
    _check_gemv_dims!(transa, m, n, A, lda, x, incx, y, incy)
    # Same aliasing contract as Rgemm! on the Float64 path.
    (Base.mightalias(y, x) || Base.mightalias(y, A)) &&
        throw(ArgumentError("Rgemv!: y must not alias x or A (BLAS convention)"))

    # Output length (BLAS GEMV): m for "N", n for "T"/"C". The negative-
    # stride START index must be derived from THIS length, not max(m, n)
    # (same convention as the BigFloat branch).
    len_out = nota ? m : n

    # Scale y by beta, honoring incy.
    if iszero(beta)
        iy = incy < 0 ? (1 - len_out) * incy + 1 : 1
        @inbounds for _ in 1:len_out
            y[iy] = zero(T)
            iy += incy
        end
    elseif !isone(beta)
        iy = incy < 0 ? (1 - len_out) * incy + 1 : 1
        @inbounds for _ in 1:len_out
            y[iy] *= beta
            iy += incy
        end
    end

    if nota
        # Form y = alpha * A * x + beta * y
        # y[i] = alpha * sum_j A[i,j] * x[1+(j-1)*incx] + beta * y[i]
        iy = incy < 0 ? (1 - len_out) * incy + 1 : 1
        @inbounds for i in 1:m
            s = zero(T)
            ix = incx < 0 ? (1 - n) * incx + 1 : 1
            for j in 1:n
                s += A[i, j] * x[ix]
                ix += incx
            end
            y[iy] += alpha * s
            iy += incy
        end
    else
        # Form y = alpha * Aᵀ * x + beta * y
        # y[j] = alpha * sum_i A[i,j] * x[1+(i-1)*incx] + beta * y[j]
        ix = incx < 0 ? (1 - m) * incx + 1 : 1
        @inbounds for i in 1:m
            tmp = alpha * x[ix]
            if !iszero(tmp)
                iy = incy < 0 ? (1 - len_out) * incy + 1 : 1
                for j in 1:n
                    y[iy] += tmp * A[i, j]
                    iy += incy
                end
            end
            ix += incx
        end
    end
    return y
end

"""
    Rscal!(n, da, dx, incx)

BLAS SCAL: `dx[i] *= da` for i=1..n with stride.
Ported from Rscal.cpp.
"""
function Rscal!(
    n::Integer, da::T, dx::AbstractVector{T}, incx::Integer,
) where {T<:BigFloat}
    n <= 0 && return dx
    isone(da) && return dx
    ix = incx < 0 ? (-n + 1) * incx + 1 : 1
    @inbounds for _ in 1:n
        # Use operate_to! (calls mpfr_mul ccall) for in-place scaling.
        # operate_to! supports aliasing (output == input).
        MutableArithmetics.operate_to!(dx[ix], *, da, dx[ix])
        ix += incx
    end
    return dx
end

function Rscal!(
    n::Integer, da::T, dx::AbstractVector{T}, incx::Integer,
) where {T<:Union{Float32,Float64}}
    n <= 0 && return dx
    isone(da) && return dx
    ix = incx < 0 ? (-n + 1) * incx + 1 : 1
    @inbounds for _ in 1:n
        dx[ix] *= da
        ix += incx
    end
    return dx
end

"""
    Rcopy!(n, dx, incx, dy, incy)

BLAS COPY: `dy[i] = dx[i]` for i=1..n with strides.
Ported from MPLAPACK Rcopy.cpp.
"""
function Rcopy!(
    n::Integer, dx::AbstractVector{T}, incx::Integer,
    dy::AbstractVector{T}, incy::Integer,
) where {T<:BigFloat}
    n <= 0 && return dy
    ix = incx < 0 ? (-n + 1) * incx + 1 : 1
    iy = incy < 0 ? (-n + 1) * incy + 1 : 1
    @inbounds for _ in 1:n
        # Value-copy: `dy[iy] = dx[ix]` would store the SAME BigFloat object
        # in both vectors, so an in-place MPFR op on dy[iy] would
        # corrupt dx[ix].
        mp_set!(dy[iy], dx[ix])
        ix += incx
        iy += incy
    end
    return dy
end

function Rcopy!(
    n::Integer, dx::AbstractVector{T}, incx::Integer,
    dy::AbstractVector{T}, incy::Integer,
) where {T<:Real}
    n <= 0 && return dy
    ix = incx < 0 ? (-n + 1) * incx + 1 : 1
    iy = incy < 0 ? (-n + 1) * incy + 1 : 1
    @inbounds for _ in 1:n
        dy[iy] = dx[ix]
        ix += incx
        iy += incy
    end
    return dy
end
