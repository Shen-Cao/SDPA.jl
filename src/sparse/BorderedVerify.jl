# Independent enclosure of the original Schur action in physical row
# order. The inputs are the stored coefficients and iterate, never the
# rounded assembled matrix or its factors. Each operation is outward
# rounded at guard precision; cancellation cannot turn an unresolved
# interval into a verified direction.
const _BV_Round = Base.MPFR.MPFRRoundingMode
const _BV_DOWN = Base.MPFR.MPFRRoundDown
const _BV_UP = Base.MPFR.MPFRRoundUp
const _BV_NEAREST = Base.MPFR.MPFRRoundNearest
const _BORDERED_MAX_VERIFY_SLOTS = 1_000_000

function _bordered_verify_slots(nrows::Int, dim::Int)
    cap = _BORDERED_MAX_VERIFY_SLOTS
    0 <= dim <= isqrt(cap ÷ 4) && 0 <= nrows <= (cap - 6) ÷ 2 ||
        _bordered_refuse("direction verification exceeds $cap guard-precision slots")
    slots = 4 * dim * dim + 2 * nrows + 6
    slots <= cap || _bordered_refuse("direction verification exceeds $cap guard-precision slots")
    return slots
end

_bordered_verify_dim(blocks) = maximum((b for b in blocks if b >= 2); init=0)


struct _BorderedVerifier
    lo::Vector{BigFloat}
    hi::Vector{BigFloat}
    alo::Matrix{BigFloat}
    ahi::Matrix{BigFloat}
    blo::Matrix{BigFloat}
    bhi::Matrix{BigFloat}
    points::Vector{BigFloat}
    lpscratch::Vector{BigFloat}
    margin::BigFloat
    tol::BigFloat
end

function _BorderedVerifier(::Type{T}, nrows::Int, dim::Int) where {T}
    _bordered_verify_slots(nrows, dim)
    p = T === BigFloat ? precision(BigFloat) : 53
    # Source BigFloats themselves are passed to MPFR without re-rounding.
    # Guard precision controls the tightness, not validity, of the bounds.
    guard = Base.checked_add(Base.checked_mul(p, 2), 64)
    return setprecision(BigFloat, guard) do
        tolerance = zero(BigFloat)
        working_eps = ldexp(one(BigFloat), 1 - p)
        # Even the represented tolerance must be conservative: the
        # mathematical square root need not be representable at guard p.
        ccall((:mpfr_sqrt, Base.MPFR.libmpfr), Cint,
              (Ref{BigFloat}, Ref{BigFloat}, _BV_Round),
              tolerance, working_eps, _BV_DOWN)
        _BorderedVerifier(
            _independent_zeros(BigFloat, nrows), _independent_zeros(BigFloat, nrows),
            _independent_zeros(BigFloat, dim, dim), _independent_zeros(BigFloat, dim, dim),
            _independent_zeros(BigFloat, dim, dim), _independent_zeros(BigFloat, dim, dim),
            _independent_zeros(BigFloat, 2), _independent_zeros(BigFloat, 2), zero(BigFloat),
            tolerance,
        )
    end
end

struct _BorderedSchur{T<:Real}
    factors::_BorderedFactors{T}
    incidence::_BorderedIncidence
    verifier::_BorderedVerifier
    candidate::Vector{T}
end

function _BorderedSchur(F::_BorderedFactors{T}, inc::_BorderedIncidence, blocks) where {T}
    m = length(F.layout.order)
    dim = _bordered_verify_dim(blocks)
    return _BorderedSchur(F, inc, _BorderedVerifier(T, m, dim), _independent_zeros(T, m))
end

@inline _bv_point!(::BigFloat, a::BigFloat) = a
@inline function _bv_point!(out::BigFloat, a::Float64)
    # 53-bit binary data convert exactly at our verification precision.
    ccall((:mpfr_set_d, Base.MPFR.libmpfr), Cint,
          (Ref{BigFloat}, Cdouble, _BV_Round), out, a, _BV_NEAREST)
    return out
end

@inline function _bv_fma!(out, a, b, rnd::_BV_Round)
    ccall((:mpfr_fma, Base.MPFR.libmpfr), Cint,
          (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, _BV_Round),
          out, a, b, out, rnd)
    return nothing
end

@inline function _bv_mul!(out, a, b, rnd::_BV_Round)
    ccall((:mpfr_mul, Base.MPFR.libmpfr), Cint,
          (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, _BV_Round), out, a, b, rnd)
    return nothing
end

@inline function _bv_add!(out, a, b, rnd::_BV_Round)
    ccall((:mpfr_add, Base.MPFR.libmpfr), Cint,
          (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, _BV_Round), out, a, b, rnd)
    return nothing
end

@inline function _bv_sub!(out, a, b, rnd::_BV_Round)
    ccall((:mpfr_sub, Base.MPFR.libmpfr), Cint,
          (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, _BV_Round), out, a, b, rnd)
    return nothing
end

# Add a * [lo,hi] to existing bounds. a is an exact stored point, not an
# interval. Multiplication by a negative point reverses the endpoints.
@inline function _bv_add_scaled!(outlo, outhi, a, lo, hi)
    if a < 0
        lo, hi = hi, lo
    end
    _bv_fma!(outlo, a, lo, _BV_DOWN)
    _bv_fma!(outhi, a, hi, _BV_UP)
    return nothing
end

@inline function _bv_coefficients(f, A::SDPASparseMatrix)
    if A.type === :dense
        @inbounds for j in axes(A.de_ele, 2), i in axes(A.de_ele, 1)
            a = A.de_ele[i, j]
            iszero(a) || f(i, j, a)
        end
    else
        @inbounds for k in 1:A.nonZeroEffect
            a = A.sp_ele[k]
            iszero(a) && continue
            i, j = A.row_index[k], A.column_index[k]
            f(i, j, a)
            i == j || f(j, i, a)
        end
    end
    return nothing
end

function _bv_sdp!(v, entries, x, problem, X, Zinv)
    # Exact gates, not tolerances: a non-finite or asymmetric X / Zinv
    # (e.g. from an asymmetric warm start) declines the direction, which
    # the solve reports as CHOLESKY_FAILURE at the last accepted iterate.
    all(isfinite, X) && all(isfinite, Zinv) &&
        issymmetric(X) && issymmetric(Zinv) || return false
    alo, ahi, blo, bhi = v.alo, v.ahi, v.blo, v.bhi
    a1, a2 = v.points
    n = size(X, 1)
    # U = sum_j x_j A_j.
    @inbounds for j in 1:n, i in 1:n
        mp_zero!(alo[i, j])
        mp_zero!(ahi[i, j])
    end
    for e in entries
        w = _bv_point!(a1, x[e.row])
        _bv_coefficients(problem.A[e.row].SDP_sp_block[e.block_index]) do i, j, val
            a = _bv_point!(a2, val)
            _bv_add_scaled!(alo[i, j], ahi[i, j], a, w, w)
        end
    end

    # V = X U Zinv. Use two alternating pairs of small cone-sized
    # matrices; no row-pair Schur matrix entries are evaluated or stored.
    @inbounds for j in 1:n, i in 1:n
        l, h = blo[i, j], bhi[i, j]
        mp_zero!(l)
        mp_zero!(h)
        for k in 1:n
            iszero(X[i, k]) && continue
            _bv_add_scaled!(l, h, _bv_point!(a1, X[i, k]), alo[k, j], ahi[k, j])
        end
    end
    @inbounds for j in 1:n, i in 1:n
        l, h = alo[i, j], ahi[i, j]
        mp_zero!(l)
        mp_zero!(h)
        for k in 1:n
            iszero(Zinv[k, j]) && continue
            _bv_add_scaled!(l, h, _bv_point!(a1, Zinv[k, j]), blo[i, k], bhi[i, k])
        end
    end

    # Frobenius projections of V. X, Zinv and A_j are symmetric, so
    # transposition and cyclic trace show this action equals the
    # symmetrically stored Schur assembly in exact arithmetic.
    for e in entries
        l, h = v.lo[e.row], v.hi[e.row]
        _bv_coefficients(problem.A[e.row].SDP_sp_block[e.block_index]) do i, j, val
            a = _bv_point!(a2, val)
            _bv_add_scaled!(l, h, a, alo[i, j], ahi[i, j])
        end
    end
    return true
end

function _bv_lp!(v, incidence, x, problem, pt)
    a1, a2 = v.points
    l, h = v.lpscratch
    for (k, entries) in enumerate(incidence.lp)
        isempty(entries) && continue
        mp_zero!(l)
        mp_zero!(h)
        for e in entries
            a = _bv_point!(a1, problem.A[e.row].LP_sp_val[e.coeff_index])
            w = _bv_point!(a2, x[e.row])
            _bv_add_scaled!(l, h, a, w, w)
        end
        for vscale in (pt.xMat.LP_block[k], pt.invzMat.LP_block[k])
            isfinite(vscale) && vscale > 0 || return false
            a = _bv_point!(a1, vscale)
            _bv_mul!(l, l, a, _BV_DOWN)
            _bv_mul!(h, h, a, _BV_UP)
        end
        for e in entries
            a = _bv_point!(a1, problem.A[e.row].LP_sp_val[e.coeff_index])
            _bv_add_scaled!(v.lo[e.row], v.hi[e.row], a, l, h)
        end
    end
    return true
end

function _verify_bordered_direction!(v::_BorderedVerifier, incidence, problem, pt, x, g)
    length(x) == length(g) == length(v.lo) || throw(DimensionMismatch("bordered verifier RHS"))
    all(isfinite, x) && all(isfinite, g) || return false
    gmax = maximum(abs, g; init=zero(eltype(g)))
    iszero(gmax) && return all(iszero, x)
    _zero_bordered_values!(v.lo)
    _zero_bordered_values!(v.hi)
    sdp_l = 0
    for (k, entries) in enumerate(incidence.sdp)
        problem.blockStruct[k] >= 2 || continue
        sdp_l += 1
        isempty(entries) && continue
        X = pt.xMat.SDP_block[sdp_l].de_ele
        Zinv = pt.invzMat.SDP_block[sdp_l].de_ele
        _bv_sdp!(v, entries, x, problem, X, Zinv) || return false
    end
    _bv_lp!(v, incidence, x, problem, pt) || return false

    # Accept only if the entire interval satisfies the RHS-relative
    # target. Round *inward* here so rounding cannot widen acceptance.
    a1, a2 = v.points
    _bv_mul!(v.margin, _bv_point!(a1, gmax), v.tol, _BV_DOWN)
    for i in eachindex(g)
        l, h = v.lo[i], v.hi[i]
        isfinite(l) && isfinite(h) && l <= h || return false
        b = _bv_point!(a1, g[i])
        _bv_sub!(a2, b, v.margin, _BV_UP)
        l >= a2 || return false
        _bv_add!(a2, b, v.margin, _BV_DOWN)
        h <= a2 || return false
    end
    return true
end
