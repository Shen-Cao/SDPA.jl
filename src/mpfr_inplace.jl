# src/mpfr_inplace.jl
#
# Zero-allocation BigFloat arithmetic wrappers for hot loops.
#
# Most operations are imported from MutableArithmetics.jl (MPL-2.0,
# jump-dev), which already implements the `operate_to!`/`operate!` API for
# BigFloat via direct `ccall` to libmpfr. We add short-name wrappers
# (`mp_add!`, `mp_mul!`, `mp_fma!`, etc.) for hot-loop readability, and
# fill in the small gaps (div, sqrt, neg, set) that MA does not wrap.
#
# Why zero-allocation matters
# --------------------------
# Each `a + b` for BigFloat allocates a fresh 88-byte BigFloat object.
# In hot loops these short-lived objects make GC pressure dominate wall
# time, hence the zero-allocation kernels below. Wrapping the in-place
# `ccall` directly eliminates these allocations.
#
# Thread safety
# -------------
# MPFR built with `--enable-thread-safe` (Julia's bundled libmpfr is, see
# `mpfr_buildopt_tls_p()`) supports concurrent calls on disjoint
# `BigFloat` objects from different threads. Per-thread state (precision,
# rounding mode, caches) is stored in TLS. `setprecision(BigFloat, n) do
# ... end` on Julia ≥ 1.12 uses a ScopedValue (task-local, thread-safe).
#
# Aliasing
# --------
# All MPFR functions used here support full aliasing: `mpfr_add(x, x, y,
# r)` is well-defined. We rely on this for `mp_fma!(s, a, b, s)` which
# accumulates into `s`.

using MutableArithmetics

# Re-export the MA API verbatim so callers can use either spelling.
export operate_to!, operate!, mutable_copy

# ---------------------------------------------------------------------------
# Short-name wrappers for the operations MA already implements.
# These are zero-cost: just rename + `@inline`.
# ---------------------------------------------------------------------------

# mp_add!(r, a, b)  →  r = a + b
@inline mp_add!(r::BigFloat, a::BigFloat, b::BigFloat) =
    operate_to!(r, +, a, b)

# mp_sub!(r, a, b)  →  r = a - b
@inline mp_sub!(r::BigFloat, a::BigFloat, b::BigFloat) =
    operate_to!(r, -, a, b)

# mp_mul!(r, a, b)  →  r = a * b
@inline mp_mul!(r::BigFloat, a::BigFloat, b::BigFloat) =
    operate_to!(r, *, a, b)

# mp_fma!(r, x, y, w)  →  r = x*y + w   (single rounding)
@inline mp_fma!(r::BigFloat, x::BigFloat, y::BigFloat, w::BigFloat) =
    operate_to!(r, Base.fma, x, y, w)

# The effective MPFR rounding mode is task-local on
# Julia >= 1.12 (Base.MPFR stores it in a ScopedValue and reads it via
# rounding_raw, base/mpfr.jl:246). Passing the bare global
# ROUNDING_MODE[] ignored scoped setrounding do-blocks (verified: mp_div!
# disregarded RoundDown while x/y honored it). All ccalls below use this
# accessor so the in-place kernels match the Base operators' semantics.
@inline _mpfr_rounding() = Base.MPFR.rounding_raw(BigFloat)

# mp_fms!(r, x, y, z)  →  r = x*y - z   (single rounding)
# Implemented via fma with negated z (no allocation because we operate
# in-place on a pre-allocated buffer the caller passes).
# We provide the direct ccall below to avoid allocating a negated z.
@inline function mp_fms!(r::BigFloat, x::BigFloat, y::BigFloat, z::BigFloat)
    ccall((:mpfr_fms, Base.MPFR.libmpfr), Int32,
          (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat},
           Base.MPFR.MPFRRoundingMode),
          r, x, y, z, _mpfr_rounding())
    return r
end

# mp_copy!(r, x)  →  r = x
@inline mp_copy!(r::BigFloat, x::BigFloat) =
    operate_to!(r, copy, x)

# mp_add_mul!(r, y, z)  →  r += y * z   (single rounding via mpfr_fma)
# More efficient than `r = r + y*z` because it avoids the temporary for y*z.
# Note: this is equivalent to `operate!(add_mul, r, y, z)` but uses the
# short fma path directly.
@inline function mp_add_mul!(r::BigFloat, y::BigFloat, z::BigFloat)
    # r = r + y*z = fma(y, z, r). Aliasing of r is supported.
    return mp_fma!(r, y, z, r)
end

# mp_sub_mul!(r, y, z)  →  r -= y * z
# Implementation: r = -(y*z - r) = -(fms(y, z, r)).
# mpfr_fms computes r = x*y - w, so fms(y, z, r) = y*z - r; negating gives
# r - y*z.  Two ops (fms + neg), still zero-allocation.
@inline function mp_sub_mul!(r::BigFloat, y::BigFloat, z::BigFloat)
    # tmp = y*z - r   (in-place: r is the w operand AND the output)
    mp_fms!(r, y, z, r)
    # Now r = y*z - r_old.  Negate to get r_old - y*z.
    return mp_neg!(r, r)
end

# ---------------------------------------------------------------------------
# Operations MA does not wrap: div, sqrt, set, set_si, neg, abs, pow_si.
# Implemented as direct ccalls, matching MA's ccall signature conventions.
# ---------------------------------------------------------------------------

# mp_div!(r, a, b)  →  r = a / b
@inline function mp_div!(r::BigFloat, a::BigFloat, b::BigFloat)
    ccall((:mpfr_div, Base.MPFR.libmpfr), Int32,
          (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat},
           Base.MPFR.MPFRRoundingMode),
          r, a, b, _mpfr_rounding())
    return r
end

# mp_sqrt!(r, x)  →  r = sqrt(x)
@inline function mp_sqrt!(r::BigFloat, x::BigFloat)
    ccall((:mpfr_sqrt, Base.MPFR.libmpfr), Int32,
          (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
          r, x, _mpfr_rounding())
    return r
end

# mp_rec_sqrt!(r, x)  →  r = 1/sqrt(x)   (single rounding, faster than
# sqrt then div). Used by Cholesky for the inverted diagonal.
@inline function mp_rec_sqrt!(r::BigFloat, x::BigFloat)
    ccall((:mpfr_rec_sqrt, Base.MPFR.libmpfr), Int32,
          (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
          r, x, _mpfr_rounding())
    return r
end

# mp_neg!(r, x)  →  r = -x. (Aliased mp_neg!(x, x) is fine — just flips sign.)
@inline function mp_neg!(r::BigFloat, x::BigFloat)
    ccall((:mpfr_neg, Base.MPFR.libmpfr), Int32,
          (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
          r, x, _mpfr_rounding())
    return r
end

# mp_abs!(r, x)  →  r = |x|
@inline function mp_abs!(r::BigFloat, x::BigFloat)
    ccall((:mpfr_abs, Base.MPFR.libmpfr), Int32,
          (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
          r, x, _mpfr_rounding())
    return r
end

# mp_set!(r, x)  →  r = x (same as mp_copy!, named for symmetry with mpfr_set)
@inline function mp_set!(r::BigFloat, x::BigFloat)
    ccall((:mpfr_set, Base.MPFR.libmpfr), Int32,
          (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
          r, x, _mpfr_rounding())
    return r
end

# mp_set_si!(r, n)  →  r = Int(n)
@inline function mp_set_si!(r::BigFloat, n::Integer)
    ccall((:mpfr_set_si, Base.MPFR.libmpfr), Int32,
          (Ref{BigFloat}, Clong, Base.MPFR.MPFRRoundingMode),
          r, n, _mpfr_rounding())
    return r
end

# mp_set_d!(r, x)  →  r = Float64(x)
@inline function mp_set_d!(r::BigFloat, x::Float64)
    ccall((:mpfr_set_d, Base.MPFR.libmpfr), Int32,
          (Ref{BigFloat}, Cdouble, Base.MPFR.MPFRRoundingMode),
          r, x, _mpfr_rounding())
    return r
end

# mp_pow_si!(r, x, n)  →  r = x^n   (n is a signed integer)
@inline function mp_pow_si!(r::BigFloat, x::BigFloat, n::Integer)
    ccall((:mpfr_pow_si, Base.MPFR.libmpfr), Int32,
          (Ref{BigFloat}, Ref{BigFloat}, Clong, Base.MPFR.MPFRRoundingMode),
          r, x, n, _mpfr_rounding())
    return r
end

# mp_zero!(r)  →  r = 0
@inline mp_zero!(r::BigFloat) = mp_set_si!(r, 0)

# mp_one!(r)  →  r = 1
@inline mp_one!(r::BigFloat) = mp_set_si!(r, 1)

# ---------------------------------------------------------------------------
# Compensated dot product (KBN) is NOT needed — the F1/F2/F3
# inner loops are single-step FMAs (`s += v * G[r,c]` → `mp_fma!(s,v,G,s)`).
# A KBN wrapper would be useful for higher-accuracy column dot products,
# but the tolerance relaxation (`rtol=1e-20`) accepts the slight
# change in summation order from naive FMA.
#
# If needed later, MutableArithmetics' `add_dot` reducer (see
# `MA/src/reduce.jl:23-49`) provides the KBN machinery via
# `buffered_operate_to!(buf, add_dot, ...)`.
# ---------------------------------------------------------------------------
