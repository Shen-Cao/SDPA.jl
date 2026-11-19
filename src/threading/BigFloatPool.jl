# Per-task BigFloat workspace pool.
#
# Avoids the per-operation allocation that plagues naive BigFloat
# multithreading (each +, * allocates a new mpfr_t on the heap).
#
# The pool reduces WORKSPACE MATRIX allocations (the Matrix{BigFloat}
# containers themselves). It does NOT eliminate per-arithmetic-op
# allocations (every BigFloat +/* still allocates a result mpfr_t).
# For full elimination, you'd need MPFR's in-place add!/mul! APIs
# which we use selectively in hot loops.
#
# Usage with OhMyThreads:
#     @tasks for chunk in work
#         @local pool = BigFloatPool(block_size; precision=precision(BigFloat), max_mats=10)
#         compute_block!(pool)
#     end
#
# Resource cap: each pool tracks its total bytes allocated. If max_bytes
# is exceeded, acquire_mat! errors rather than allocating more. The cap
# is applied at CONSTRUCTION time too: pre-allocation alone must
# fit, else the constructor errors descriptively.
#
# Release guard: acquire registers the handed-out object's
# object_id in `checked_out`; release_mat!/release_vec! error on a
# double or foreign release (an object not currently checked out from
# this pool) instead of silently corrupting the free list.

# Floor for the pool byte budget: pools constructed without pre-allocation
# (mat_size=(0,0), e.g. the min_eigen per-task pool) or with size
# mismatches acquire on demand, so the budget must not collapse to zero.
const _DEFAULT_POOL_BUDGET = 64 * 1024^2  # 64 MiB

# Per-element byte estimate used for pool byte accounting (budget math and
# on-demand acquire checks). For isbits T (e.g. Float64), `sizeof(T)` is the
# exact per-element cost. For BigFloat, `sizeof(BigFloat)` is only the size
# of the C struct (32 B on 64-bit): each element ALSO carries a ~16 B Julia
# object header plus an MPFR mantissa allocation of ~8·⌈precision/64⌉ bytes
# at `precision` bits. sizeof(BigFloat) undercounts by ~3× at 256-bit;
# use the closed form below:
#     64 + 8 * cld(precision, 64)   (96 B at precision 256)
_elem_bytes(::Type{T}, precision::Int) where {T<:Real} =
    T <: BigFloat ? 64 + 8 * cld(precision, 64) : sizeof(T)

"""
    BigFloatPool{T<:Real}

A per-task pool of pre-allocated BigFloat matrices and vectors.
Created once per task (via `OhMyThreads.@local`) and reused across
all iterations of the task's work chunk.

Fields
------
- `precision::Int`:        MPFR precision in bits (ACCOUNTING-ONLY, see below)
- `max_bytes::Int`:        cap on total pool size
- `bytes_allocated::Int`:  on-demand bytes allocated (fresh acquisitions)
- `mats::Vector{Matrix{T}}`: free list of available matrices
- `vecs::Vector{Vector{T}}`: free list of available vectors
- `checked_out::Set{UInt}`: object_ids of live (acquired, unreleased)
  checkouts — mats and vecs share one set (object_id is unique among
  live objects); guards double/foreign release

`precision` is ACCOUNTING-ONLY: it is used solely for byte math
(`_elem_bytes` in the `max_bytes` budget) and never sets element
precision. Element precision follows the task-local BigFloat precision
at fill/acquire time by design — pools are created per task with
`precision=precision(BigFloat)` and live entirely within one solve
(one-precision-per-solve discipline), so the accounting value and the
actual element precision coincide.
"""
mutable struct BigFloatPool{T<:Real}
    precision::Int
    max_bytes::Int
    bytes_allocated::Int
    mats::Vector{Matrix{T}}
    vecs::Vector{Vector{T}}
    checked_out::Set{UInt}
end

"""
    BigFloatPool{T}(; precision=256, max_mats=10, max_vecs=10, mat_size=(0,0), vec_size=0, max_bytes=nothing) where T

Create a pool with pre-allocated matrices and vectors of the given sizes.

`max_bytes` (the effective cap) bounds the TOTAL pool budget —
pre-allocated plus on-demand acquisitions. With `max_bytes=nothing`
(default) the cap is computed as
`4 * _elem_bytes(T, precision) * mat_size[1] * mat_size[2] * max_mats`
(at least `_DEFAULT_POOL_BUDGET`, capped at `Sys.free_memory() ÷ 2` to
avoid exhausting RAM). An explicit `max_bytes::Int` PINS the cap exactly
(no auto floor/free-memory clamp) — for tests and tight-memory
deployments. The cap is applied BEFORE pre-allocation: if the
pre-allocation alone would exceed it, the constructor errors
descriptively. `bytes_allocated` tracks on-demand (fresh) acquisitions
and starts at zero.
"""
function BigFloatPool{T}(;
    precision::Int=256,
    max_mats::Int=10,
    max_vecs::Int=10,
    mat_size::Tuple{Int,Int}=(0, 0),
    vec_size::Int=0,
    max_bytes::Union{Nothing,Int}=nothing,
) where {T<:Real}
    mat_bytes = _elem_bytes(T, precision) * mat_size[1] * mat_size[2] * max_mats
    vec_bytes = _elem_bytes(T, precision) * vec_size * max_vecs
    requested = mat_bytes + vec_bytes
    # Resolve the cap BEFORE pre-allocating. `max_bytes=nothing` keeps
    # the auto behavior: 4×requested, floored at _DEFAULT_POOL_BUDGET,
    # clamped at half of free memory to avoid exhausting RAM. An explicit
    # Int pins the cap exactly.
    cap = max_bytes === nothing ?
        min(max(4 * requested, _DEFAULT_POOL_BUDGET), Sys.free_memory() ÷ 2) :
        max_bytes
    # Fail fast and descriptively if pre-allocation alone would exceed
    # the cap.
    if requested > cap
        error("BigFloatPool: pre-allocation of $requested bytes exceeds max_bytes=$cap")
    end

    # Pre-allocated mats/vecs must have INDEPENDENT elements. Plain
    # zeros(BigFloat, ...) fills the container with ONE shared BigFloat
    # object (aliased mpfr data), corrupting in-place MPFR writes.
    mats = [_independent_zeros(T, mat_size[1], mat_size[2]) for _ in 1:max_mats]
    vecs = [_independent_zeros(T, vec_size) for _ in 1:max_vecs]

    return BigFloatPool{T}(precision, cap, 0, mats, vecs, Set{UInt}())
end

# Zero a pooled container IN PLACE on re-acquisition. Elements of
# pooled BigFloat containers are already INDEPENDENT objects — created by
# _independent_zeros at construction/fresh-acquire, and kernels either
# mutate them in place (mp_zero!/mp_fma!/mp_set!, e.g. Rgemm! and
# update_eigen!'s value-copy) or assign fresh objects — so zeroing by
# VALUE (mpfr_set_si 0) preserves independence while recycling the
# element objects themselves. Zero is exact at any MPFR
# precision, so values are bit-identical to the fresh-object path.
@inline function _zero_pooled!(A::Matrix{BigFloat})
    @inbounds for i in eachindex(A)
        mp_zero!(A[i])
    end
    return A
end
@inline _zero_pooled!(A::Matrix{T}) where {T<:Real} = fill!(A, zero(T))

@inline function _zero_pooled!(v::Vector{BigFloat})
    @inbounds for i in eachindex(v)
        mp_zero!(v[i])
    end
    return v
end
@inline _zero_pooled!(v::Vector{T}) where {T<:Real} = fill!(v, zero(T))

"""
    acquire_mat!(pool, m, n)

Acquire a `Matrix{T}` of size `(m, n)` from the pool. If the pool has
a matrix of matching size, reuse it — zeroed IN PLACE (the existing
independent element objects are zeroed by value, not replaced with
fresh `BigFloat(0)`s). Otherwise allocate a new one, subject to the
`max_bytes` cap. The handed-out matrix is registered in `checked_out`;
return it via `release_mat!`.

Errors if the allocation would exceed `pool.max_bytes`.
"""
function acquire_mat!(pool::BigFloatPool{T}, m::Int, n::Int) where {T<:Real}
    # Look for a matching matrix in the free list.
    for i in length(pool.mats):-1:1
        if size(pool.mats[i]) == (m, n)
            mat = popat!(pool.mats, i)
            _zero_pooled!(mat)
            push!(pool.checked_out, objectid(mat))
            return mat
        end
    end
    # Allocate new (independent elements — see the constructor note).
    needed = _elem_bytes(T, pool.precision) * m * n
    if pool.bytes_allocated + needed > pool.max_bytes
        error("BigFloatPool: allocation of $needed bytes would exceed max_bytes=$(pool.max_bytes) (currently $(pool.bytes_allocated))")
    end
    pool.bytes_allocated += needed
    mat = _independent_zeros(T, m, n)
    push!(pool.checked_out, objectid(mat))
    return mat
end

"""
    release_mat!(pool, mat)

Return a matrix to the pool for reuse. Errors if `mat` was not
checked out from this pool (double release or foreign matrix) — the
free list is left untouched in that case.
"""
function release_mat!(pool::BigFloatPool{T}, mat::Matrix{T}) where {T<:Real}
    id = objectid(mat)
    if !(id in pool.checked_out)
        error("BigFloatPool: release of a matrix not checked out from this pool")
    end
    delete!(pool.checked_out, id)
    push!(pool.mats, mat)
    return nothing
end

"""
    acquire_vec!(pool, n)

Acquire a `Vector{T}` of length `n` from the pool (zeroed in place on
the reuse path; registered in `checked_out`).
"""
function acquire_vec!(pool::BigFloatPool{T}, n::Int) where {T<:Real}
    for i in length(pool.vecs):-1:1
        if length(pool.vecs[i]) == n
            vec = popat!(pool.vecs, i)
            _zero_pooled!(vec)
            push!(pool.checked_out, objectid(vec))
            return vec
        end
    end
    # Allocate new (independent elements — see the constructor note).
    needed = _elem_bytes(T, pool.precision) * n
    if pool.bytes_allocated + needed > pool.max_bytes
        error("BigFloatPool: allocation of $needed bytes would exceed max_bytes=$(pool.max_bytes)")
    end
    pool.bytes_allocated += needed
    vec = _independent_zeros(T, n)
    push!(pool.checked_out, objectid(vec))
    return vec
end

"""
    release_vec!(pool, vec)

Return a vector to the pool for reuse. Errors if `vec` was not
checked out from this pool (double release or foreign vector).
"""
function release_vec!(pool::BigFloatPool{T}, vec::Vector{T}) where {T<:Real}
    id = objectid(vec)
    if !(id in pool.checked_out)
        error("BigFloatPool: release of a vector not checked out from this pool")
    end
    delete!(pool.checked_out, id)
    push!(pool.vecs, vec)
    return nothing
end
