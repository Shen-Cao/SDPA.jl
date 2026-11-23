@enum _BorderedState begin
    _BORDERED_EMPTY
    _BORDERED_ASSEMBLED
    _BORDERED_FAILED
    _BORDERED_FACTORED
end

# In-place storage: interior blocks become L_i, coupling panels become
# W_i = L_i\E_i, and the border becomes the factor of C - sum(W_i'W_i).
# All triangular matrices are packed; nothing stores a full m×m matrix.
mutable struct _BorderedFactors{T<:Real}
    layout::_BorderedLayout
    interiors::Vector{_SchurSkyline{T}}
    panels::Vector{Matrix{T}}
    separator::_SchurSkyline{T}
    starts::Vector{Int}
    diagonal_scale::Vector{T}
    rhs::Vector{T}
    pivot_guard::T
    state::_BorderedState
end

function _BorderedFactors{T}(layout::_BorderedLayout) where {T<:Union{Float64,BigFloat}}
    m = length(layout.order)
    k = layout.border
    starts = Vector{Int}(undef, length(layout.sizes) + 1)
    starts[1] = 1
    for (i, n) in enumerate(layout.sizes)
        starts[i + 1] = Base.checked_add(starts[i], n)
    end
    # The sentinel is also the first permuted position of the border.
    return _BorderedFactors{T}(
        layout, [_SchurSkyline{T}(fill(n, n)) for n in layout.sizes],
        [_independent_zeros(T, n, k) for n in layout.sizes],
        _SchurSkyline{T}(fill(k, k)), starts,
        _independent_zeros(T, m), _independent_zeros(T, m),
        T(16) * T(max(m, 1)) * eps(one(T)), _BORDERED_EMPTY,
    )
end

function _zero_bordered_values!(values::AbstractArray{BigFloat})
    # The constructor gave each slot independent storage. Reuse it rather
    # than allocate an entire new numerical factor on every assembly.
    for x in values
        mp_zero!(x)
    end
    return nothing
end

function _zero_bordered_values!(values::AbstractArray{T}) where {T}
    fill!(values, zero(T))
    return nothing
end

function _reset_bordered!(F::_BorderedFactors)
    F.state = _BORDERED_EMPTY
    for D in F.interiors
        _zero_bordered_values!(D.sp_ele)
    end
    for E in F.panels
        _zero_bordered_values!(E)
    end
    _zero_bordered_values!(F.separator.sp_ele)
    return nothing
end

@inline function _bordered_put!(a::AbstractArray{T}, x::T, i::Int) where {T}
    if T === BigFloat
        mp_set!(a[i], x)
    else
        a[i] = x
    end
    return nothing
end

@inline function _bordered_add_value!(a::AbstractArray{T}, x::T, i::Int) where {T}
    if T === BigFloat
        mp_add!(a[i], a[i], x)
    else
        a[i] += x
    end
    return nothing
end

@inline function _bordered_panel_add!(A::Matrix{T}, i::Int, j::Int, v::T) where {T}
    if T === BigFloat
        mp_add!(A[i, j], A[i, j], v)
    else
        A[i, j] += v
    end
    return nothing
end

# Check the destination even though a valid certificate makes crossing
# an invariant. Never silently discard a contribution if it is violated.
@inline function _bordered_add!(F::_BorderedFactors{T}, i::Int, j::Int, v::T) where {T}
    L = F.layout
    a, b = L.island_of[i], L.island_of[j]
    u, w = L.local_index[i], L.local_index[j]
    if iszero(a) && !iszero(b)
        _bordered_panel_add!(F.panels[b], w, u, v)
    elseif !iszero(a) && iszero(b)
        _bordered_panel_add!(F.panels[a], u, w, v)
    else
        a == b || error("bordered Schur contribution crosses certified islands at rows ($i, $j)")
        D = iszero(a) ? F.separator : F.interiors[a]
        l, r = minmax(u, w)
        _bordered_add_value!(D.sp_ele, v, D.offsets[l] + (r - l))
    end
    return nothing
end

@inline function _bordered_scale!(B::AbstractArray{T}, p::Int, scale::T) where {T}
    if T === BigFloat
        mp_mul!(B[p], B[p], scale)
    else
        B[p] *= scale
    end
    return nothing
end

@inline function _bordered_subtract!(B::AbstractArray{T}, p::Int, a::T, b::T) where {T}
    if T === BigFloat
        mp_sub_mul!(B[p], a, b)
    else
        B[p] -= a * b
    end
    return nothing
end

function _copy_bordered_diagonal!(scales, start, mat::_SchurSkyline)
    for i in 1:(length(mat.offsets) - 1)
        _bordered_put!(scales, mat.sp_ele[mat.offsets[i]], start + (i - 1))
    end
    return nothing
end

# Right-looking packed Cholesky, serial within a single interior.
# The diagonal is an inverse square root, not an explicit matrix inverse.
function _bordered_packed_cholesky!(
    mat::_SchurSkyline{T}, scales::Vector{T}, start::Int, guard::T,
) where {T}
    values, off = mat.sp_ele, mat.offsets
    n = length(off) - 1
    tmp = zero(T)
    @inbounds for i in 1:n
        d = off[i]
        pivot, scale = values[d], scales[start + (i - 1)]
        isfinite(pivot) && pivot > 0 && isfinite(scale) && scale > 0 || return false
        # Conservative cancellation check. A positive rounded pivot is not
        # proof of SPD; compare to the ORIGINAL diagonal, including the
        # original border diagonal before subtracting any W_i'W_i.
        if T === BigFloat
            mp_div!(tmp, pivot, scale)
            tmp > guard || return false
            mp_rec_sqrt!(tmp, pivot)
            isfinite(tmp) && tmp > 0 || return false
            mp_set!(values[d], tmp)
        else
            pivot >= floatmin(Float64) && scale >= floatmin(Float64) || return false
            pivot / scale > guard || return false
            tmp = inv(sqrt(pivot))
            isfinite(tmp) && tmp > 0 || return false
            values[d] = tmp
        end
        for a in (d + 1):(off[i + 1] - 1)
            _bordered_scale!(values, a, values[d])
        end
        for j in (i + 1):n
            a = d + (j - i)
            k = off[j]
            for b in a:(off[i + 1] - 1)
                _bordered_subtract!(values, k, values[a], values[b])
                k += 1
            end
        end
    end
    return all(isfinite, values)
end

# Unblocked forward/backward solves walk one packed factor per vector, or
# a contiguous column of a panel. start is the first slot in that vector.
function _bordered_forward!(x, start::Int, L::_SchurSkyline)
    vals, off = L.sp_ele, L.offsets
    n = length(off) - 1
    @inbounds for i in 1:n
        p = start + (i - 1)
        d = off[i]
        _bordered_scale!(x, p, vals[d])
        for j in (i + 1):n
            _bordered_subtract!(x, start + (j - 1), vals[d + (j - i)], x[p])
        end
    end
    return nothing
end

function _bordered_backward!(x, start::Int, L::_SchurSkyline)
    vals, off = L.sp_ele, L.offsets
    n = length(off) - 1
    @inbounds for i in n:-1:1
        p = start + (i - 1)
        d = off[i]
        for j in n:-1:(i + 1)
            _bordered_subtract!(x, p, vals[d + (j - i)], x[start + (j - 1)])
        end
        _bordered_scale!(x, p, vals[d])
    end
    return nothing
end

function _factor_bordered_interior!(F::_BorderedFactors, i::Int)
    D, E = F.interiors[i], F.panels[i]
    all(isfinite, D.sp_ele) && all(isfinite, E) || return false
    start = F.starts[i]
    _copy_bordered_diagonal!(F.diagonal_scale, start, D)
    _bordered_packed_cholesky!(D, F.diagonal_scale, start, F.pivot_guard) || return false
    n, k = size(E)
    for j in 1:k
        _bordered_forward!(E, 1 + (j - 1) * n, D)
    end
    return all(isfinite, E)
end

function _bordered_accumulate_border!(sep::_SchurSkyline{T}, W::Matrix{T}) where {T}
    vals, off = sep.sp_ele, sep.offsets
    k = size(W, 2)
    @inbounds for i in 1:k, j in i:k, r in axes(W, 1)
        a, b = W[r, i], W[r, j]
        p = off[i] + (j - i)
        if T === Float64
            product = a * b
            # A normal separator does not make a lost Gram contribution
            # meaningful. Decline nonzero products outside the normal range.
            if !iszero(a) && !iszero(b) && abs(product) < floatmin(Float64)
                return false
            end
            vals[p] -= product
        else
            _bordered_subtract!(vals, p, a, b)
        end
    end
    return true
end

function _factor_bordered!(F::_BorderedFactors{T}) where {T}
    F.state === _BORDERED_ASSEMBLED || error("bordered factor requires fresh assembly")
    # Do not allow a failed or interrupted factor to service any RHS.
    F.state = _BORDERED_FAILED
    sep = F.separator
    all(isfinite, sep.sp_ele) || return false
    _copy_bordered_diagonal!(F.diagonal_scale, F.starts[end], sep)
    count = length(F.interiors)
    if Threads.nthreads() > 1 && maximum(F.layout.sizes) >= 64
        # Independent packed interiors/panels; no inner threaded kernels.
        # Vector{Bool}, not packed BitVector (concurrent bit writes race).
        ok = Vector{Bool}(undef, count)
        @tasks for i in 1:count
            ok[i] = _factor_bordered_interior!(F, i)
        end
        all(ok) || return false
    else
        for i in 1:count
            _factor_bordered_interior!(F, i) || return false
        end
    end
    # Fixed island/row reduction order, even when interiors ran in parallel.
    for W in F.panels
        _bordered_accumulate_border!(sep, W) || return false
    end
    _bordered_packed_cholesky!(sep, F.diagonal_scale, F.starts[end], F.pivot_guard) || return false
    F.state = _BORDERED_FACTORED
    return true
end

function _solve_bordered!(
    dst::AbstractVector{T}, F::_BorderedFactors{T}, g::AbstractVector{T},
) where {T}
    F.state === _BORDERED_FACTORED || error("bordered solve requires valid factors")
    Base.require_one_based_indexing(dst, g)
    perm, x = F.layout.order, F.rhs
    length(perm) == length(dst) == length(g) || throw(DimensionMismatch("bordered RHS size"))
    all(isfinite, g) || return false
    @inbounds for i in eachindex(perm)
        _bordered_put!(x, g[perm[i]], i)
    end
    border_start = F.starts[end]
    for (i, W) in enumerate(F.panels)
        start = F.starts[i]
        _bordered_forward!(x, start, F.interiors[i])
        @inbounds for j in axes(W, 2), r in axes(W, 1)
            _bordered_subtract!(x, border_start + (j - 1), W[r, j], x[start + (r - 1)])
        end
    end
    _bordered_forward!(x, border_start, F.separator)
    _bordered_backward!(x, border_start, F.separator)
    for (i, W) in enumerate(F.panels)
        start = F.starts[i]
        @inbounds for j in axes(W, 2), r in axes(W, 1)
            _bordered_subtract!(x, start + (r - 1), W[r, j], x[border_start + (j - 1)])
        end
        _bordered_backward!(x, start, F.interiors[i])
    end
    all(isfinite, x) || return false
    # Only publish the direction after every piece succeeded.
    @inbounds for i in eachindex(perm)
        _bordered_put!(dst, x[i], perm[i])
    end
    return true
end
