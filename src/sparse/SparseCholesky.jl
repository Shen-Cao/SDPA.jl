"""
    SparseCholesky

Left-looking sparse Cholesky factorization and sparse triangular solve
for profile-form symmetric positive-definite matrices.

Mirrors C++ `Lal::getCholesky` at `sdpa_linear.cpp:395-437` and
`Lal::solveSystems` at `:698-854`.

Profile form storage (upper triangle, column by column):
- `diagonalIndex[j]` = start index of column j in `sp_ele` (1-based)
- `diagonalIndex[j+1]` = end index (exclusive)
- Within column j: diagonal entry first, then super-diagonal entries
  sorted by row in increasing order
- `row_index[k]` = column of entry k (the smaller index)
- `column_index[k]` = row of entry k (the larger index)

After factorization:
- Diagonal stores `1/sqrt(d)` (inverted, so the solve is a multiply)
- Sub-diagonal stores `L[r,j] = reduced_A[r,j] / sqrt(reduced_A[j,j])`
"""

"""
    sparse_cholesky!(mat::SDPASparseMatrix{T}, diagonalIndex::Vector{Int}) -> Bool

Blocked (right-looking) sparse Cholesky for profile-form SPD matrices.
Returns `true` on success; `false` if a diagonal entry was non-positive
(`≤ 0`), in which case it is set to zero and the factorization stops early.

Deliberate divergence from C++ `Lal::getCholesky` (which always returns
`true`): the caller can now detect a singular/non-PD B matrix.

Port from C++ `Lal::getCholesky` (`sdpa_linear.cpp:395-437`), whose
`Rpotrf("Lower")` kernel is scalar and single-threaded.  This Julia port
is the only threaded Cholesky in either codebase.

Threading: columns are partitioned into block-columns of width `nb`.
Within a block-column the factorization is a serial left-looking sweep
(a later range in the block needs the contributions of all earlier
ranges); the *updates* a source range emits to its trailing ranges are
independent of each other (sub-diagonal entry `(i, j)` touches only the
disjoint target range `j`), so they are parallelized:

  - Phase 1 (serial over source ranges): factor range `i` (diagonal
    1/sqrt + sub-diagonal scaling), then apply its *within-block*
    updates (`targets ≤ c2`) — the `k1` loop over within-block targets
    is run with `@tasks`.
  - Phase 2 (parallel over target block-columns): apply the *deferred*
    updates (`targets > c2`) from all source ranges of the block.  Each
    `@tasks` chunk owns one target block-column and iterates the source
    ranges in increasing `i` order.

The serial chain is only across block-columns (mirrors the blocked
`Rpotrf!` structure: `Rpotf2`-like diagonal sweep + parallel
`Rgemm`-like trailing updates).

Bit-identity: for a fixed target entry `(r, j)` the scalar algorithm
accumulates contributions from source ranges `i` in increasing `i`
order; the blocked scheme preserves that order exactly (block-columns
are processed in order; within a block-column each target range's
contributions come from sources in increasing `i` order; and each
`(k1, k2)` pair writes a distinct target entry, so `@tasks` never
reorders two updates of the same entry).  The BigFloat update
`sp_ele[k3] -= tmp * tmp2` uses `mp_sub_mul!` directly on the array
element — a fused multiply-subtract (`mpfr_fms` + `mpfr_neg`, two MPFR
ops) — so it is allocation-free, and no per-task scratch BigFloat
is needed.
"""

# Block-column width for the blocked factorization.
const _SPARSE_CHOLESKY_NB = 64
# Minimum number of within-block targets before parallelizing a source
# range's Phase-1 update loop (avoid @tasks overhead for tiny ranges).
const _SPARSE_CHOLESKY_PAR_MIN = 8

"""
    _sparse_profile_update!(mat, diagonalIndex, k1)

Apply the rank-1 update emitted by sub-diagonal entry `k1` of its source
range `i = mat.row_index[k1]`:

    L[:, j] -= L[j, i] * L[:, i],   j = mat.column_index[k1]

The `k2` loop walks the tail of source range `i` and updates the target
range `j`'s entries (linear forward scan, entries sorted by row).  All
`(k1, k2)` pairs across the whole factorization update *distinct* target
entries, so concurrent calls on different `k1` never race.
"""
@inline function _sparse_profile_update!(
    mat::SDPASparseMatrix{T}, diagonalIndex::Vector{Int}, k1::Int,
) where {T<:Real}
    tmp = mat.sp_ele[k1]
    target_col = mat.column_index[k1]
    k3 = diagonalIndex[target_col]
    indexB2 = diagonalIndex[target_col + 1]
    src_end = diagonalIndex[mat.row_index[k1] + 1] - 1
    if T <: BigFloat
        @inbounds for k2 in k1:src_end
            tmp2 = mat.sp_ele[k2]
            target_row = mat.column_index[k2]
            # Find entry (target_row, target_col) in the target range.
            # Entries sorted by row ⇒ linear forward scan from previous k3.
            found = false
            while k3 < indexB2
                if mat.column_index[k3] == target_row
                    # sp_ele[k3] -= tmp * tmp2  (mpfr_fms + mpfr_neg)
                    mp_sub_mul!(mat.sp_ele[k3], tmp, tmp2)
                    k3 += 1
                    found = true
                    break
                end
                k3 += 1
            end
            # The profile is fill-in closed (block-level skyline),
            # so every update target must be present; a silent skip would
            # truncate the factor. error(), not @assert:
            # reachable when a truncated profile is forced via
            # SDPA_FORCE_SPARSE_PROFILE=1.
            found || error("sparse profile not fill-in closed: target ($target_row, $target_col) missing")
        end
    else
        @inbounds for k2 in k1:src_end
            tmp2 = mat.sp_ele[k2]
            target_row = mat.column_index[k2]
            found = false
            while k3 < indexB2
                if mat.column_index[k3] == target_row
                    mat.sp_ele[k3] -= tmp * tmp2
                    k3 += 1
                    found = true
                    break
                end
                k3 += 1
            end
            # Same guard as the BigFloat branch above.
            found || error("sparse profile not fill-in closed: target ($target_row, $target_col) missing")
        end
    end
    return nothing
end

"""
    _sparse_cholesky_factor_range!(mat, diagonalIndex, i) -> Bool

Factor range `i` in place: diagonal → `1/sqrt(d)` (or zeroed if `d ≤ 0`),
sub-diagonal entries scaled by the diagonal.  Serial (called from the
block-column sweep only).

Returns `false` if the diagonal is non-positive (`≤ 0`), in which case it
is zeroed and the sub-diagonal scaling still runs (entries multiplied by
the zeroed diagonal become 0), so the factor state stays C++-parity; the
caller aborts the factorization on `false`.
"""
@inline function _sparse_cholesky_factor_range!(
    mat::SDPASparseMatrix{T}, diagonalIndex::Vector{Int}, i::Int,
) where {T<:Real}
    indexA1 = diagonalIndex[i]
    indexA2 = diagonalIndex[i + 1]
    ok = true
    if T <: BigFloat
        buf = zero(BigFloat)
        if mat.sp_ele[indexA1] <= zero(T)
            ok = false
            mat.sp_ele[indexA1] = zero(T)
        else
            mp_rec_sqrt!(buf, mat.sp_ele[indexA1])  # mpfr_rec_sqrt!
            mp_set!(mat.sp_ele[indexA1], buf)
        end
        for k1 in (indexA1 + 1):(indexA2 - 1)
            mp_mul!(buf, mat.sp_ele[k1], mat.sp_ele[indexA1])  # mpfr_mul!
            mp_set!(mat.sp_ele[k1], buf)
        end
    else
        if mat.sp_ele[indexA1] <= zero(T)
            ok = false
            mat.sp_ele[indexA1] = zero(T)
        else
            mat.sp_ele[indexA1] = 1 / sqrt(mat.sp_ele[indexA1])
        end
        for k1 in (indexA1 + 1):(indexA2 - 1)
            mat.sp_ele[k1] *= mat.sp_ele[indexA1]
        end
    end
    return ok
end

"""
    _sparse_cholesky_deferred_block!(mat, diagonalIndex, c1, c2, t0, nb, nDIM)

Phase-2 unit: apply all deferred updates from source ranges `[c1, c2]`
to the target block-column `[t0, min(t0 + nb - 1, nDIM)]`.  Sources are
visited in increasing `i` order, preserving the scalar per-entry
accumulation order (bit-identity).  Different `t0` touch disjoint target
block-columns, so concurrent calls are race-free.
"""
function _sparse_cholesky_deferred_block!(
    mat::SDPASparseMatrix{T}, diagonalIndex::Vector{Int},
    c1::Int, c2::Int, t0::Int, nb::Int, nDIM::Int,
) where {T<:Real}
    tend = min(t0 + nb - 1, nDIM)
    column_index = mat.column_index
    @inbounds for i in c1:c2
        a1 = diagonalIndex[i] + 1
        a2 = diagonalIndex[i + 1] - 1
        if a1 <= a2
            # Source range i's sub-diagonal entries are sorted by target
            # column, so the entries targeting [t0, tend] form a slice.
            lo = searchsortedfirst(column_index, t0, a1, a2, Base.Order.Forward)
            hi = searchsortedlast(column_index, tend, a1, a2, Base.Order.Forward)
            for k1 in lo:hi
                _sparse_profile_update!(mat, diagonalIndex, k1)
            end
        end
    end
    return nothing
end

function sparse_cholesky!(
    mat::SDPASparseMatrix{T}, diagonalIndex::Vector{Int}
) where {T<:Real}
    nDIM = mat.nRow
    @assert nDIM == mat.nCol "sparse_cholesky! requires square matrix"
    @assert length(diagonalIndex) == nDIM + 1 "diagonalIndex length must be nDIM+1"

    use_tasks = Threads.nthreads() > 1
    nb = _SPARSE_CHOLESKY_NB
    column_index = mat.column_index

    c1 = 1
    while c1 <= nDIM
        c2 = min(c1 + nb - 1, nDIM)

        # Phase 1: factor each range of the block-column, applying its
        # within-block updates (targets j ≤ c2) immediately so a later
        # range of the block is fully reduced before factorization.
        for i in c1:c2
            _sparse_cholesky_factor_range!(mat, diagonalIndex, i) || return false
            a1 = diagonalIndex[i] + 1
            a2 = diagonalIndex[i + 1] - 1
            if a1 <= a2
                # Within-block targets form a prefix (sorted by column_index).
                hi = searchsortedlast(column_index, c2, a1, a2, Base.Order.Forward)
                if use_tasks && hi - a1 + 1 >= _SPARSE_CHOLESKY_PAR_MIN
                    @tasks for k1 in a1:hi
                        @set split=:roundrobin
                        _sparse_profile_update!(mat, diagonalIndex, k1)
                    end
                else
                    for k1 in a1:hi
                        _sparse_profile_update!(mat, diagonalIndex, k1)
                    end
                end
            end
        end

        # Phase 2: deferred updates from this block-column to trailing
        # block-columns (> c2).  All target block-columns are independent.
        if c2 < nDIM
            t0_first = c2 + 1
            t0_last = ((nDIM - 1) ÷ nb) * nb + 1
            if use_tasks
                @tasks for t0 in t0_first:nb:t0_last
                    @set split=:roundrobin
                    _sparse_cholesky_deferred_block!(
                        mat, diagonalIndex, c1, c2, t0, nb, nDIM)
                end
            else
                for t0 in t0_first:nb:t0_last
                    _sparse_cholesky_deferred_block!(
                        mat, diagonalIndex, c1, c2, t0, nb, nDIM)
                end
            end
        end

        c1 = c2 + 1
    end
    return true
end

"""
    solve_systems_sparse!(x, mat, b, diagonalIndex) -> x

Sparse triangular solve: `L·L'·x = b`.

Forward sweep (`L·z = b`), then backward sweep (`L'·x = z`).
The matrix `mat` must already be factorized by `sparse_cholesky!`.

Port from C++ `Lal::solveSystems` (`sdpa_linear.cpp:698-854`).

BigFloat path uses an mp_mul!/mp_sub!/mp_set! pattern with a
pre-allocated local accumulator; array writes are in-place value
copies (mp_set!), avoiding shared-storage corruption.
"""
function solve_systems_sparse!(
    x::AbstractVector{T},
    mat::SDPASparseMatrix{T},
    b::AbstractVector{T},
    diagonalIndex::Vector{Int},
) where {T<:Real}
    # VALUE copy (BigFloat): plain `copyto!` copies REFERENCES, so
    # subsequent in-place mp_set! writes into x would corrupt b through
    # the shared objects (the BigFloat reference-copy aliasing hazard). _copyto_independent!
    # is a per-element value copy for BigFloat, plain copyto! for isbits.
    _copyto_independent!(x, b)

    local buf::T
    has_buf = false
    if T <: BigFloat
        buf = zero(BigFloat)
        has_buf = true
    end

    # Forward sweep: L·z = b
    # Process entries in storage order (column by column).
    #   diagonal: x[col] *= 1/L[col,col]   (scale)
    #   off-diag: x[row] -= L[row,col] * x[col]  (eliminate)
    @inbounds for index in 1:mat.nonZeroEffect
        col = mat.row_index[index]      # column (smaller index)
        row = mat.column_index[index]   # row    (larger  index)
        value = mat.sp_ele[index]
        if col == row
            if has_buf
                mp_mul!(buf, value, x[col])
                mp_set!(x[col], buf)
            else
                x[col] *= value
            end
        else
            if has_buf
                # x[row] -= value * x[col]
                mp_mul!(buf, value, x[col])
                mp_sub!(buf, x[row], buf)
                mp_set!(x[row], buf)
            else
                x[row] -= value * x[col]
            end
        end
    end

    # Backward sweep: L'·x = z
    # Process entries in reverse storage order.
    #   diagonal: x[col] *= 1/L[col,col]   (scale)
    #   off-diag: x[col] -= L[row,col] * x[row]  (back-substitute)
    @inbounds for index in mat.nonZeroEffect:-1:1
        col = mat.row_index[index]      # column
        row = mat.column_index[index]   # row
        value = mat.sp_ele[index]
        if col == row
            if has_buf
                mp_mul!(buf, value, x[col])
                mp_set!(x[col], buf)
            else
                x[col] *= value
            end
        else
            if has_buf
                # x[col] -= value * x[row]
                mp_mul!(buf, value, x[row])
                mp_sub!(buf, x[col], buf)
                mp_set!(x[col], buf)
            else
                x[col] -= value * x[row]
            end
        end
    end

    return x
end

"""
    permute_vec!(out, v, ordering)

Apply AMD ordering to a vector: out[i] = v[ordering[i]].
Mirror C++ permuteVec at sdpa_newton.cpp:1274-1305.
"""
function permute_vec!(out::AbstractVector, v::AbstractVector,
                      ordering::AbstractVector{Int})
    @inbounds for i in eachindex(v)
        out[i] = v[ordering[i]]
    end
    return out
end
