"""
Aggregate sparsity detection and fill-reducing ordering for the
sparse Schur complement.

The aggregate sparsity pattern of the Schur complement matrix B is
determined by which constraint pairs (i, j) share a nonzero entry
in at least one block of the problem. If A_i and A_j both have a
nonzero in block k at position (r, c), then B[i, j] will be nonzero
(assuming X and Z are generic).

This module:
1. Builds the adjacency graph from the constraint matrices {A_i}
2. Decides whether to use sparse or dense Schur complement
3. Computes a fill-reducing ordering via AMD

Decision rule:
  - if m < 100: dense
  - if nBlock < 5: dense
  - if aggregate_density > 0.25: dense
  - else: sparse with AMD ordering
"""

using AMD
using Metis
using SparseArrays

"""
    build_aggregate_graph(problem::SDPAProblem{T}) -> SparseMatrixCSC{Bool,Int}

Build the aggregate sparsity adjacency matrix of the Schur complement.
Entry (i, j) is true if constraints i and j share a nonzero in any block.
"""
function build_aggregate_graph(problem::SDPAProblem{T}) where {T<:Real}
    m = problem.m
    # Build a set of (i, block, row, col) tuples for each constraint.
    # Two constraints that share (block, row, col) are connected.
    # We use a Dict mapping (block, row, col) → Set{Int} of constraint indices.
    position_map = Dict{Tuple{Int,Int,Int}, Set{Int}}()

    for i in 1:m
        Ai = problem.A[i]
        # SDP blocks
        for sdp_k in 1:Ai.SDP_sp_nBlock
            blk = Ai.SDP_sp_index[sdp_k]
            Ai_block = Ai.SDP_sp_block[sdp_k]
            if Ai_block.type === :sparse
                for nz in 1:Ai_block.nonZeroEffect
                    r = Ai_block.row_index[nz]
                    c = Ai_block.column_index[nz]
                    # get!: single lookup per entry (the m·n² dense-block
                    # scan below is the hot case).
                    push!(get!(position_map, (blk, r, c), Set{Int}()), i)
                end
            else
                # Dense block: all entries are potentially nonzero
                n = Ai_block.nRow
                for r in 1:n, c in 1:n
                    if !iszero(Ai_block.de_ele[r, c])
                        push!(get!(position_map, (blk, r, c), Set{Int}()), i)
                    end
                end
            end
        end
        # LP blocks: iterate the compact nonzero list only (every stored
        # entry is nonzero by construction). LP_sp_index[lp_pos] is the
        # problem-level LP scalar position (identity values).
        for lp_pos in 1:Ai.LP_nnz
            lp_k = Ai.LP_sp_index[lp_pos]
            push!(get!(position_map, (-1, lp_k, 0), Set{Int}()), i)
        end
    end

    # Build adjacency from shared positions, deduping to the FIRST
    # occurrence of each unordered pair. A pair sharing k
    # positions would otherwise be pushed k times: on a dense-support
    # problem (one n=50 block, m=1000, every constraint dense) every
    # position carries the full 1000-constraint clique, so the undeduped
    # volume is 2500·C(1000,2) ≈ 1.25×10⁹ pairs (~20 GB of Ints) before
    # the density gate can route dense. Identical constraint sets (the
    # dense-block regime) are skipped wholesale after the first.
    unique_pairs = Tuple{Int,Int}[]
    seen_pairs = Set{Tuple{Int,Int}}()
    processed_sets = Set{Set{Int}}()
    for (_, constraint_set) in position_map
        n_c = length(constraint_set)
        n_c <= 1 && continue
        constraint_set in processed_sets && continue
        push!(processed_sets, constraint_set)
        cl = collect(constraint_set)
        for a in 1:n_c, b in (a+1):n_c
            i1 = cl[a]
            i2 = cl[b]
            key = i1 < i2 ? (i1, i2) : (i2, i1)
            if !(key in seen_pairs)
                push!(seen_pairs, key)
                push!(unique_pairs, (i1, i2))
            end
        end
    end
    # Pre-size rows/cols to the deduped edge count (both orientations +
    # the diagonal) so push! never reallocates (mirrors
    # _block_level_ordering's pre-sizing).
    rows = Int[]
    cols = Int[]
    sizehint!(rows, 2 * length(unique_pairs) + m)
    sizehint!(cols, 2 * length(unique_pairs) + m)
    for (i1, i2) in unique_pairs
        push!(rows, i1, i2)
        push!(cols, i2, i1)
    end
    # Diagonal is always present
    for i in 1:m
        push!(rows, i)
        push!(cols, i)
    end

    # combine=| : duplicate edges combine with OR (union), not the generic
    # `+` default. The dedup above makes duplicates impossible, but
    # the explicit combine documents the intended semantics. (SparseArrays
    # 1.12 takes `combine` positionally; the Bool-vector default already
    # routes to `|` — passing it is the explicit statement.)
    return sparse(rows, cols, trues(length(rows)), m, m, |)
end

"""
    compute_aggregate_density(adj::SparseMatrixCSC{Bool,Int}) -> Float64

Compute the density (fraction of nonzeros) of the adjacency matrix.
"""
compute_aggregate_density(adj::SparseMatrixCSC{Bool,Int}) = nnz(adj) / (size(adj, 1) * size(adj, 2))

"""
    decide_dense_or_sparse(m::Int, nBlock::Int, density::Float64) -> Bool

Return true if dense Schur complement should be used.

Thresholds:
  - m < 100: always dense
  - nBlock < 5: always dense
  - density > 0.25: dense
"""
function decide_dense_or_sparse(m::Int, nBlock::Int, density::Float64)
    m < 100 && return true
    nBlock < 5 && return true
    density > 0.25 && return true
    return false
end

"""
    compute_ordering(adj; method=:amd) -> Vector{Int}

Compute a fill-reducing ordering for the adjacency graph.

- `method=:amd` (default): Approximate Minimum Degree (AMD.jl).
- `method=:metis`: METIS nested dissection (Metis.jl).

Returns a permutation vector `p` such that `adj[p, p]` has reduced
fill-in.
"""
function compute_ordering(
    adj::SparseMatrixCSC{Bool,Int}; method::Symbol=:amd,
)
    if method === :amd
        return compute_amd_ordering(adj)
    elseif method === :metis
        # Metis.permutation returns (perm, invperm). Both are Vector{Int}.
        # The perm gives new[i] = old[perm[i]] (matching our ordering convention).
        perm, _ = Metis.permutation(adj)
        return perm
    else
        error("Unknown ordering method: $method (expected :amd or :metis)")
    end
end

"""
    compute_amd_ordering(adj::SparseMatrixCSC{Bool,Int}) -> Vector{Int}

Compute an Approximate Minimum Degree ordering for the adjacency graph.
Returns a permutation vector `p` such that `adj[p, p]` has reduced fill-in.
"""
function compute_amd_ordering(adj::SparseMatrixCSC{Bool,Int})
    # AMD expects a symmetric sparse matrix (Float64 or Int pattern).
    # Convert Bool pattern to Float64 for AMD.jl compatibility.
    pattern = SparseMatrixCSC{Float64,Int}(
        adj.m, adj.n, adj.colptr, adj.rowval, ones(Float64, nnz(adj)))
    return AMD.amd(pattern)
end

"""
    analyze_sparsity(problem; method=:metis) -> (use_dense::Bool, ordering::Vector{Int}, adj)

Analyze the problem's sparsity structure and decide:
1. Whether to use dense or sparse Schur complement
2. If sparse, the fill-reducing ordering to use (AMD or METIS)

Returns (true, Int[], adj) for dense path, (false, ordering, adj) for sparse path.
The adjacency graph is returned so the caller can build the skyline profile
without recomputation.
"""
function analyze_sparsity(
    problem::SDPAProblem{T}; method::Symbol=:metis, force_sparse::Bool=false,
) where {T<:Real}
    m = problem.m
    nBlock = problem.nBlock
    # Early gate: when the m/nBlock pre-rules already route dense, the
    # position-level graph is dead work — its density can never change the
    # decision (mirror C++ sdpa_chordal.cpp:349 `m <= 100 || nBlock <= 5`).
    # Skipping it avoids the m·n² Dict-key scan and the Σ c_p² clique-pair
    # blow-up on dense-support problems. force_sparse bypasses the gate:
    # solve.jl's force_sparse override needs the graph for its METIS
    # ordering (the empty adj returned here is an opaque dense-routing
    # signal, never consumed).
    if !force_sparse && decide_dense_or_sparse(m, nBlock, 0.0)
        return true, Int[], spzeros(Bool, m, m)
    end
    adj = build_aggregate_graph(problem)
    density = compute_aggregate_density(adj)

    if decide_dense_or_sparse(m, nBlock, density)
        return true, Int[], adj
    else
        ordering = compute_ordering(adj; method=method)
        return false, ordering, adj
    end
end

"""
    build_skyline_from_graph!(adj, ordering, reverse_ordering) -> Vector{Int}

Compute the skyline from the **position-level** aggregate adjacency graph.
Two constraints are connected iff they share the same (block, row, col)
position in some constraint matrix.

This is an APPROXIMATION of the true Schur complement structure: B[i,j] is
structurally nonzero whenever A_i and A_j share a block at ANY positions
(X·A_i·Z⁻¹ is dense within the block; X, Z⁻¹ are dense PD throughout the
IPM). Block-level connectivity — which C++ SDPA-GMP uses via
`Chordal::makeGraph` (sdpa_chordal.cpp:141-211, merging the ENTIRE block
constraint list per constraint) — is the exact structure, not this
position-level graph. The position-level skyline is therefore NOT the
profile used for the sparse path: solve.jl builds the profile from the
block-level skyline (`build_skyline_from_problem!`), and
`make_aggrigate_index_sdp!` errors on any block-level pair outside it
(mirroring C++ "program bug" at sdpa_newton.cpp:481). This function
remains for ordering (AMD/METIS) and analysis purposes only.
"""
function build_skyline_from_graph!(
    adj::SparseMatrixCSC{Bool,Int},
    ordering::AbstractVector{<:Integer},
    reverse_ordering::AbstractVector{<:Integer},
)
    m = length(ordering)
    skyline = collect(1:m)  # diagonal at minimum
    rows = rowvals(adj)

    for j in 1:m
        orig_j = ordering[j]
        max_row = j  # diagonal
        for ptr in nzrange(adj, orig_j)
            orig_i = rows[ptr]
            p_i = reverse_ordering[orig_i]
            if p_i > max_row
                max_row = p_i
            end
        end
        skyline[j] = max_row
    end

    return skyline
end

"""
    build_skyline_from_problem!(problem, ordering, reverse_ordering)
        -> Vector{Int}

Compute the skyline from **block-level** connectivity. Two constraints
are connected iff they both touch the same SDP block (regardless of
position within the block) or the same LP scalar.

This is the CORRECT connectivity for the sparse profile, matching C++
SDPA-GMP's `Chordal::makeGraph` (sdpa_chordal.cpp:141-211): B[i,j] is
structurally nonzero whenever A_i and A_j share a block, because
X·A_i·Z⁻¹ is dense within the block. `makeGraph` merges the SDP AND the
LP constraint lists (sdpa_chordal.cpp:189-195, 204-211); the LP merge
keeps the LP Schur contributions (`_compute_bmat_sparse_lp!`) inside the
profile — a missing LP pair would silently truncate the Schur matrix
(C++ rErrors "program bug", sdpa_newton.cpp:677). The block-level
skyline covers every (i,j) pair that `make_aggrigate_index_sdp!` emits,
and solve.jl uses it BOTH for the sparse/dense density decision and for
the actual sparse profile. The position-level
`build_skyline_from_graph!` under-approximates this structure and is
NOT a C++-matching alternative.

The returned skyline is **fill-in closed**: after recording the original
connectivity it is extended to the symbolic-factorization envelope of
the ordered graph (every position the profile-form Cholesky factor can
touch), mirroring C++ SDPA-GMP's `Chordal::symbfac` profile
(sdpa_newton.cpp:225-340). Without this closure the profile is
under-closed and the factorization produces an inexact factor.
"""
function build_skyline_from_problem!(
    problem::SDPAProblem{T},
    ordering::AbstractVector{<:Integer},
    reverse_ordering::Vector{Int},
) where {T<:Real}
    m = problem.m
    skyline = collect(1:m)  # diagonal at minimum

    # Iterate each SDP block; for each pair (k1, k2) of constraints
    # touching that block, extend the skyline.
    for l in 1:length(problem.SDP_nConstraint)
        nC_l = problem.SDP_nConstraint[l]
        nC_l <= 1 && continue
        constrs = problem.SDP_constraint[l]
        for k1 in 1:nC_l
            i = constrs[k1]
            p_i = reverse_ordering[i]
            for k2 in (k1+1):nC_l
                j = constrs[k2]
                p_j = reverse_ordering[j]
                ii = min(p_i, p_j)
                jj = max(p_i, p_j)
                if jj > skyline[ii]
                    skyline[ii] = jj
                end
            end
        end
    end

    # Iterate each LP scalar position; same connectivity (C++
    # makeGraph's LP merge, sdpa_chordal.cpp:204-211). The LP tables are
    # sized per LP scalar (SDPAProblem.jl) and contain only constraints
    # with a nonzero LP value in `LP_constraint[l]` — exactly
    # the pairs `_compute_bmat_sparse_lp!` emits.
    for l in 1:length(problem.LP_nConstraint)
        nC_l = problem.LP_nConstraint[l]
        nC_l <= 1 && continue
        constrs = problem.LP_constraint[l]
        for k1 in 1:nC_l
            i = constrs[k1]
            p_i = reverse_ordering[i]
            for k2 in (k1+1):nC_l
                j = constrs[k2]
                p_j = reverse_ordering[j]
                ii = min(p_i, p_j)
                jj = max(p_i, p_j)
                if jj > skyline[ii]
                    skyline[ii] = jj
                end
            end
        end
    end

    # Fill-in closure (C++ `Chordal`/symbfac semantics, sdpa_newton.cpp:
    # 225-340): the skyline must contain every position the Cholesky
    # factor needs, not just the original B connectivity.  If column i's
    # envelope reaches row j (i < j ≤ skyline[i]), the elimination of i
    # emits updates to column j that touch every row ≤ skyline[i], so
    # column j's envelope must reach at least skyline[i] — otherwise the
    # profile-form Cholesky silently drops genuine fill-in and produces
    # an inexact factor.
    #
    # This is the symbolic-factorization envelope of the ordered graph,
    # computed to a fixed point (processing columns in increasing order
    # makes each `skyline[i]` final before it is used as a source).
    @inbounds for i in 1:m
        hi = skyline[i]
        for j in (i + 1):hi
            if skyline[j] < hi
                skyline[j] = hi
            end
        end
    end

    return skyline
end

"""
    profile_from_skyline!(skyline, ::Type{T})
        -> (sparse_mat::SDPASparseMatrix{T}, diagonalIndex::Vector{Int})

Build a profile-form sparse matrix from a precomputed skyline.
"""
function profile_from_skyline!(
    skyline::Vector{Int},
    ::Type{T},
) where {T<:Real}
    m = length(skyline)

    # Build diagonalIndex from skyline.
    diagonalIndex = Vector{Int}(undef, m + 1)
    diagonalIndex[1] = 1
    for j in 1:m
        nz_in_col = skyline[j] - j + 1
        diagonalIndex[j + 1] = diagonalIndex[j] + nz_in_col
    end
    nnz = diagonalIndex[m + 1] - 1

    # Allocate sparse matrix.
    sparse_mat = SDPASparseMatrix{T}(m, m, :sparse, nnz)
    sparse_mat.nonZeroCount = nnz
    sparse_mat.nonZeroEffect = nnz

    # Fill row_index and column_index within skyline.
    idx = 1
    for j in 1:m
        sparse_mat.row_index[idx] = j
        sparse_mat.column_index[idx] = j
        idx += 1
        for i in (j + 1):skyline[j]
            sparse_mat.row_index[idx] = j
            sparse_mat.column_index[idx] = i
            idx += 1
        end
    end

    return sparse_mat, diagonalIndex
end
