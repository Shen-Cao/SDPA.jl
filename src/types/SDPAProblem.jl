"""
    SDPAProblem{T<:Real}

A complete SDP problem instance. Mirrors C++ `sdpa::InputData` at
`sdpa_dataset.h:79-121`.

The problem is

    minimize    ⟨C, X⟩
    subject to  ⟨A_i, X⟩ = b_i  for i = 1..m
                X ⪰ 0 (block-positive-semidefinite)

where `X` lives in the product cone of `blockStruct`. Each block is
either SDP (`blockStruct[k] >= 2`, a symmetric matrix of that size)
or LP (`blockStruct[k] < 0` or `== 1`, a length-`|bs|` nonnegative
vector).

Fields
------
- `m::Int`:                                  constraint count (mDIM)
- `nBlock::Int`:                             total block count
- `blockStruct::Vector{Int}`:                signed block sizes
- `b::SDPAVector{T}`:                        RHS vector (length m)
- `C::SDPASparseLinearSpace{T}`:             objective matrix
- `A::Vector{SDPASparseLinearSpace{T}}`:     constraint matrices (length m)
- `SDP_nConstraint::Vector{Int}`:            per-SDP-block list of how many
  constraints touch block k
- `SDP_constraint::Vector{Vector{Int}}`:     `SDP_constraint[k]` is the list
  of constraint indices i with a nonzero in block k
- `SDP_blockIndex::Vector{Vector{Int}}`:     parallel to `SDP_constraint`;
  the per-A_i sub-index where block k's contribution lives
- `LP_nConstraint`, `LP_constraint`:          same, for LP scalars
- `LP_blockIndex::Vector{Vector{Int}}`:       parallel to `LP_constraint`;
  stores the problem-level LP SCALAR index (= the table's own outer
  index l; C++ parity — sdpa_dataset.cpp `initialize_index_LP` pushes
  `LPBlock.lookup(l)`). NOT a compact storage position: do NOT index
  `A[i].LP_sp_val` with it; use `get_lp_value(A[i], idx)` instead.

The index tables are populated by `initialize_index!` and consumed by
the IPM to skip zero block-pairs in the Schur complement assembly.
"""
struct SDPAProblem{T<:Real}
    m::Int
    nBlock::Int
    blockStruct::Vector{Int}
    b::SDPAVector{T}
    C::SDPASparseLinearSpace{T}
    A::Vector{SDPASparseLinearSpace{T}}

    SDP_nConstraint::Vector{Int}
    SDP_constraint::Vector{Vector{Int}}
    SDP_blockIndex::Vector{Vector{Int}}

    LP_nConstraint::Vector{Int}
    LP_constraint::Vector{Vector{Int}}
    LP_blockIndex::Vector{Vector{Int}}

    function SDPAProblem{T}(
        m::Int,
        nBlock::Int,
        blockStruct::AbstractVector{Int},
        b::SDPAVector{T},
        C::SDPASparseLinearSpace{T},
        A::Vector{SDPASparseLinearSpace{T}},
    ) where {T<:Real}
        m >= 0 || error("m must be non-negative, got $m")
        nBlock >= 0 || error("nBlock must be non-negative, got $nBlock")
        length(blockStruct) == nBlock ||
            throw(DimensionMismatch("blockStruct length != nBlock"))
        length(b) == m ||
            throw(DimensionMismatch("b length $(length(b)) != m $m"))
        length(A) == m ||
            throw(DimensionMismatch("A length $(length(A)) != m $m"))
        # Total LP scalar count: each LP block of size |bs| contributes
        # |bs| separate LP entries. Sized this way (not nBlock) because
        # LP_sp_index stores LP scalar positions, not problem-block indices.
        nLP_total = lp_scalar_count(blockStruct)
        # Index tables are populated lazily by initialize_index!.
        return new{T}(
            m, nBlock, collect(Int, blockStruct), b, C, A,
            # SDP tables sized nBlock (only SDP blocks get populated)
            zeros(Int, nBlock),
            [Int[] for _ in 1:nBlock],
            [Int[] for _ in 1:nBlock],
            # LP tables sized nLP_total (one entry per LP scalar)
            zeros(Int, max(nLP_total, 1)),
            [Int[] for _ in 1:max(nLP_total, 1)],
            [Int[] for _ in 1:max(nLP_total, 1)],
        )
    end
end

"""
    initialize_index!(problem)

Populate the per-block constraint-sparsity tables `SDP_constraint`,
`SDP_blockIndex`, `LP_constraint`, `LP_blockIndex`. For each SDP block k,
record which constraint indices i have a nonzero in A_i's block k.
Mirrors C++ `InputData::initialize_index_SDP` at
`sdpa_dataset.cpp:430-460` plus `initialize_index_LP` at `:490-510`.

Zero-nonzero (empty) blocks are skipped. The C++ `SparseLinearSpace`
keeps only blocks with `SDP_NonZeroNumber[l] > 0` / `LP_NonZeroNumber[l]
== true` at construction (sdpa_struct.cpp:885-912, 948-972), and its
reader records a block only when it carries a value != 0
(sdpa_io.cpp:539, 550); the Julia readers allocate every SDP block as
an empty shell, so the SDP filter must happen here:

- For SDP blocks the per-A_i sub-block is a sparse matrix; "nonzero"
  means `nonZeroEffect > 0` (at least one stored entry). Blocks the
  reader never allocated (`isassigned` false) are empty by definition.
- For LP blocks the per-A_i sub-block is a compact nonzero list
  (`LP_nnz` / `LP_sp_index` / `LP_sp_val`); every stored entry is nonzero
  by construction, so iterating the compact list is the filter.
"""
function initialize_index!(problem::SDPAProblem{T}) where {T<:Real}
    fill!(problem.SDP_nConstraint, 0)
    fill!(problem.LP_nConstraint, 0)
    for k in 1:problem.nBlock
        empty!(problem.SDP_constraint[k])
        empty!(problem.SDP_blockIndex[k])
    end
    for k in 1:length(problem.LP_constraint)
        empty!(problem.LP_constraint[k])
        empty!(problem.LP_blockIndex[k])
    end

    bs = problem.blockStruct
    for i in 1:problem.m
        Ai = problem.A[i]
        # Map between A_i's internal block list (1..SDP_sp_nBlock) and the
        # problem-level block indices (Ai.SDP_sp_index[k] is the original
        # blockStruct index for the k-th SDP entry of A_i).
        for sdp_k in 1:Ai.SDP_sp_nBlock
            # Skip unallocated or zero-nonzero (empty) blocks — C++
            # SparseLinearSpace keeps only blocks with NonZeroNumber > 0
            # (sdpa_struct.cpp:885-912), so C++ initialize_index never
            # sees them. The isassigned guard MUST come first —
            # dereferencing SDP_sp_block[sdp_k] before it throws
            # UndefRefError on an unallocated block.
            if !isassigned(Ai.SDP_sp_block, sdp_k)
                continue
            end
            block = Ai.SDP_sp_block[sdp_k]
            block.nonZeroEffect == 0 && continue
            global_k = Ai.SDP_sp_index[sdp_k]
            push!(problem.SDP_constraint[global_k], i)
            push!(problem.SDP_blockIndex[global_k], sdp_k)
            problem.SDP_nConstraint[global_k] += 1
        end
        # Compact LP storage keeps ONLY nonzero scalars (the C++
        # reader keeps a scalar only when its value != 0,
        # sdpa_io.cpp:550), so iterating the compact list IS the
        # nonzero filter — no value-based skip needed.
        for lp_pos in 1:Ai.LP_nnz
            # LP_sp_index[lp_pos] is the problem-level LP scalar position
            # of this stored nonzero (compact storage: sorted nonzero
            # positions, NOT identity).
            global_k = Ai.LP_sp_index[lp_pos]
            push!(problem.LP_constraint[global_k], i)
            # C++ parity: the stored value is the global LP scalar index
            # (= global_k = the table's own outer index); consumers read
            # coefficients via get_lp_value(A[i], idx), never via this
            # table indexing LP_sp_val.
            push!(problem.LP_blockIndex[global_k], global_k)
            problem.LP_nConstraint[global_k] += 1
        end
    end
    return problem
end

"""
    multi_inner_product_to_a!(ret, problem, xMat)

Compute `ret[i] = ⟨A_i, xMat⟩` for every constraint i (Frobenius inner
product summed over blocks). Mirrors C++ `InputData::multi_InnerProductToA`
at `sdpa_dataset.cpp:477-485`.

Used by:
  - residual computation: `primal residual = b - ⟨A_i, xMat⟩`
  - objective evaluation: `objValPrimal = ⟨C, xMat⟩` (called with C
    reformatted as a sparse A_i and a single-element ret)

xMat must be a DenseLinearSpace (i.e. an iterate, not a constraint).
"""
function multi_inner_product_to_a!(
    ret::SDPAVector{T}, problem::SDPAProblem{T}, xMat::SDPADenseLinearSpace{T}
) where {T<:Real}
    length(ret) == problem.m ||
        throw(DimensionMismatch("ret length != m"))
    fill!(ret.ele, zero(T))
    # Thread over the outer i loop. Each iteration writes ret.ele[i]
    # from exactly one task (output partitioned by i); xMat is read-only.
    # Bit-identical to the serial version because there is no shared
    # write target and no reduction.
    @tasks for i in 1:problem.m
        Ai = problem.A[i]
        s = zero(T)
        # SDP blocks (skip unallocated blocks — an unassigned
        # shell contributes nothing and must not throw).
        for sdp_k in 1:Ai.SDP_sp_nBlock
            if !isassigned(Ai.SDP_sp_block, sdp_k)
                continue
            end
            global_k = Ai.SDP_sp_index[sdp_k]
            xmat_block = _lookup_sdp_block(xMat, global_k, problem.blockStruct)
            Ai_block_dense = Ai.SDP_sp_block[sdp_k]
            if Ai_block_dense.type === :sparse
                for nz in 1:Ai_block_dense.nonZeroEffect
                    r = Ai_block_dense.row_index[nz]
                    c = Ai_block_dense.column_index[nz]
                    v = Ai_block_dense.sp_ele[nz]
                    s += v * xmat_block[r, c]
                    if r != c
                        s += v * xmat_block[c, r]
                    end
                end
            else
                if size(Ai_block_dense.de_ele) == size(xmat_block.de_ele)
                    @inbounds for k in eachindex(Ai_block_dense.de_ele)
                        s += Ai_block_dense.de_ele[k] * xmat_block.de_ele[k]
                    end
                else
                    throw(DimensionMismatch(
                        "A_i block $sdp_k size $(size(Ai_block_dense.de_ele)) != xMat block $global_k size $(size(xmat_block.de_ele))"
                    ))
                end
            end
        end
        # LP blocks: iterate the compact nonzero lists only.
        for lp_pos in 1:Ai.LP_nnz
            lp_k = Ai.LP_sp_index[lp_pos]
            s += Ai.LP_sp_val[lp_pos] * xMat.LP_block[lp_k]
        end
        ret.ele[i] = s
    end
    return ret
end

"""
    _lookup_sdp_block(xMat, global_k, blockStruct)

Return the entry of `xMat.SDP_block` corresponding to the problem-level
block index `global_k`. `xMat` was constructed from `blockStruct`; its
SDP_block list is in the order of the positive entries of `blockStruct`
(see DenseLinearSpace(blockStruct) constructor). So we need to map
`global_k` (1..nBlock problem index) to its position in xMat.SDP_block
(which is the count of bs >= 2 up to and including global_k).
"""
function _lookup_sdp_block(
    xMat::SDPADenseLinearSpace{T}, global_k::Int, blockStruct::AbstractVector{Int}
) where {T<:Real}
    pos = 0
    for k in 1:global_k
        if blockStruct[k] >= 2
            pos += 1
        end
    end
    return xMat.SDP_block[pos]
end

"""
    multi_plus_to_a!(retMat, problem, xVec)

Compute `retMat = Σ_i xVec[i] * A_i` (block-wise dense accumulation).
Mirrors C++ `InputData::multi_plusToA` at `sdpa_dataset.cpp:487-489`.

Used by:
  - residual computation: `dual residual = C - Σ_i yVec[i] * A_i`
"""
function multi_plus_to_a!(
    retMat::SDPADenseLinearSpace{T}, problem::SDPAProblem{T}, xVec::SDPAVector{T}
) where {T<:Real}
    # Deliberately serial: threading this needs per-task retMat buffers
    # plus a reduction, and the block count is small.
    length(xVec) == problem.m ||
        throw(DimensionMismatch("xVec length != m"))
    setzero!(retMat)
    for i in 1:problem.m
        Ai = problem.A[i]
        xi = xVec.ele[i]
        # SDP blocks.
        for sdp_k in 1:Ai.SDP_sp_nBlock
            if !isassigned(Ai.SDP_sp_block, sdp_k)
                continue
            end
            global_k = Ai.SDP_sp_index[sdp_k]
            ret_block = _lookup_sdp_block(retMat, global_k, problem.blockStruct)
            Ai_block = Ai.SDP_sp_block[sdp_k]
            if Ai_block.type === :sparse
                for nz in 1:Ai_block.nonZeroEffect
                    r = Ai_block.row_index[nz]
                    c = Ai_block.column_index[nz]
                    v = Ai_block.sp_ele[nz]
                    ret_block.de_ele[r, c] += xi * v
                    # Symmetric fill: if off-diagonal, also fill (c, r) so the
                    # accumulated dense matrix is symmetric. The C++ stores
                    # only the upper triangle of A_i but operates on the full
                    # symmetric matrix; mirror that here.
                    if r != c
                        ret_block.de_ele[c, r] += xi * v
                    end
                end
            else
                # Validate like the sibling multi_inner_product_to_a!
                # dense branch — @inbounds with mismatched sizes reads OOB
                # memory instead of failing loudly.
                if size(Ai_block.de_ele) == size(ret_block.de_ele)
                    @inbounds for k in eachindex(Ai_block.de_ele)
                        ret_block.de_ele[k] += xi * Ai_block.de_ele[k]
                    end
                else
                    throw(DimensionMismatch(
                        "A_i block $sdp_k size $(size(Ai_block.de_ele)) != retMat block $global_k size $(size(ret_block.de_ele))"
                    ))
                end
            end
        end
        # LP blocks: iterate the compact nonzero lists only.
        for lp_pos in 1:Ai.LP_nnz
            lp_k = Ai.LP_sp_index[lp_pos]
            retMat.LP_block[lp_k] += xi * Ai.LP_sp_val[lp_pos]
        end
    end
    return retMat
end
