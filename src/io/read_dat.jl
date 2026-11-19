"""
    read_dat(path, T=Float64)

Parse an SDPA dense-format `.dat` file into an `SDPAProblem{T}`.

The dense format is structurally similar to `.dat-s` for the header
(mDIM, nBLOCK, bLOCKsTRUCT, b vector), but differs in the body:

```
<comment (optional)>
<mDIM>
<nBLOCK>
<bLOCKsTRUCT>          signed block sizes
<b vector>              m floats
<C>                     nBlock dense matrices, each fully written
<A_1>                   nBlock dense matrices for constraint 1
<A_2>
...
<A_m>
```

Each dense matrix is written as a full `size × size` block (row-major
inside the file, but read into Julia's column-major `Matrix{T}`).
Symmetric duplicates are stored once (upper triangle only) to match
the sparse parser's convention.

Mirrors C++ `IO::read` (the `isDataSparse = false` branch) at
`sdpa_io.cpp:495-560`.

The reader uses the same format-tolerant token-stream approach as
`read_dat_s`: lines starting with `"` or `*` are comments; non-numeric
text (`{`, `}`, `,`) between values is silently skipped.

The resulting `SDPAProblem{T}` is structurally identical to what
`read_dat_s` would have produced from the equivalent `.dat-s` file.
Equivalence contract: `read_dat('example1.dat', T)` and
`read_dat_s('example1.dat-s', T)` produce problems with `b`, `C`, `A`
all equal to within `eps(T)`.
"""
function read_dat(
    path::AbstractString, ::Type{T}=Float64
) where {T<:Real}
    lines = readlines(path)
    while !isempty(lines) && isempty(strip(lines[end]))
        pop!(lines)
    end

    data_lines = String[]
    for line in lines
        stripped = lstrip(line)
        if isempty(stripped)
            continue
        end
        first_char = first(stripped)
        if first_char === '"' || first_char === '*'
            continue
        end
        push!(data_lines, rstrip(line))
    end

    # Accept degenerate headers — mDIM=0 (no b vector) and
    # nBLOCK=0 (no blockStruct line) files legitimately have only 3 data
    # lines, and the fully-degenerate conjunction mDIM=0 AND nBLOCK=0 has
    # just 2 ("0" / "0"). Gate on >= 2: any file passing it guarantees a
    # line 2 exists (the 1-token-header branch reads line_tokens[2]), and
    # the remaining truncations are caught by the header checks below.
    length(data_lines) >= 2 ||
        error("SDPA .dat file needs >= 2 non-comment lines (mDIM, nBLOCK[, blockStruct/b]), got $(length(data_lines))")

    # Tokenize: extract leading numeric prefix from each whitespace- or
    # punctuation-separated word. Drops `{`, `}`, `,`, `=`.
    numeric_prefix = r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    tokens = String[]
    token_line = Int[]
    line_tokens = Vector{Vector{String}}()
    for (ln, line) in enumerate(data_lines)
        # Replace common brace/comma punctuation with spaces so they don't
        # contaminate the word boundaries.
        cleaned = replace(line, r"[{},]" => " ")
        lt = String[]
        for word in split(cleaned; keepempty=false)
            m = match(numeric_prefix, word)
            if m !== nothing
                push!(lt, m.match)
                push!(tokens, m.match)
                push!(token_line, ln)
            end
        end
        push!(line_tokens, lt)
    end

    # Line-anchored header (mirrors read_dat_s): mDIM and nBLOCK
    # may share line 1 (2 tokens) or sit on their own lines (1 token
    # each); bLOCKsTRUCT must carry exactly nBLOCK tokens on the following
    # line.
    if length(line_tokens[1]) == 2
        n_header = 2
    elseif length(line_tokens[1]) == 1
        length(line_tokens[2]) == 1 ||
            error("line 2: expected exactly 1 token for nBLOCK, got $(length(line_tokens[2]))")
        n_header = 3
    else
        error("line 1: expected 1 or 2 token(s) for mDIM [nBLOCK], got $(length(line_tokens[1]))")
    end
    token_idx = 1
    mDim, token_idx = _read_int(tokens, token_idx; where="mDIM", path=path)
    nBlock, token_idx = _read_int(tokens, token_idx; where="nBLOCK", path=path)
    mDim >= 0 || error("mDIM must be non-negative, got $mDim")
    nBlock >= 0 || error("nBLOCK must be non-negative, got $nBlock")
    # With nBLOCK=0 the blockStruct line may be absent entirely, so
    # the line-anchored count check only applies when there are blocks.
    # The header check is unchanged for nBlock > 0.
    (nBlock == 0 || length(line_tokens[n_header]) == nBlock) ||
        error("line $n_header: expected $nBlock token(s) for bLOCKsTRUCT, got $(length(line_tokens[n_header]))")

    blockStruct = Vector{Int}(undef, nBlock)
    for k in 1:nBlock
        v, token_idx = _read_int(tokens, token_idx; where="bLOCKsTRUCT[$k]", path=path)
        blockStruct[k] = v
    end

    # Fail fast on a truncated file BEFORE allocating mDim-sized
    # storage (mirrors read_dat_s.jl) — the header promises mDim b-vector
    # values, so at least that many tokens must remain.
    remaining = length(tokens) - token_idx + 1
    remaining >= mDim ||
        error("truncated .dat file $(path): header declares mDIM=$mDim " *
              "but only $remaining token(s) remain for the b vector (b[1:$mDim])")

    b_values = Vector{T}(undef, mDim)
    for k in 1:mDim
        v, token_idx = _read_value(T, tokens, token_idx; where="b[$k]", path=path)
        b_values[k] = v
    end

    # Classify blocks.
    sdp_indices = findall(bs -> bs >= 2, blockStruct)
    nSDP = length(sdp_indices)
    # LP scalar count: each LP block of size |bs| contributes |bs| separate
    # LP scalars (e.g. blockStruct=[2,-3] → 3 LP variables, not 1).
    # Matches read_dat_s.jl semantics. init=0: nBlock=0 gives an empty
    # blockStruct.
    nLP  = lp_scalar_count(blockStruct)

    b_vec = SDPAVector{T}(b_values)

    function make_empty_space()
        S = SDPASparseLinearSpace{T}(nSDP, nLP)
        copyto!(S.SDP_sp_index, sdp_indices)
        # LP_sp_index is compact: nonzero LP positions are inserted by
        # setelement_lp! as entries are read. The stored index values are
        # the problem-level LP scalar positions.
        for k in 1:nSDP
            bs = blockStruct[sdp_indices[k]]
            S.SDP_sp_block[k] = SDPASparseMatrix{T}(bs, bs, :sparse, 0)
        end
        return S
    end

    C = make_empty_space()
    A = [make_empty_space() for _ in 1:mDim]

    # Read C: nBlock dense matrices in problem-block order. Each SDP
    # block has bs*bs entries (full matrix, including symmetric dupes);
    # we store only the upper-triangle nonzeros (i <= j). Each LP block
    # has |bs| entries.
    #
    # Each SDP block is buffered in full so the symmetric duplicate
    # can be compared against the kept upper-triangle value; a mismatched
    # lower triangle is still discarded (C++ fscanf parity — semantics
    # unchanged) but the first violation per file is reported via @warn
    # after the parse instead of silently dropping data.
    local asym_first = nothing  # (label, block, i, j, upper, lower)
    local asym_count = 0
    function read_dense_into!(
        target::SDPASparseLinearSpace{T}, label::String
    ) where {T<:Real}
        sdp_pos = 0
        lp_offset = 0
        for k in 1:nBlock
            bs = blockStruct[k]
            if bs >= 2
                sdp_pos += 1
                # Buffer the full block (row-major file order preserved:
                # i outer, j inner), then extract the upper triangle.
                buf = Matrix{T}(undef, bs, bs)
                for i in 1:bs, j in 1:bs
                    v, token_idx = _read_value(
                        T, tokens, token_idx;
                        where="$(label) block $(k) [$(i),$(j)]", path=path,
                    )
                    buf[i, j] = v
                end
                # Store only upper triangle (i <= j) and nonzeros,
                # matching the sparse parser's stored form.
                for i in 1:bs, j in i:bs
                    v = buf[i, j]
                    if v != 0
                        # Pre-allocate capacity to avoid per-entry resize.
                        block = target.SDP_sp_block[sdp_pos]
                        if block.nonZeroEffect == block.nonZeroCount
                            new_cap = max(8, block.nonZeroCount * 2)
                            resize!(block.row_index, new_cap)
                            resize!(block.column_index, new_cap)
                            resize!(block.sp_ele, new_cap)
                            block.nonZeroCount = new_cap
                        end
                        block.nonZeroEffect += 1
                        idx = block.nonZeroEffect
                        block.row_index[idx] = i
                        block.column_index[idx] = j
                        block.sp_ele[idx] = v
                    end
                end
                # Flag lower-triangle values that differ from their
                # mirrored upper entry (they are silently discarded by
                # the upper-triangle keep above).
                for i in 2:bs, j in 1:i-1
                    if buf[i, j] != buf[j, i]
                        asym_count += 1
                        if asym_first === nothing
                            asym_first = (
                                label=label, block=k, i=i, j=j,
                                upper=buf[j, i], lower=buf[i, j],
                            )
                        end
                    end
                end
            else
                # LP block: |bs| entries, store nonzeros at absolute position.
                lp_size = abs(bs)
                for lp_k in 1:lp_size
                    v, token_idx = _read_value(
                        T, tokens, token_idx;
                        where="$(label) LP block $(k) [$(lp_k)]", path=path,
                    )
                    if v != 0
                        setelement_lp!(target, lp_offset + lp_k, v)
                    end
                end
                lp_offset += lp_size
            end
        end
    end

    read_dense_into!(C, "C")
    for k in 1:mDim
        read_dense_into!(A[k], "A[$k]")
    end

    if token_idx <= length(tokens)
        error(
            "$(path): $(length(tokens) - token_idx + 1) trailing token(s) " *
            "after the last entry (truncated file?)",
        )
    end

    # Semantics: the upper triangle is kept; the discarded
    # data is now reported instead of silent.
    if asym_count > 0
        @warn "read_dat: $(path): dense input is asymmetric — taking the upper triangle" first_violation=asym_first n_violations=asym_count
    end

    return _finalize_problem!(mDim, nBlock, blockStruct, b_vec, C, A, T)
end
