"""
    read_dat_s(path, T=Float64)

Parse an SDPA sparse-format `.dat-s` file into an `SDPAProblem{T}`.

Format (per https://sdpa.github.io/format.html and C++ `IO::read` at
`sdpa_io.cpp:120-170`):

The parser is **format-tolerant**: it accepts both the canonical plain
layout (used by SDPLIB) and the annotated "verbose" layout (used by
SDPA's example files). The header is line-strict (one token for
mDIM and nBLOCK, exactly nBLOCK tokens for bLOCKsTRUCT) and the entry
region is validated per line (every line after the b-vector must contain
only whole 5-tuples), so truncation is reported with a line number
instead of a downstream token shift. The b-vector itself may span lines
(the NSPS benchmark writes ~7 values per line); on the line where it
ends, the tokens after the b tail must still form whole 5-tuples —
a stray extra b token there is rejected with its line number. The
difference:

```
Verbose:                    Plain (SDPLIB):
  "comment"                   174
  3  =  mDIM                  2
  1  =  nBLOCK                161 -174
  2  = bLOCKsTRUCT            2.0 2.0 ... (m floats)
  48, -8, 20                  0 1 2 2 1.0
  0 1 1 1 -11                 ...
```

Lines starting with `"` or `*` are treated as comments. Numeric values
are read token-by-token (whitespace- and newline-separated), ignoring
any non-numeric trailing characters like `= mDIM`. This matches the
C++ `fscanf("%d", ...)` and `gmp_fscanf("%Fe", ...)` semantics.

Entry lines (after the b vector) have format:
  `mat block i j value` where mat=0 → C, mat=k≥1 → A_k.

For `T = BigFloat`, every value is parsed via `parse(BigFloat, str)`
directly (NOT via Float64 then convert) so the full string precision
is preserved.

The parser partitions blocks into SDP (`blockStruct[k] >= 2`) vs LP
(`blockStruct[k] < 2`) and sets `SDP_sp_index` / `LP_sp_index`
correctly so that the `SparseLinearSpace` block indices map to the
original problem-block indices (matching the convention used by
`SDPAProblem.initialize_index!`). This mirrors C++
`IO::setBlockStruct` at `sdpa_io.cpp:428-540`.

After parsing, every block of every `C` and `A_i` is conditionally
converted to dense via `change_to_dense!` using its native 0.20
density gate (matching C++ `sdpa_main.cpp:197-199`).
"""
function read_dat_s(
    path::AbstractString, ::Type{T}=Float64
) where {T<:Real}
    lines = readlines(path)
    # Drop trailing empty lines.
    while !isempty(lines) && isempty(strip(lines[end]))
        pop!(lines)
    end

    # First pass: filter out comment lines (start with `"` or `*`).
    data_lines = String[]
    for line in lines
        stripped = lstrip(line)
        if isempty(stripped)
            continue  # blank lines are skipped
        end
        first_char = first(stripped)
        if first_char === '"' || first_char === '*'
            continue  # comment line
        end
        push!(data_lines, rstrip(line))
    end

    # Accept degenerate headers — mDIM=0 (no b vector) and
    # nBLOCK=0 (no blockStruct line) files legitimately have only 3 data
    # lines, and the fully-degenerate conjunction mDIM=0 AND nBLOCK=0 has
    # just 2 ("0" / "0"). Gate on >= 2: any file passing it guarantees a
    # line 2 exists (the 1-token-header branch reads line_tokens[2]), and
    # the remaining truncations are caught by the header checks below. A
    # 1-line combined header ("0 0") is rejected here by design — it
    # carries no blockStruct or b content at all.
    length(data_lines) >= 2 ||
        error("SDPA .dat-s file needs >= 2 non-comment lines (mDIM, nBLOCK[, blockStruct/b]), got $(length(data_lines))")

    # Second pass: tokenize the data lines into per-line token lists plus
    # a flat stream with each token's source line. For each
    # whitespace-separated word, extract the leading numeric portion (with
    # optional sign, decimal point, and exponent), dropping any trailing
    # "= key" annotation. This mirrors C++ fscanf("%d") and
    # gmp_fscanf("%Fe") semantics, which scan forward to the next numeric
    # character and parse from there.
    numeric_prefix = r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    tokens = String[]
    token_line = Int[]
    line_tokens = Vector{Vector{String}}()
    for (ln, line) in enumerate(data_lines)
        lt = String[]
        for word in split(line; keepempty=false)
            # Some words carry brace/comma punctuation (e.g. "48," from a
            # comma-packed b vector, "{0.5" from a brace-packed one).
            # Strip it so numbers glued to punctuation are not silently
            # dropped (parity with read_dat.jl's cleanup).
            word = replace(word, r"[{},]" => " ")
            for sub in split(word; keepempty=false)
                m = match(numeric_prefix, sub)
                if m !== nothing
                    push!(lt, m.match)
                    push!(tokens, m.match)
                    push!(token_line, ln)
                end
                # Words like "=" or "mDIM" produce no token; they're skipped.
            end
        end
        push!(line_tokens, lt)
    end

    # Line-anchored header: mDIM and nBLOCK may share line 1
    # (2 tokens) or sit on their own lines (1 token each; canonical
    # SDPLIB/example/benchmark layout) — both layouts are accepted, and
    # bLOCKsTRUCT must carry exactly nBLOCK tokens on the following line.
    # A truncated header is then reported here, not as a downstream token
    # shift.
    if length(line_tokens[1]) == 2
        n_header = 2   # bLOCKsTRUCT is line 2
    elseif length(line_tokens[1]) == 1
        length(line_tokens[2]) == 1 ||
            error("line 2: expected exactly 1 token for nBLOCK, got $(length(line_tokens[2]))")
        n_header = 3   # bLOCKsTRUCT is line 3
    else
        error("line 1: expected 1 or 2 token(s) for mDIM [nBLOCK], got $(length(line_tokens[1]))")
    end
    token_idx = 1
    mDim, token_idx = _read_int(tokens, token_idx; where="mDIM", path=path)
    nBlock, token_idx = _read_int(tokens, token_idx; where="nBLOCK", path=path)
    mDim >= 0 || error("mDIM must be non-negative, got $mDim")
    nBlock >= 0 || error("nBLOCK must be non-negative, got $nBlock")
    # With nBLOCK=0 the blockStruct line may be absent entirely (a
    # blank line was already filtered out), so the line-anchored count
    # check only applies when there are blocks to describe. The header
    # unchanged for nBlock > 0.
    (nBlock == 0 || length(line_tokens[n_header]) == nBlock) ||
        error("line $n_header: expected $nBlock token(s) for bLOCKsTRUCT, got $(length(line_tokens[n_header]))")

    # Read nBlock blockStruct values.
    blockStruct = Vector{Int}(undef, nBlock)
    for k in 1:nBlock
        v, token_idx = _read_int(tokens, token_idx; where="bLOCKsTRUCT[$k]", path=path)
        blockStruct[k] = v
    end

    # LP prefix offsets, computed once — lp_prefix[k] is the number
    # of LP scalars contained in blocks BEFORE block k (blocks with
    # bs < 2 contribute |bs| scalars each; SDP blocks contribute 0).
    # Recomputing this prefix sum per entry made LP-heavy files parse in
    # O(nEntries × nBlock); C++ precomputes the equivalent table.
    lp_prefix = Vector{Int}(undef, nBlock + 1)
    lp_prefix[1] = 0
    for k in 1:nBlock
        lp_prefix[k+1] = lp_prefix[k] + (blockStruct[k] < 2 ? abs(blockStruct[k]) : 0)
    end

    # Fail fast on a truncated file BEFORE allocating mDim-sized
    # storage — the header promises mDim b-vector values, so at least that
    # many tokens must remain. A corrupt mDIM (e.g. 1e9 from a damaged
    # header) would otherwise allocate huge storage first and only
    # error later with a context-free message.
    remaining = length(tokens) - token_idx + 1
    remaining >= mDim ||
        error("truncated .dat-s file $(path): header declares mDIM=$mDim " *
              "but only $remaining token(s) remain for the b vector (b[1:$mDim])")

    # Read mDim b values. The b-vector may span lines (NSPS writes ~7
    # values per line), so it is consumed from the flat stream; a truncated
    # b that stops mid-line is caught by the per-line entry validation
    # below (the leftover partial tuple cannot form whole 5-tuples).
    b_start_idx = token_idx
    b_values = Vector{T}(undef, mDim)
    for k in 1:mDim
        v, token_idx = _read_value(T, tokens, token_idx; where="b[$k]", path=path)
        b_values[k] = v
    end
    # Every line AFTER the b-completion line must contain only whole
    # 5-tuples. The b-completion line itself is exempt (the b-vector may
    # end mid-line with entries continuing on the same line).
    first_entry_line = token_idx <= length(tokens) ? token_line[token_idx] : length(line_tokens) + 1
    for ln in (first_entry_line + 1):length(line_tokens)
        length(line_tokens[ln]) % 5 == 0 ||
            error(
                "line $ln: $(length(line_tokens[ln])) token(s) — entries must " *
                "come in whole 5-tuples (mat block i j value); truncated " *
                "b-vector or entry?",
            )
    end

    # The b-completion line is the ONE line exempt from the %5 check
    # above, so a stray extra b token there is consumed as the first
    # entry's mat index and shifts the parse. With mDim known the tokens
    # are attributable: the b tokens on that line are those the b loop
    # consumed there (global indices < token_idx); the REMAINDER of the
    # line is entry tokens, which must come in whole 5-tuples. Skipped
    # when there is no b vector (mDIM=0) or nothing after it (no
    # entries — leftovers there are caught by the trailing-token check).
    if mDim > 0 && token_idx <= length(tokens)
        completion_ln = token_line[token_idx - 1]
        b_on_completion = count(==(completion_ln), view(token_line, b_start_idx:token_idx - 1))
        line_total = length(line_tokens[completion_ln])
        remainder = line_total - b_on_completion
        remainder % 5 == 0 ||
            error(
                "$(path): line $(completion_ln): b-vector ends mid-line and " *
                "the $(remainder) remaining token(s) on that line are not " *
                "a whole multiple of 5 — entries must come in whole 5-tuples " *
                "(mat block i j value); line has $(line_total) token(s), " *
                "$(b_on_completion) attributed to b (stray b token or " *
                "truncated entry?)",
            )
    end

    # Classify blocks: SDP (bs >= 2) vs LP (bs < 2).
    # LP scalar count: each LP block of size |bs| contributes |bs|
    # separate LP scalars. A blockStruct entry of -174 means 174 LP
    # variables, not 1. This matches C++ conventions.
    sdp_indices = findall(bs -> bs >= 2, blockStruct)
    nSDP = length(sdp_indices)
    # init=0: nBlock=0 gives an empty blockStruct.
    nLP  = lp_scalar_count(blockStruct)

    b_vec = SDPAVector{T}(b_values)

    function make_empty_space()
        S = SDPASparseLinearSpace{T}(nSDP, nLP)
        copyto!(S.SDP_sp_index, sdp_indices)
        # LP_sp_index is compact: nonzero LP positions are inserted by
        # setelement_lp! as entries are read (C++ SparseLinearSpace keeps
        # only nonzero LP blocks, sdpa_struct.cpp:948-972). The stored
        # index values are the problem-level LP scalar positions.
        for k in 1:nSDP
            bs = blockStruct[sdp_indices[k]]
            S.SDP_sp_block[k] = SDPASparseMatrix{T}(bs, bs, :sparse, 0)
        end
        return S
    end

    C = make_empty_space()
    A = [make_empty_space() for _ in 1:mDim]

    # Parse the remaining tokens as 5-tuples (mat, block, i, j, value).
    # Reject duplicate (mat, block, i, j) positions — the sparse
    # path sums duplicates while the dense conversion last-wins, so a
    # repeated position silently produces path-dependent results (C++
    # inherits the same per-path behavior; we fail loudly instead).
    seen_entries = Set{UInt128}()
    while token_idx + 4 <= length(tokens)
        mat_idx, token_idx = _read_int(tokens, token_idx; where="entry mat index", path=path)
        block_idx, token_idx = _read_int(tokens, token_idx; where="entry block index", path=path)
        i, token_idx = _read_int(tokens, token_idx; where="entry row i", path=path)
        j, token_idx = _read_int(tokens, token_idx; where="entry column j", path=path)
        value, token_idx = _read_value(T, tokens, token_idx; where="entry value", path=path)

        1 <= block_idx <= nBlock ||
            error("$(path): block_idx $block_idx out of [1, $nBlock]")
        mat_idx >= 0 ||
            error("$(path): mat_idx must be >= 0 (0=C, k=A_k); got $mat_idx")
        mat_idx <= mDim ||
            error("$(path): mat_idx $mat_idx out of [0, $mDim]")

        entry_key = (UInt128(mat_idx) << 96) |
                    (UInt128(block_idx) << 64) |
                    (UInt128(i) << 32) | UInt128(j)
        if entry_key in seen_entries
            error(
                "$(path): duplicate entry (mat=$mat_idx block=$block_idx i=$i j=$j)",
            )
        end
        push!(seen_entries, entry_key)

        bs = blockStruct[block_idx]
        target = (mat_idx == 0) ? C : A[mat_idx]
        if bs >= 2
            # SDP block. Validate matrix-entry indices with file context
            # (mirrors the C++ bounds-check at sdpa_struct.cpp:1276-1278 /
            # 1612-1614, which hard-errors with rError). `setelement_sdp!`
            # cannot see `bs` (the lazy first-access block is (0,0)), so
            # the reader validates against the declared block size.
            if !(1 <= i <= bs) || !(1 <= j <= bs) || !(i <= j)
                error(
                    "$(path): SDP entry (mat=$mat_idx block=$block_idx) " *
                    "invalid indices i=$i j=$j for block size $bs (need 1<=i<=j<=$bs)",
                )
            end
            # Find its position in sdp_indices.
            pos = findfirst(==(block_idx), sdp_indices)
            # error(), not @assert: reachable from file input —
            # @assert can be elided at higher optimization levels.
            pos !== nothing || error("SDP block $block_idx not in sdp_indices")
            setelement_sdp!(target, pos, i, j, value)
        else
            # LP block: each element j is a separate LP scalar within
            # the expanded LP array. Position = offset for this block + j.
            i == j ||
                error(
                    "$(path): LP entry (mat=$mat_idx block=$block_idx) " *
                    "must have i == j, got i=$i, j=$j",
                )
            1 <= j <= abs(bs) ||
                error(
                    "$(path): LP entry (mat=$mat_idx block=$block_idx) " *
                    "index j=$j out of [1, $(abs(bs))] (LP block size $(abs(bs)), i==j required)",
                )
            # O(1) lookup into the precomputed prefix table replaces
            # the per-entry O(nBlock) prefix scan (values identical).
            lp_pos = lp_prefix[block_idx] + j
            setelement_lp!(target, lp_pos, value)
        end
    end

    if token_idx <= length(tokens)
        error(
            "$(path): $(length(tokens) - token_idx + 1) trailing token(s) " *
            "after the last entry (truncated file?)",
        )
    end

    return _finalize_problem!(mDim, nBlock, blockStruct, b_vec, C, A, T)
end

"""
    _io_error_prefix(path, where, idx)

Build the common error prefix naming the file, the field being read
(`where`, e.g. "mDIM" or "b[3]"), and the flat-stream token position, so
truncation and parse failures are attributable instead of context-free.
"""
function _io_error_prefix(path::AbstractString, where::AbstractString, idx::Int)
    io = IOBuffer()
    if !isempty(path)
        print(io, path, ": ")
    end
    if !isempty(where)
        print(io, "while reading ", where, " (token #", idx, "): ")
    end
    return String(take!(io))
end

"""
    _read_int(tokens, idx; where="", path="")

Read a single integer from `tokens` at position `idx`. Returns
`(value, next_idx)`. Mirrors C++ `fscanf("%d", ...)`. Errors are prefixed
with the optional `where`/`path` context.
"""
function _read_int(tokens::Vector{String}, idx::Int; where::AbstractString="", path::AbstractString="")
    prefix = _io_error_prefix(path, where, idx)
    if idx > length(tokens)
        error(prefix * "unexpected end of token stream while reading integer")
    end
    value = try
        parse(Int, tokens[idx])
    catch err
        err isa InterruptException && rethrow()
        error(prefix * sprint(showerror, err))
    end
    return value, idx + 1
end

"""
    _read_value(T, tokens, idx; where="", path="")

Read a single numeric value of type `T` from `tokens` at position
`idx`. Returns `(value, next_idx)`. Mirrors C++ `gmp_fscanf("%Fe", ...)`.
Errors are prefixed with the optional `where`/`path` context.
"""
function _read_value(::Type{T}, tokens::Vector{String}, idx::Int; where::AbstractString="", path::AbstractString="") where {T<:Real}
    prefix = _io_error_prefix(path, where, idx)
    if idx > length(tokens)
        error(prefix * "unexpected end of token stream while reading value")
    end
    value = try
        parse(T, tokens[idx])
    catch err
        err isa InterruptException && rethrow()
        error(prefix * sprint(showerror, err))
    end
    return value, idx + 1
end
