"""
    parse_dat_s_header(path::AbstractString)

Parse ONLY the header (comment line, mDIM, nBLOCK, bLOCKsTRUCT) of an
SDPA sparse `.dat-s` file. Returns a NamedTuple `(comment, mDim, nBlock,
blockStruct)`.

Both on-disk layouts are detected:

* annotated (SDPA example files): optional comment line(s) starting with
  `"` or `*`, then three `value = key` lines — `   3  =  mDIM` — with
  the value taken from before the `=`.
* plain (SDPLIB): NO comment line — three bare lines: mDIM, nBLOCK,
  bLOCKsTRUCT values. `comment` is `""` for such files.

Detection: after any comment lines, the first header line decides — it
contains `=` (annotated) or is a bare integer (plain). On the plain path
the bLOCKsTRUCT line must carry exactly nBLOCK tokens; a count mismatch
errors descriptively. A file with fewer than 3 header lines errors as
too short instead of throwing a context-free ArgumentError/EOFError.

Header-only fast reader: it does NOT parse the b-vector or the
matrix entries. The full sparse parser is `read_dat_s`.
"""
function parse_dat_s_header(path::AbstractString)
    lines = readlines(path)

    # Collect leading comment lines ('"' or '*' prefix — the same rule
    # read_dat_s uses). The plain SDPLIB layout has none.
    first_data = findfirst(lines) do line
        s = lstrip(line)
        return !(startswith(s, "\"") || startswith(s, "*"))
    end
    comment = first_data === nothing ? join(lines, "\n") :
              join(lines[1:first_data-1], "\n")
    data = first_data === nothing ? String[] : lines[first_data:end]

    # Plain needs 3 header lines; annotated needs a comment + 3 (a
    # comment-less annotated file still has its 3 lines). Anything
    # shorter cannot hold a header.
    length(data) >= 3 ||
        error("smoke_parser: $(path): file too short for a .dat-s header (need comment + 3 header lines)")

    if occursin('=', data[1])
        # Annotated layout, e.g. "   3  =  mDIM" — value before '='.
        mDim = _smoke_header_int(path, strip(split(data[1], '=')[1]))
        nBlock = _smoke_header_int(path, strip(split(data[2], '=')[1]))
        block_struct_str = strip(split(data[3], '=')[1])
        blockStruct = [_smoke_header_int(path, s) for s in split(block_struct_str)]
    else
        # Plain SDPLIB layout: bare mDIM / nBLOCK / bLOCKsTRUCT lines.
        mDim = _smoke_header_int(path, strip(data[1]))
        nBlock = _smoke_header_int(path, strip(data[2]))
        toks = split(strip(data[3]))
        length(toks) == nBlock || error(
            "smoke_parser: $(path): header line 3: expected $(nBlock) " *
            "bLOCKsTRUCT token(s) for nBLOCK=$(nBlock), got $(length(toks))",
        )
        blockStruct = [_smoke_header_int(path, s) for s in toks]
    end
    return (comment=comment, mDim=mDim, nBlock=nBlock, blockStruct=blockStruct)
end

"""
    _smoke_header_int(path, s) -> Int

Parse a smoke-parser header token as an integer, erroring with the file
name and offending token instead of a context-free ArgumentError.
"""
function _smoke_header_int(path::AbstractString, s::AbstractString)
    return try
        parse(Int, s)
    catch err
        err isa InterruptException && rethrow()
        error("smoke_parser: $(path): header token is not an integer: \"$(s)\"")
    end
end
