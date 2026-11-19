"""
    read_ini_s(path::AbstractString, problem::SDPAProblem{T}) -> SDPASolution{T}

Read an SDPA-GMP initial-point file (`.ini-s`) and return a fully
initialized `SDPASolution{T}` (X, y, Z filled; Cholesky inverses and
`invzMat` computed) suitable for warm-starting `solve!` via the `init`
keyword.

**File format** — two layouts are auto-detected:

    line 1:  the dual vector y — `m` numeric tokens (comma/space
             separated). Stored NEGATED: the file carries the
             SCS/MOSEK-sign y and the reader stores `y = -file_value`
             (C++: "yVec is opposite sign").

    then, EITHER the C++ sdpa-gmp layout (sdpa_io.cpp:150-200) as
    5-tuples `target l i j value`:
               target = 1 → zMat, otherwise → xMat
               l      = problem block index, 1..nBlock
               i, j   = 1-based positions; SDP blocks give the upper
                        triangle (i <= j), symmetric fill; LP blocks
                        require i == j (LP-scalar index within the block)
               value  = scalar
    OR the NSPS-exporter layout as 4-tuples `l i j value`:
               l      = matrix block label: 1..nBlock for Z,
                        nBlock+1..2·nBlock for X
               i, j, value as above.

    Detection: the C++ layout is used when the remaining token count is
    divisible by 5 but NOT by 4; the exporter layout when divisible by 4
    (a count divisible by both prefers the exporter layout — compatibility default).
    Missing entries stay zero (e.g. X's SDP off-diagonals) — but an ini-s
    with NO X entries yields X = 0 and always fails: X is never
    regularized (only Z is), and an interior-point warm start needs
    X ≻ 0.

After filling, `compute_inverse!` recomputes `invCholeskyX`,
`invCholeskyZ`, `invzMat` (mirrors C++ `currentPt.computeInverse` after
`IO::read`, `sdpa_main.cpp:253-256`).

If the point is not strictly interior, it is regularized (LP scalars
floored at 1e-6; ε·I added to Z's SDP diagonals, ε up to 1e-1). Each
mechanism emits ONE `@warn` naming the file and the magnitude used
— the semantics are otherwise unchanged. X failing its Cholesky is NOT
    regularizable and errors immediately naming X. Duplicate `(vector,
block, i, j)` entries keep last-value-wins semantics and warn once with
the count.
"""
function read_ini_s(path::AbstractString, problem::SDPAProblem{T}) where {T<:Real}
    raw = read(path, String)
    toks = _ini_numeric_tokens(raw)
    m = problem.m
    nBlock = length(problem.blockStruct)
    length(toks) >= m ||
        error("read_ini_s: expected >= $m numeric tokens for yVec, got $(length(toks))")
    rest = toks[(m + 1):end]

    # Auto-detect the entry layout: C++ 5-tuple layout when the
    # count is divisible by 5 but not 4; NSPS-exporter 4-tuple layout when
    # divisible by 4 (a count divisible by both prefers the exporter
    # layout — compatibility default).
    if length(rest) % 5 == 0 && length(rest) % 4 != 0
        cpp_layout = true
    elseif length(rest) % 4 == 0
        cpp_layout = false
    else
        error(
            "read_ini_s: $(length(rest)) trailing tokens are not divisible " *
            "by 5 (target l i j value) or 4 (l i j value)",
        )
    end

    xMat = SDPADenseLinearSpace{T}(problem.blockStruct)
    zMat = SDPADenseLinearSpace{T}(problem.blockStruct)

    # Duplicate (matrix, block, i, j) entries keep last-value-wins
    # semantics (a plain overwrite), but silently clobbering hides
    # exporter bugs — count duplicates and warn once after the fill.
    seen_keys = Set{Tuple{Int,Int,Int,Int}}()
    n_duplicates = Ref(0)
    function note_entry!(mat_id::Int, blk_idx::Int, i::Int, j::Int)
        key = (mat_id, blk_idx, i, j)
        if key in seen_keys
            n_duplicates[] += 1
        else
            push!(seen_keys, key)
        end
        return nothing
    end

    # Shared entry filler: writes one (block, i, j) entry into target.
    # lbl is the raw block-label token, kept for error-message context.
    function fill_entry!(
        target::SDPADenseLinearSpace{T}, blk_idx::Int, i::Int, j::Int, v::T,
        lbl::Int,
    ) where {T<:Real}
        bs = problem.blockStruct[blk_idx]
        if bs >= 2
            # SDP block: position among SDP blocks; symmetric fill.
            # Validate matrix-entry indices with file context (mirrors the
            # C++ bounds-check at sdpa_struct.cpp:1276-1278 / 1612-1614).
            if !(1 <= i <= bs) || !(1 <= j <= bs) || !(i <= j)
                error(
                    "$(path): SDP entry (l=$lbl block=$blk_idx) " *
                    "invalid indices i=$i j=$j for block size $bs (need 1<=i<=j<=$bs)",
                )
            end
            sdp_pos = count(x -> x >= 2, problem.blockStruct[1:blk_idx])
            blk = target.SDP_block[sdp_pos].de_ele
            blk[i, j] = v
            # Independent mirror: sharing the SAME BigFloat object across
            # the transposed slots (A3) would corrupt later in-place MPFR
            # ops; value-copy via mp_set! into the pre-allocated slot.
            # For i == j the second write is a harmless self-copy.
            if T <: BigFloat
                mp_set!(blk[j, i], v)
            else
                blk[j, i] = v
            end
        else
            # LP block: scalar index = preceding LP scalars + i.
            i == j ||
                error("read_ini_s: LP block $blk_idx entry must have i == j, got i=$i j=$j")
            # Bounds-validate i with file context (mirrors the SDP branch).
            if !(1 <= i <= abs(bs))
                error(
                    "$(path): LP entry (l=$lbl block=$blk_idx) " *
                    "invalid index i=$i for LP block size $(abs(bs)) (need 1<=i<=$(abs(bs)))",
                )
            end
            lp_off = sum(abs(x) for x in problem.blockStruct[1:blk_idx-1] if x < 2; init=0)
            target.LP_block[lp_off + i] = v
        end
        return nothing
    end

    if cpp_layout
        # C++ sdpa-gmp layout: 5-tuples target l i j value.
        for k in 1:5:length(rest)
            target_k = parse(Int, rest[k])
            l = parse(Int, rest[k + 1])
            i = parse(Int, rest[k + 2])
            j = parse(Int, rest[k + 3])
            v = parse(T, rest[k + 4])
            1 <= l <= nBlock ||
                error("read_ini_s: block label $l out of [1, $nBlock]")
            (target_k == 1 || target_k == 2) ||
                error("read_ini_s: target $target_k must be 1 (zMat) or 2 (xMat)")
            note_entry!(target_k, l, i, j)
            fill_entry!(target_k == 1 ? zMat : xMat, l, i, j, v, l)
        end
    else
        # NSPS-exporter layout: 4-tuples l i j value with l in 1..2nBlock.
        for k in 1:4:length(rest)
            l = parse(Int, rest[k])
            i = parse(Int, rest[k + 1])
            j = parse(Int, rest[k + 2])
            v = parse(T, rest[k + 3])
            1 <= l <= 2 * nBlock ||
                error("read_ini_s: block label $l out of [1, $(2 * nBlock)]")
            mat_idx = (l - 1) ÷ nBlock          # 0 → zMat, 1 → xMat
            blk_idx = (l - 1) % nBlock + 1      # problem-block index
            note_entry!(mat_idx, blk_idx, i, j)
            fill_entry!(mat_idx == 0 ? zMat : xMat, blk_idx, i, j, v, l)
        end
    end

    # One warning for the whole file (not per entry) when duplicates
    # were silently overwritten.
    if n_duplicates[] > 0
        @warn "read_ini_s: $(path): $(n_duplicates[]) duplicate (vector,block,i,j) entries — last value wins" file=path n_duplicates=n_duplicates[]
    end

    # yVec: first m tokens, negated (C++ "yVec is opposite sign").
    yVec = SDPAVector{T}(m)
    for k in 1:m
        yVec.ele[k] = -parse(T, toks[k])
    end

    current_pt = SDPASolution{T}(m, problem.blockStruct, T(1))
    current_pt.xMat = xMat
    current_pt.zMat = zMat
    current_pt.yVec = yVec
    work = SDPAWorkVariables{T}(problem.blockStruct)
    ok = compute_inverse!(current_pt, work)
    if !ok
        # Regularize: MOSEK's dual slack Z is often indefinite at the
        # ~1e-7 level at an optimal point (negative LP entries / slightly
        # negative SDP eigenvalues), which makes the Cholesky-based
        # inverse computation fail. Standard warm-start practice is to
        # clip/shift the point to the interior:
        #   1. LP scalars: clip to a small positive floor.
        #   2. SDP blocks: add ε·I (ε increased until PD).
        # The shift is ~1e-7..1e-2 — negligible vs the λ=1e4 cold start,
        # and it makes the point usable by both the step-length eigen
        # machinery and the IPM proper. (C++ SDPA-GMP does NOT
        # regularize: its `computeInverse` failure is ignored in pinpal
        # and the first iteration runs on garbage inverses.)
        # Each mechanism warns once with the magnitude actually used.
        δ = T(1e-6)
        n_lp_floored = 0
        for k in eachindex(current_pt.zMat.LP_block)
            if current_pt.zMat.LP_block[k] < δ
                current_pt.zMat.LP_block[k] = δ
                n_lp_floored += 1
            end
        end
        if n_lp_floored > 0
            @warn "read_ini_s: $(path): initial point is not strictly interior — regularized (n_lp_floored=$(n_lp_floored), lp_floor=$(Float64(δ)))" file=path n_lp_floored=n_lp_floored lp_floor=δ
        end
        ok = compute_inverse!(current_pt, work)
        if !ok
            # X is never regularized (only Z is), so if X itself is not
            # PD the εI loop below cannot help — fail fast naming the
            # failing side instead of shifting Z in vain. This is the
            # y-only-ini-s case: no X entries read → X = 0.
            x_ok = if Threads.nthreads() > 1
                threaded_get_inv_chol!(
                    current_pt.invCholeskyX, current_pt.xMat, work.DLS1,
                )
            else
                get_inv_chol!(
                    current_pt.invCholeskyX, current_pt.xMat, work.DLS1,
                )
            end
            x_ok || error(
                "read_ini_s: initial X is not positive definite (X is " *
                "never regularized — only Z is; an ini-s with no X " *
                "entries yields X = 0, which cannot start a warm start)",
            )
        end
        z_shift_eps = zero(T)
        for ε in (T(1e-8), T(1e-6), T(1e-4), T(1e-2), T(1e-1))
            ok && break
            for sdp_k in 1:current_pt.zMat.SDP_nBlock
                blk = current_pt.zMat.SDP_block[sdp_k].de_ele
                n = size(blk, 1)
                @inbounds for d in 1:n
                    blk[d, d] += ε
                end
            end
            z_shift_eps += ε
            ok = compute_inverse!(current_pt, work)
        end
        if z_shift_eps > 0
            @warn "read_ini_s: $(path): initial point is not strictly interior — regularized (z_shift_eps=$(Float64(z_shift_eps)))" file=path z_shift_eps=z_shift_eps
        end
        ok || error(
            "read_ini_s: initial Z is not positive definite even after " *
            "regularization (LP floor 1e-6, SDP εI up to 1e-1); X was fine",
        )
    end
    return current_pt
end

"""
    _ini_numeric_tokens(s::AbstractString) -> Vector{String}

Extract all numeric tokens (integers and floats, scientific notation
included) from the ini-s text, in file order. Equivalent to the C++
`%*[^0-9+-]%Fe` / `%*[^0-9+-]%d` skip-and-scan loop: every non-numeric
character (commas, whitespace, quotes) is a separator.
"""
function _ini_numeric_tokens(s::AbstractString)
    tok_re = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    return collect(eachmatch(tok_re, s)) .|> m -> m.match
end
