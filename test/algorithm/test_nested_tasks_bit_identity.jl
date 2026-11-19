# test/algorithm/test_nested_tasks_bit_identity.jl
#
# nested-`@tasks` bit-identity harness.
#
# `Rgemm!` opens its own OhMyThreads `@tasks` region whenever
# m*n*k > _RGEMM_SERIAL_THRESHOLD = 4096 (src/mplapack/Rgemm.jl:175), and
# `threaded_compute_bmat_dense!` (ThreadedOps.jl:46; F1 path at :96-97)
# and `threaded_compute_bmat_sparse_sdp!` (SchurComplement.jl ~:1459)
# DO nest `sdpa_mul!` → `Rgemm!` → its own `@tasks` inside their `@tasks`
# bodies for every SDP block with m·n·k > 4096 on the BigFloat path.
#
# This harness checks the nesting empirically with TWO problems, both
# BigFloat(200):
#
#   P1 (nesting): a synthetic ONE-block 20×20 SDP problem (20³ = 8000 >
#   4096 → every F1/F2 Schur product nests Rgemm!'s `@tasks` inside
#   threaded_compute_bmat_dense!'s `@tasks`) with m = 25 constraints
#   (< 100 → decide_dense_or_sparse routes DENSE unconditionally,
#   AggregateSparsity.jl:120-125).
#
#   P2 (multi-block chain): THREE 18×18 SDP blocks, m = 15 constraints,
#   each constraint touching 2 ADJACENT blocks (the 120-block chain of
#   test_sparse_e2e.jl, shrunk to keep per-solve < 10 s; 18³ =
#   5832 > 4096 also nests). Constraints sharing a block contribute to
#   the SAME bmat[i,j] from DIFFERENT block regions — the cross-block
#   accumulation regime. P1 alone could not see the F3 flattening race
#   (two tasks mp_add!-ing one bmat element concurrently →
#   nondeterministic accumulation + pdFEAS stalls, test_block_sparse_f1);
#   P2 closes that coverage hole.
#
# Two subprocesses (JULIA_NUM_THREADS=1 and =4, set on the child Cmd env)
# each solve BOTH problems `reps` times; every solve's FULL trajectory
# (all 9 IterationRow fields per iteration) plus final objective /
# iteration count / terminal phase is compared byte-for-byte.
# `string(::BigFloat)` is shortest-round-trip at the value's own
# precision, hence injective on distinct values — ulp-level divergence
# anywhere shows up.
#
# Verdict: ok=true iff every rep's canonical block is byte-identical
# across thread counts; a worker crash or timeout counts as a race
# outcome too (ok=false).
#
# Comparison is EXACT string equality of per-rep canonical blocks (SHA is
# only an extras/test dep, unavailable at `--project=<repo>` runtime, and
# exact compare also gives free first-divergence localization). The FNV-128
# per-rep hash in each REP line is informational only.
#
# Usage (Julia 1.12, repo root as cwd):
#   <julia> --project=. test/algorithm/test_nested_tasks_bit_identity.jl --campaign <reps>
#       Parent mode: 1-thread vs 4-thread subprocesses, `reps` reps each
#       (one rep = one solve of BOTH problems), prints "BIT-IDENTICAL:
#       yes/no" + timings + projected-50-rep ETA, exit 0/1.
#   <julia> --project=. test/algorithm/test_nested_tasks_bit_identity.jl \
#           --worker <nthreads> <reps> <outfile> [datpath1 datpath2]
#       Worker mode (spawned by the parent; JULIA_NUM_THREADS arrives via
#       the child env, asserted against the <nthreads> arg). Without the
#       dat paths the worker rebuilds both identical .dat-s files from
#       the fixed seeds below.
#   <julia> --project=. test/algorithm/test_nested_tasks_bit_identity.jl
#       (or `include`d from runtests.jl later) → @testset: campaign at
#       reps=2, @test_skip on single-core boxes.
#
# Included from runtests.jl: the testset include runs the campaign at
# reps=2 (see test/runtests.jl).

using SDPA
using Test  # stdlib; only exercised in testset mode

const _PROJECT_ROOT = dirname(dirname(@__DIR__))
const _JULIA_BIN = joinpath(Sys.BINDIR, "julia")
const _HARNESS = abspath(@__FILE__)
const _SOLVE_PRECISION = 200
const _BLOCK_N = 20      # 20³ = 8000 > 4096 → Rgemm! parallel branch (nesting precondition)
const _NCONSTRAINT = 25  # m < 100 → decide_dense_or_sparse always dense
const _MB_NBLOCKS = 3      # P2: SDP blocks in the chain
const _MB_BLOCK_N = 18     # 18³ = 5832 > 4096 → Rgemm! parallel branch (nesting precondition)
const _MB_NCONSTRAINT = 15 # m < 100 → decide_dense_or_sparse always dense
const _TESTSET_REPS = 2
const _MAXITER = 50      # bound worst-case solve time; typical convergence is well under
const _TRAJ_FIELDS = ("iter", "mu", "thetaP", "thetaD", "objP", "objD",
                      "alphaP", "alphaD", "beta")
const _REP_FIELDS = ("P1 fnv128", "P1 objP", "P1 iterations", "P1 phase",
                     "P2 fnv128", "P2 objP", "P2 iterations", "P2 phase")

# ---------------------------------------------------------------------------
# Deterministic xorshift* RNG: the .dat-s must be byte-identical whether the
# parent builds it or a standalone worker rebuilds it, independent of any
# stdlib Random version drift.
# ---------------------------------------------------------------------------
mutable struct _XorStar
    s::UInt64
end

function _next01!(r::_XorStar)
    x = r.s
    x ⊻= x >> 12
    x ⊻= x << 25
    x ⊻= x >> 27
    r.s = x
    u = (x * 0x2545F4914F6CDD1D) >> 11  # 53 bits
    return Float64(u) / 9007199254740992.0  # / 2^53 → [0, 1)
end

"""
    _build_nested_problem_string() -> String

The synthetic nested-`@tasks` problem as .dat-s text (pure function of the
constants above — fixed seed).

- ONE SDP block, 20×20: every F1/F2 Schur product inside
  threaded_compute_bmat_dense!'s `@tasks` is a 20×20 · 20×20 `sdpa_mul!` →
  `Rgemm!` with m·n·k = 8000 > 4096 → the nested `@tasks` region under test.
- 25 constraints: m < 100 keeps `decide_dense_or_sparse` on the dense path.
- Strictly feasible primal (X₀ = I, b_i = tr(A_i·I) = tr(A_i)) and dual
  (y₀ = 0, Z₀ = C ≻ 0 by Gershgorin: diag ≥ 0.96, |off-diag| ≤ 0.04 ⇒
  row sums ≤ 0.96 + 19·0.04·(1/2)… safely < bound) → clean convergence.
"""
function _build_nested_problem_string()
    n, m = _BLOCK_N, _NCONSTRAINT
    rng = _XorStar(UInt64(0x0DD00FEEDC0FFEE5))
    A = [Matrix{Float64}(undef, n, n) for _ in 1:m]
    for i in 1:m
        Ai = A[i]
        for k in 1:n, l in k:n
            v = 2.0 * _next01!(rng) - 1.0  # [-1, 1)
            Ai[k, l] = v
            Ai[l, k] = v
        end
    end
    C = Matrix{Float64}(undef, n, n)
    for k in 1:n, l in k:n
        v = 0.04 * (2.0 * _next01!(rng) - 1.0)  # |off-diag| < 0.04
        v = (k == l) ? 1.0 + v : v              # diag > 0.96
        C[k, l] = v
        C[l, k] = v
    end
    io = IOBuffer()
    println(io, m, " ", 1)  # mDIM nBLOCK
    println(io, n)          # blockStruct: one n×n SDP block
    println(io, join((repr(sum(A[i][k, k] for k in 1:n)) for i in 1:m), " "))  # b
    for k in 1:n, l in k:n  # C entries (upper triangle)
        println(io, "0 1 ", k, " ", l, " ", repr(C[k, l]))
    end
    for i in 1:m, k in 1:n, l in k:n  # A_i entries (upper triangle)
        println(io, i, " 1 ", k, " ", l, " ", repr(A[i][k, l]))
    end
    return String(take!(io))
end

"""
    _build_mbchain_problem_string() -> String

The multi-block chain problem (P2) as .dat-s text (pure function of the
constants above — fixed seed): `_MB_NBLOCKS` SDP blocks of
`_MB_BLOCK_N`×`_MB_BLOCK_N`; each of the m constraints touches exactly 2
ADJACENT blocks (constraints 1..⌈m/(nb-1)⌉ → blocks 1-2, the rest →
blocks 2-3 — the 120-block chain of test_sparse_e2e.jl, shrunk to
keep per-solve < 10 s). Any bmat[i,j] whose two constraints sit on the
SAME block pair accumulates from TWO block regions in serial block order
inside threaded_compute_bmat_dense! — exactly the regime the F3
flattening raced on. m = 15 < 100 keeps the dense path.

Strictly feasible like P1: X₀ = I ⇒ b_i = tr(A_i) summed over both
touched blocks; y₀ = 0, Z₀ = C ≻ 0 per block by Gershgorin (diag > 0.96,
off-diagonal row-sum bound 17·0.04 = 0.68 < 0.96).
"""
function _build_mbchain_problem_string()
    nb, n, m = _MB_NBLOCKS, _MB_BLOCK_N, _MB_NCONSTRAINT
    per_pair = cld(m, nb - 1)
    pair_of(i) = min(cld(i, per_pair), nb - 1)  # constraint i → blocks (p, p+1)
    rng = _XorStar(UInt64(0x5EEDB10C5C0A75))
    # A_i per touched block (symmetric, entries in [-1, 1)); b_i = tr(A_i)
    # over both touched blocks (X₀ = I strict feasibility).
    A = [Dict{Int,Matrix{Float64}}() for _ in 1:m]
    b = Vector{Float64}(undef, m)
    for i in 1:m
        p = pair_of(i)
        b[i] = 0.0
        for bl in (p, p + 1)
            Ai = Matrix{Float64}(undef, n, n)
            for k in 1:n, l in k:n
                v = 2.0 * _next01!(rng) - 1.0  # [-1, 1)
                Ai[k, l] = v
                Ai[l, k] = v
            end
            A[i][bl] = Ai
            b[i] += sum(Ai[k, k] for k in 1:n)
        end
    end
    # C per block (strictly PD, same construction as P1).
    C = [Matrix{Float64}(undef, n, n) for _ in 1:nb]
    for bl in 1:nb, k in 1:n, l in k:n
        v = 0.04 * (2.0 * _next01!(rng) - 1.0)  # |off-diag| < 0.04
        v = (k == l) ? 1.0 + v : v              # diag > 0.96
        C[bl][k, l] = v
        C[bl][l, k] = v
    end
    io = IOBuffer()
    println(io, m, " ", nb)  # mDIM nBLOCK
    println(io, join(fill(n, nb), " "))  # blockStruct
    println(io, join((repr(v) for v in b), " "))
    for bl in 1:nb, k in 1:n, l in k:n  # C entries (upper triangle per block)
        println(io, "0 ", bl, " ", k, " ", l, " ", repr(C[bl][k, l]))
    end
    for i in 1:m, bl in sort(collect(keys(A[i])))  # A_i entries, sorted for determinism
        for k in 1:n, l in k:n
            println(io, i, " ", bl, " ", k, " ", l, " ", repr(A[i][bl][k, l]))
        end
    end
    return String(take!(io))
end

# Two-lane FNV-1a (informational digest only — the authoritative comparison
# is exact string equality of the canonical blocks).
function _fnv128(s::AbstractString)
    h1 = UInt64(0xcbf29ce484222325)
    h2 = UInt64(0x9e3779b97f4a7c15)
    for b in codeunits(s)
        h1 = (h1 ⊻ UInt64(b)) * UInt64(0x100000001b3)
        h2 = (h2 ⊻ UInt64(b)) * UInt64(0x100000001b3)
        h2 = (h2 << 7) | (h2 >> 57)
    end
    return string(h1; base=16, pad=16) * string(h2; base=16, pad=16)
end

_med(v::Vector{Float64}) =
    (s = sort(v); L = length(s); isodd(L) ? s[(L + 1) >> 1] : (s[L >> 1] + s[(L >> 1) + 1]) / 2)

# ---------------------------------------------------------------------------
# Problem registry: P1 (single-block nesting) + P2 (multi-block chain)
# ---------------------------------------------------------------------------

function _validate_p1(problem)
    problem.blockStruct == [_BLOCK_N] || error(
        "worker: P1: expected a single $(_BLOCK_N)×$(_BLOCK_N) SDP block, got $(problem.blockStruct)")
    _BLOCK_N^3 > SDPA._RGEMM_SERIAL_THRESHOLD || error(
        "worker: P1: $(_BLOCK_N)³=$(_BLOCK_N^3) no longer exceeds the Rgemm! serial " *
        "threshold $(SDPA._RGEMM_SERIAL_THRESHOLD) — harness no longer nests")
    use_dense, _, _ = SDPA.analyze_sparsity(problem)
    use_dense || error("worker: P1 unexpectedly routed SPARSE — harness invalid")
end

function _validate_p2(problem)
    problem.blockStruct == fill(_MB_BLOCK_N, _MB_NBLOCKS) || error(
        "worker: P2: expected $(_MB_NBLOCKS) $(_MB_BLOCK_N)×$(_MB_BLOCK_N) SDP blocks, got $(problem.blockStruct)")
    _MB_BLOCK_N^3 > SDPA._RGEMM_SERIAL_THRESHOLD || error(
        "worker: P2: $(_MB_BLOCK_N)³=$(_MB_BLOCK_N^3) no longer exceeds the Rgemm! serial " *
        "threshold $(SDPA._RGEMM_SERIAL_THRESHOLD) — harness no longer nests")
    # Cross-block coverage precondition: constraints must SPAN blocks so
    # that some bmat[i,j] accumulates from ≥ 2 block regions (the F3
    # race regime). Middle blocks see constraints from several pairs.
    per_pair = cld(_MB_NCONSTRAINT, _MB_NBLOCKS - 1)
    pair_of(i) = min(cld(i, per_pair), _MB_NBLOCKS - 1)
    expected = [count(i -> pair_of(i) <= bl <= pair_of(i) + 1, 1:_MB_NCONSTRAINT)
                for bl in 1:_MB_NBLOCKS]
    problem.SDP_nConstraint == expected || error(
        "worker: P2: per-block constraint counts $(problem.SDP_nConstraint) != " *
        "expected $expected — no cross-block spanning")
    use_dense, _, _ = SDPA.analyze_sparsity(problem)
    use_dense || error("worker: P2 unexpectedly routed SPARSE — harness invalid")
end

function _problem_specs(dir::AbstractString)
    p1 = joinpath(dir, "nested$(_BLOCK_N).dat-s")
    p2 = joinpath(dir, "mbchain$(_MB_NBLOCKS)x$(_MB_BLOCK_N).dat-s")
    write(p1, _build_nested_problem_string())
    write(p2, _build_mbchain_problem_string())
    return [
        (label="P1-single", path=p1, validate=_validate_p1),
        (label="P2-mbchain", path=p2, validate=_validate_p2),
    ]
end

# ---------------------------------------------------------------------------
# Worker mode: `--worker <nthreads> <reps> <outfile> [datpath1 datpath2]`
# ---------------------------------------------------------------------------
function _worker_main(nthreads::Int, reps::Int, outfile::AbstractString,
                      datpaths::Union{Nothing,Vector{<:AbstractString}})
    Threads.nthreads() == nthreads || error(
        "worker: Threads.nthreads()=$(Threads.nthreads()) != requested $nthreads " *
        "(JULIA_NUM_THREADS env not applied to this subprocess?)")
    if datpaths === nothing
        specs = _problem_specs(mktempdir())
    else
        length(datpaths) == 2 || error(
            "worker: expected 0 or 2 dat paths, got $(length(datpaths))")
        specs = [
            (label="P1-single", path=String(datpaths[1]), validate=_validate_p1),
            (label="P2-mbchain", path=String(datpaths[2]), validate=_validate_p2),
        ]
    end
    foreach(s -> s.validate(SDPA.read_dat_s(s.path, BigFloat)), specs)

    solve_once(datpath) = SDPA.solve!(
        SDPA.read_dat_s(datpath, BigFloat),
        SDPA.Parameters{BigFloat}(precision=_SOLVE_PRECISION, maxIteration=_MAXITER);
        verbose=false, io=devnull, reverse_primal_dual=false)

    foreach(s -> solve_once(s.path), specs)  # untimed JIT warmup, results discarded

    open(outfile, "w") do io
        print(io, "WORKER nthreads=", nthreads, " reps=", reps)
        for s in specs
            print(io, " dat[", s.label, "]=", _fnv128(read(s.path, String)))
        end
        println(io)
        for rep in 1:reps
            # One rep = one solve of BOTH problems. The REP line carries
            # (fnv128, objP, iterations, phase) per problem, P1 then P2;
            # both problems' TRAJ rows follow under the same rep index.
            rep_fields = String[]
            traj_rows = String[]
            ts = Float64[]
            for s in specs
                t = @elapsed log = solve_once(s.path)
                isempty(log.trajectory) && error(
                    "worker: rep $rep ($(s.label)) produced an empty trajectory")
                row1 = log.trajectory[1]
                collect(string.(fieldnames(typeof(row1)))) == collect(_TRAJ_FIELDS) || error(
                    "worker: IterationRow fields changed: $(fieldnames(typeof(row1)))")
                rows = String["TRAJ " * string(rep) * " " * join(
                    (string(getfield(row, f)) for f in fieldnames(typeof(row))), " ")
                    for row in log.trajectory]
                push!(rep_fields, _fnv128(join(rows, "\n")),
                      string(log.final_obj_primal), string(log.iterations),
                      string(log.terminal_phase))
                append!(traj_rows, rows)
                push!(ts, t)
            end
            println(io, "REP ", rep, " ", join(rep_fields, " "))
            foreach(r -> println(io, r), traj_rows)
            println(io, "TIME ", rep, " ", join(ts, " "))
            flush(io)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Parent mode
# ---------------------------------------------------------------------------
function _parse_worker_output(text::AbstractString, reps::Int)
    blocks = Dict{Int,Vector{String}}()
    times = Dict{Int,NTuple{2,Float64}}()  # rep → (t_P1, t_P2)
    for line in split(text, "\n")
        m = match(r"^(REP|TRAJ) (\d+) (.*)$", line)
        if m !== nothing
            push!(get!(blocks, parse(Int, m.captures[2]), String[]), line)
            continue
        end
        mt = match(r"^TIME (\d+) (.+)$", line)
        if mt !== nothing
            ts = parse.(Float64, split(mt.captures[2]))
            length(ts) == 2 || error(
                "worker output: TIME line carries $(length(ts)) values, expected 2 (P1, P2)")
            times[parse(Int, mt.captures[1])] = (ts[1], ts[2])
        end
    end
    sort(collect(keys(blocks))) == collect(1:reps) ||
        error("worker output: expected reps 1:$reps, got $(sort(collect(keys(blocks))))")
    sort(collect(keys(times))) == collect(1:reps) ||
        error("worker output: expected $reps TIME lines, got $(sort(collect(keys(times))))")
    return blocks, [times[r] for r in 1:reps]
end

function _first_diff(b1::Vector{String}, b4::Vector{String})
    for i in 1:min(length(b1), length(b4))
        b1[i] == b4[i] && continue
        t1, t4 = split(b1[i]), split(b4[i])
        for j in 1:min(length(t1), length(t4))
            t1[j] == t4[j] && continue
            labels = t1[1] == "REP" ? _REP_FIELDS : _TRAJ_FIELDS
            field = j <= 2 ? "rep-index" : (j - 2 <= length(labels) ? labels[j - 2] : "token$j")
            return "line $i ($(t1[1]) #$(t1[2])), field `$field`: t1=\"$(t1[j])\" vs t4=\"$(t4[j])\""
        end
        return "line $i ($(t1[1]) #$(t1[2])): token count differs ($(length(t1)) vs $(length(t4)))"
    end
    return length(b1) == length(b4) ? "unknown" :
           "line count differs ($(length(b1)) vs $(length(b4))) — different iteration counts"
end

"""
    run_campaign(reps; threads_list=(1, 4), timeout_s=nothing)

Spawn one subprocess per thread count (JULIA_NUM_THREADS set on the CHILD
Cmd env — the parent's own thread count is untouched), each solving the
identical problem SET (P1 single-block 20×20 + P2 3-block 18×18 chain)
`reps` times; one rep = one solve of BOTH problems. Returns a NamedTuple
`(ok, report, times, walls)`; `ok=true` iff every rep's canonical block
(REP header + all TRAJ rows of both problems) is byte-identical across
thread counts. Worker crash or timeout ⇒ `ok=false` with the reason (a
hang is a race outcome too).

`timeout_s` defaults to `900 + 240·reps` seconds, a budget calibrated
for subprocess Julia startup + compilation + the two small solves; on
slow machines scale it via `SDPA_TEST_TIMEOUT_SCALE` (multiplier, e.g.
"3" triples the budget) instead of editing the formula.
"""
function run_campaign(reps::Int; threads_list=(1, 4), timeout_s::Union{Nothing,Float64}=nothing)
    reps >= 1 || error("run_campaign: reps must be >= 1")
    if timeout_s === nothing
        scale = something(tryparse(Float64, get(ENV, "SDPA_TEST_TIMEOUT_SCALE", "")), 1.0)
        timeout_s = (900.0 + 240.0 * reps) * max(scale, 0.1)
    end
    dir = mktempdir()
    specs = _problem_specs(dir)
    blocks = Dict{Int,Dict{Int,Vector{String}}}()
    times = Dict{Int,Vector{NTuple{2,Float64}}}()
    walls = Dict{Int,Float64}()
    for nt in threads_list
        outfile = joinpath(dir, "worker_t$nt.out")
        env = copy(ENV)
        env["JULIA_NUM_THREADS"] = string(nt)  # child-only override
        # keep the child off the network: it loads the resolved project only
        env["JULIA_PKG_SERVER"] = ""
        cmd = Cmd(`$(_JULIA_BIN) --project=$(_PROJECT_ROOT) $(_HARNESS) --worker $nt $reps $outfile $(specs[1].path) $(specs[2].path)`;
                  env=env)
        t0 = time()
        proc = run(cmd; wait=false)
        while process_running(proc) && time() - t0 < timeout_s
            sleep(2)
        end
        if process_running(proc)
            kill(proc)
            return (ok=false,
                    report="campaign aborted: nthreads=$nt worker TIMED OUT after " *
                           "$(round(time() - t0; digits=1)) s (a hang is a race outcome)",
                    times=times, walls=walls)
        end
        success(proc) || return (ok=false,
                                 report="campaign aborted: nthreads=$nt worker exited " *
                                        "nonzero ($(proc.exitcode)) — see its stderr above",
                                 times=times, walls=walls)
        blocks[nt], times[nt] = _parse_worker_output(read(outfile, String), reps)
        walls[nt] = time() - t0
        @info "campaign: nthreads=$nt worker finished in $(round(walls[nt]; digits=1)) s"
    end
    nt1, nt4 = threads_list[1], threads_list[2]
    for r in 1:reps
        b1, b4 = blocks[nt1][r], blocks[nt4][r]
        b1 == b4 || return (ok=false,
                            report="BIT-IDENTICAL: no — rep $r diverged: " * _first_diff(b1, b4),
                            times=times, walls=walls)
    end
    s1, s4 = times[nt1], times[nt4]
    overhead = (walls[nt1] - sum(sum.(s1))) + (walls[nt4] - sum(sum.(s4)))  # startup+compile+warmup
    # Each rep now costs BOTH problems per thread count; the 50-rep
    # projection sums the per-problem median solves over both problems
    # AND both thread counts.
    med_rep(nt) = _med([t[1] for t in times[nt]]) + _med([t[2] for t in times[nt]])
    proj50 = 50 * (med_rep(nt1) + med_rep(nt4)) + overhead
    lines = String[]
    for nt in (nt1, nt4)
        s = times[nt]
        p1 = [t[1] for t in s]
        p2 = [t[2] for t in s]
        push!(lines, "worker nthreads=$nt: wall=$(round(walls[nt]; digits=1))s, " *
                     "P1 per-solve min/med/max=$(round(minimum(p1); digits=2))/" *
                     "$(round(_med(p1); digits=2))/$(round(maximum(p1); digits=2))s, " *
                     "P2 per-solve min/med/max=$(round(minimum(p2); digits=2))/" *
                     "$(round(_med(p2); digits=2))/$(round(maximum(p2); digits=2))s (n=$(length(s)) reps)")
    end
    push!(lines, "projected 50+50-rep campaign ≈ 50×[(medP1+medP2)@$nt1 + (medP1+medP2)@$nt4]+overhead = " *
                 "$(round(proj50; digits=1))s ($(round(proj50 / 60; digits=1)) min)")
    push!(lines, "BIT-IDENTICAL: yes — all $reps reps (P1 + P2) byte-identical across " *
                 "nthreads=$nt1 vs nthreads=$nt4 (all 9 IterationRow fields per " *
                 "iteration + final objective/iterations/phase, both problems)")
    return (ok=true, report=join(lines, "\n"), times=times, walls=walls)
end

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
function _testset_entry()
    @testset "nested @tasks bit-identity: 1 vs 4 threads, single-block + chain fixtures" begin
        if Sys.CPU_THREADS > 1
            res = run_campaign(_TESTSET_REPS)
            @info res.report
            @test res.ok
        else
            @test_skip false  # single-core box: the campaign needs >= 2 hardware threads
        end
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    if length(ARGS) >= 4 && ARGS[1] == "--worker"
        _worker_main(parse(Int, ARGS[2]), parse(Int, ARGS[3]), ARGS[4],
                     length(ARGS) >= 5 ? String.(ARGS[5:end]) : nothing)
    elseif length(ARGS) >= 2 && ARGS[1] == "--campaign"
        reps = parse(Int, ARGS[2])
        res = run_campaign(reps)
        println(res.report)
        exit(res.ok ? 0 : 1)
    elseif isempty(ARGS)
        _testset_entry()
    else
        error("usage: [--campaign <reps>] | [--worker <nthreads> <reps> <outfile> [datpath1 datpath2]] | (no args → @testset)")
    end
else
    _testset_entry()  # included from runtests.jl
end
