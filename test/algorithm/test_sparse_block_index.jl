# SDP-block index invariant in sparse Schur assembly (regression matrix).
#
# Root cause (verified): "l == SDP-block position" is violated in TWO
# sparse paths with OPPOSITE failure modes:
#   * serial:   compute_bmat_sparse_sdp! advances a separate `sdp_l`
#               counter only when the location-table entry count is
#               nonzero — an SDP block with ZERO constraints desyncs
#               it, so every later block reads the WRONG
#               xMat/invzMat block (silent wrong Schur contributions).
#   * threaded: threaded_compute_bmat_sparse_sdp! uses the
#               problem-block index `l` DIRECTLY as the SDP-block
#               index — wrong/OOB whenever LP blocks exist. The
#               make_bmat! guard `has_lp` only tested "LP constraint
#               count > 0", so an LP block with ZERO LP constraints
#               slipped the mixed problem into the threaded path.
#
# Construction note: `initialize_index!` builds the constraint tables
# from the actual nonzeros, so a block with no nonzero entries is
# naturally excluded (see test_empty_block_lists.jl). To exercise a
# 0-constraint block in a fixture whose blocks ARE touched, we
# hand-mutate the constraint tables AFTER read_dat_s (which populates
# them) and BEFORE assembly/solve.
#
# Cell map (dense == sparse comparisons use identity ordering and a full
# profile, so _profile_from_dense(bmat) is the exact dense reference):
#   1. serial sparse + 0-constraint SDP block (hand-mutated) == dense
#   2. mixed problem, LP block with ZERO LP constraints, nthreads=4
#      subprocess: routes serial, obj1 === obj4
#   3. mixed LP+SDP (real LP constraints): sparse == dense
#   4. SDP-only normal (example1): sparse == dense
#   5. threaded sparse (SDP-only + 0-constraint block) == serial sparse
#   6. SDP-only + 0-constraint block, nthreads=4 subprocess: obj1===obj4
#      (exercises the threaded path)

using SDPA
using Test

include("_sparse_block_index_helpers.jl")  # _rebuild_true_tables!

const _PROJECT_ROOT = dirname(dirname(@__DIR__))
const _JULIA_BIN = joinpath(Sys.BINDIR, "julia")
const _FIXTURE_DIR = joinpath(_PROJECT_ROOT, "test", "fixtures")
const _HELPERS_PATH = joinpath(@__DIR__, "_sparse_block_index_helpers.jl")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function _scale_block!(M::AbstractMatrix{T}, s::T) where {T<:Real}
    @inbounds for i in eachindex(M)
        M[i] *= s
    end
    return M
end

# Build a NewtonState configured for the sparse path (full-profile
# location table, identity ordering) — same idiom as the other
# sparse-path tests.
function _setup_sparse_state(problem::SDPA.SDPAProblem{T}) where {T<:Real}
    ns = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
    SDPA.compute_formula!(ns, problem)
    ns.use_sparse = true
    ns.ordering = collect(1:problem.m)
    ns.reverse_ordering = collect(1:problem.m)
    dummy = zeros(T, problem.m, problem.m)
    sb, di = _profile_from_dense(dummy, ns.ordering)
    ns.sparse_bmat = sb
    ns.diagonalIndex = di
    SDPA.make_aggrigate_index_sdp!(ns, problem, ns.ordering)
    return ns
end

# Run sparse assembly (via make_bmat!, use_sparse=true) and dense assembly
# (make_bmat!, use_sparse=false) on the same current_pt, and return the
# profile-form dense reference plus the sparse sp_ele.
function _sparse_vs_dense(T, problem, current_pt, work)
    ns_s = _setup_sparse_state(problem)
    SDPA.make_bmat!(ns_s, problem, current_pt, work)

    ns_d = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
    SDPA.compute_formula!(ns_d, problem)
    SDPA.make_bmat!(ns_d, problem, current_pt, work)  # use_sparse = false
    sp_ref, _ = _profile_from_dense(ns_d.bmat.de_ele, ns_s.ordering)
    return ns_s.sparse_bmat.sp_ele, sp_ref.sp_ele
end

# Run an assembly script in a subprocess at the given thread count and
# return its stdout (trimmed). `script` is interpolated into a `-e`
# program that must `using SDPA` itself (helpers included via
# _HELPERS_PATH). If the subprocess crashes (e.g. BoundsError in the
# threaded cells), `read` throws ProcessFailedException and the testset
# goes red.
function _assembly_in_subprocess(threads::Int, script::String)
    env = copy(ENV)
    env["JULIA_NUM_THREADS"] = string(threads)
    env["JULIA_PKG_SERVER"] = ""  # offline: load the resolved project only
    cmd = Cmd(`$(_JULIA_BIN) --project=$(_PROJECT_ROOT) -e $script`; env=env)
    return strip(read(cmd, String))
end

# ---------------------------------------------------------------------------
# Cell 1: serial sparse + SDP block with 0 constraints.
# ---------------------------------------------------------------------------
@testset "serial sparse assembly matches dense with a 0-constraint SDP block" begin
    for T in (BigFloat, Float64)
        problem = SDPA.read_dat_s(
            joinpath(_FIXTURE_DIR, "block_sparse.dat-s"), T)
        # Hand-mutate: block 2 has zero constraints (post-initialize_index!).
        @test problem.SDP_nConstraint[2] > 0
        problem.SDP_nConstraint[2] = 0
        # Match the real empty-block state (count 0 AND empty lists), not
        # just the zeroed count — consumers iterate the count today,
        # so this is behavior-neutral but keeps the fixture honest.
        empty!(problem.SDP_constraint[2])
        empty!(problem.SDP_blockIndex[2])

        params = SDPA.Parameters{T}()
        current_pt = SDPA.SDPASolution{T}(
            problem.m, problem.blockStruct, params.lambdaStar)
        work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)
        # Distinct per-block X / Z⁻¹ values so a wrong-block read (the
        # serial-path desync) is detectable — all blocks start as
        # identical λI.
        for blk in 1:length(current_pt.xMat.SDP_block)
            _scale_block!(current_pt.xMat.SDP_block[blk].de_ele, T(blk))
            _scale_block!(current_pt.invzMat.SDP_block[blk].de_ele, T(blk))
        end

        sp_ele, sp_ref = _sparse_vs_dense(T, problem, current_pt, work)
        rtol = T === BigFloat ? BigFloat("1e-30") : 1e-12
        @test isapprox(sp_ele, sp_ref; rtol=rtol)
    end
end

# ---------------------------------------------------------------------------
# Cell 2: threaded guard — LP block with ZERO LP constraints.
#
# Assembly-level comparison in a subprocess at JULIA_NUM_THREADS=4
# (contract: sparse assembly == serial sparse assembly bit-identical, no
# BoundsError). The old make_bmat! has_lp guard (LP constraint count > 0)
# missed the zero-LP-constraint LP block and routed the MIXED problem to
# threaded_compute_bmat_sparse_sdp!, which uses the problem-block index
# as the SDP-block index → BoundsError (the LP block precedes every SDP
# block here). The exact guard (LP blocks exist at all) routes serial at
# ANY thread count.
#
# NOTE: a solve-level variant would be vacuous — the block-level density
# gate in solve.jl (METIS ordering) falls back to dense for every small
# fixture, so a solve never reaches the sparse assembly. The
# assembly-level comparison exercises the dispatcher directly.
# ---------------------------------------------------------------------------
@testset "zero-LP-constraint LP block: assembly identical at 1 and 4 threads" begin
    fixture = joinpath(_FIXTURE_DIR, "lp_first_mixed.dat-s")
    problem = SDPA.read_dat_s(fixture, BigFloat)
    # Sanity: the fixture really is mixed with the LP block FIRST.
    @test any(bs -> bs < 2, problem.blockStruct)
    @test problem.blockStruct[1] < 2
    @test length(problem.SDP_nConstraint) == problem.nBlock
    # Rebuild the true tables and zero the LP constraints.
    _rebuild_true_tables!(problem)
    @test problem.SDP_nConstraint == [0, 3, 3, 3, 3, 3, 3]
    @test problem.LP_nConstraint == [1]

    assembly = """
    using SDPA
    include($(repr(_HELPERS_PATH)))
    problem = SDPA.read_dat_s($(repr(fixture)), BigFloat)
    _rebuild_true_tables!(problem)
    problem.LP_nConstraint[1] = 0  # LP block present, ZERO LP constraints
    T = BigFloat
    params = SDPA.Parameters{T}()
    current_pt = SDPA.SDPASolution{T}(problem.m, problem.blockStruct, params.lambdaStar)
    work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)
    # Sparse assembly via make_bmat! (the dispatcher under test).
    ns_s = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
    SDPA.compute_formula!(ns_s, problem)
    ns_s.use_sparse = true
    ns_s.ordering = collect(1:problem.m)
    ns_s.reverse_ordering = collect(1:problem.m)
    dummy = zeros(T, problem.m, problem.m)
    sb, di = _profile_from_dense(dummy, ns_s.ordering)
    ns_s.sparse_bmat = sb
    ns_s.diagonalIndex = di
    SDPA.make_aggrigate_index_sdp!(ns_s, problem, ns_s.ordering)
    SDPA.make_bmat!(ns_s, problem, current_pt, work)
    # Dense reference.
    ns_d = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
    SDPA.compute_formula!(ns_d, problem)
    SDPA.make_bmat!(ns_d, problem, current_pt, work)
    sp_ref, _ = _profile_from_dense(ns_d.bmat.de_ele, ns_s.ordering)
    println(all(isapprox.(ns_s.sparse_bmat.sp_ele, sp_ref.sp_ele; rtol=BigFloat("1e-30"))))
    """
    @test _assembly_in_subprocess(1, assembly) == "true"
    @test _assembly_in_subprocess(4, assembly) == "true"
end

# ---------------------------------------------------------------------------
# Cell 3: mixed LP+SDP with REAL LP constraints — sparse == dense.
# ---------------------------------------------------------------------------
@testset "mixed LP+SDP sparse assembly matches dense (real LP constraints)" begin
    for T in (BigFloat, Float64)
        problem = SDPA.read_dat_s(
            joinpath(_FIXTURE_DIR, "minimal_sdp_lp.dat-s"), T)
        @test any(bs -> bs < 2, problem.blockStruct)  # mixed
        @test any(>(0), problem.LP_nConstraint)       # real LP constraints

        params = SDPA.Parameters{T}()
        current_pt = SDPA.SDPASolution{T}(
            problem.m, problem.blockStruct, params.lambdaStar)
        work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)

        sp_ele, sp_ref = _sparse_vs_dense(T, problem, current_pt, work)
        rtol = T === BigFloat ? BigFloat("1e-30") : 1e-12
        @test isapprox(sp_ele, sp_ref; rtol=rtol)
    end
end

# ---------------------------------------------------------------------------
# Cell 4: SDP-only normal problem — sparse == dense.
# ---------------------------------------------------------------------------
@testset "SDP-only example1 sparse assembly matches dense" begin
    for T in (BigFloat, Float64)
        problem = SDPA.read_dat_s(
            joinpath(_FIXTURE_DIR, "example1.dat-s"), T)
        @test !any(bs -> bs < 2, problem.blockStruct)  # SDP-only

        params = SDPA.Parameters{T}()
        current_pt = SDPA.SDPASolution{T}(
            problem.m, problem.blockStruct, params.lambdaStar)
        work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)

        sp_ele, sp_ref = _sparse_vs_dense(T, problem, current_pt, work)
        rtol = T === BigFloat ? BigFloat("1e-30") : 1e-12
        @test isapprox(sp_ele, sp_ref; rtol=rtol)
    end
end

# ---------------------------------------------------------------------------
# Cell 5: threaded sparse == serial sparse (SDP-only + 0-constraint SDP
# block). Exercises the SDP-position mapping inside
# threaded_compute_bmat_sparse_sdp!.
# ---------------------------------------------------------------------------
@testset "threaded sparse assembly bit-identical to serial with a 0-constraint block" begin
    for T in (BigFloat, Float64)
        problem = SDPA.read_dat_s(
            joinpath(_FIXTURE_DIR, "sdp6_blocks.dat-s"), T)
        problem.SDP_nConstraint[3] = 0  # block 3: zero constraints
        # Match the real empty-block state (count 0 AND empty lists).
        empty!(problem.SDP_constraint[3])
        empty!(problem.SDP_blockIndex[3])

        params = SDPA.Parameters{T}()
        current_pt = SDPA.SDPASolution{T}(
            problem.m, problem.blockStruct, params.lambdaStar)
        work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)
        for blk in 1:length(current_pt.xMat.SDP_block)
            _scale_block!(current_pt.xMat.SDP_block[blk].de_ele, T(blk))
            _scale_block!(current_pt.invzMat.SDP_block[blk].de_ele, T(blk))
        end

        ns_ser = _setup_sparse_state(problem)
        ns_thr = _setup_sparse_state(problem)
        SDPA.compute_bmat_sparse_sdp!(ns_ser, problem, current_pt, work)
        SDPA.threaded_compute_bmat_sparse_sdp!(ns_thr, problem, current_pt, work)
        # Same values, same accumulation order → bit-identical.
        @test all(ns_ser.sparse_bmat.sp_ele .== ns_thr.sparse_bmat.sp_ele)
    end
end

# ---------------------------------------------------------------------------
# Cell 6: threaded path at real nthreads=4 (subprocess).
# SDP-only + 0-constraint SDP block → threaded_compute_bmat_sparse_sdp!
# at nthreads=4 must equal the serial assembly bit-for-bit.
# ---------------------------------------------------------------------------
@testset "SDP-only 0-constraint block: assembly bit-identical at 1 and 4 threads" begin
    fixture = joinpath(_FIXTURE_DIR, "sdp6_blocks.dat-s")
    problem = SDPA.read_dat_s(fixture, BigFloat)
    _rebuild_true_tables!(problem)
    @test problem.SDP_nConstraint == [3, 3, 3, 3, 3, 3]
    @test !any(bs -> bs < 2, problem.blockStruct)  # SDP-only → threaded at nthreads=4

    assembly = """
    using SDPA
    include($(repr(_HELPERS_PATH)))
    problem = SDPA.read_dat_s($(repr(fixture)), BigFloat)
    _rebuild_true_tables!(problem)
    problem.SDP_nConstraint[3] = 0  # block 3: zero constraints
    T = BigFloat
    params = SDPA.Parameters{T}()
    current_pt = SDPA.SDPASolution{T}(problem.m, problem.blockStruct, params.lambdaStar)
    work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)
    ns_ser = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
    ns_thr = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
    for ns in (ns_ser, ns_thr)
        SDPA.compute_formula!(ns, problem)
        ns.use_sparse = true
        ns.ordering = collect(1:problem.m)
        ns.reverse_ordering = collect(1:problem.m)
        dummy = zeros(T, problem.m, problem.m)
        sb, di = _profile_from_dense(dummy, ns.ordering)
        ns.sparse_bmat = sb
        ns.diagonalIndex = di
        SDPA.make_aggrigate_index_sdp!(ns, problem, ns.ordering)
    end
    SDPA.compute_bmat_sparse_sdp!(ns_ser, problem, current_pt, work)
    SDPA.threaded_compute_bmat_sparse_sdp!(ns_thr, problem, current_pt, work)
    println(all(ns_ser.sparse_bmat.sp_ele .== ns_thr.sparse_bmat.sp_ele))
    """
    @test _assembly_in_subprocess(1, assembly) == "true"
    @test _assembly_in_subprocess(4, assembly) == "true"
end

# ---------------------------------------------------------------------------
# Threaded sparse zeroing is per-element mp_zero! — direct aliasing
# assertion on threaded_compute_bmat_sparse_sdp! (BigFloat).
#
# Root cause: the threaded path zeroed sp_ele with
# `fill!(sparse_bmat.sp_ele, zero(T))`, which stores the SAME BigFloat
# object in every slot. The sequential reduction then mutates
# sp_ele[loc] IN PLACE via mp_add!, so every slot accumulated every
# contribution into one shared object — all entries read the same
# corrupted total. The serial path always used per-element mp_zero!;
# this cell locks the parity in directly:
#   (a) values bit-identical to the serial assembly (correct per-location
#       values, same accumulation order);
#   (b) spot-check: two locations that received different accumulations
#       hold different values (under the bug every entry read the same
#       total);
#   (c) zeroing-is-per-element: rebuild sp_ele as _independent_zeros
#       FIRST and re-run — the fill! bug would alias these fresh slots
#       (every entry ends up equal to the accumulated total); the
#       mp_zero! loop preserves independence.
# Runs in-process (nthreads=1): the zeroing + sequential reduction are
# thread-count-independent, so this catches the bug at any thread count.
# ---------------------------------------------------------------------------
@testset "threaded sparse zeroing is per-element (no BigFloat aliasing)" begin
    T = BigFloat
    problem = SDPA.read_dat_s(joinpath(_FIXTURE_DIR, "sdp6_blocks.dat-s"), T)

    params = SDPA.Parameters{T}()
    current_pt = SDPA.SDPASolution{T}(
        problem.m, problem.blockStruct, params.lambdaStar)
    work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)
    for blk in 1:length(current_pt.xMat.SDP_block)
        _scale_block!(current_pt.xMat.SDP_block[blk].de_ele, T(blk))
        _scale_block!(current_pt.invzMat.SDP_block[blk].de_ele, T(blk))
    end

    ns_ser = _setup_sparse_state(problem)
    ns_thr = _setup_sparse_state(problem)
    SDPA.compute_bmat_sparse_sdp!(ns_ser, problem, current_pt, work)
    SDPA.threaded_compute_bmat_sparse_sdp!(ns_thr, problem, current_pt, work)

    sp_ser = ns_ser.sparse_bmat.sp_ele
    sp_thr = ns_thr.sparse_bmat.sp_ele
    @test length(sp_ser) == length(sp_thr) == length(ns_thr.sparse_bmat.sp_ele)

    # (a) correctness + independence-in-value: bit-identical to serial.
    @test all(sp_ser .== sp_thr)
    # (b) spot-check: two locations that received different accumulations
    #     hold different values (under the bug every entry read the same
    #     total).
    #     Locate the pair by the SERIAL values (some locations legitimately
    #     coincide, e.g. block scales squared), then verify the threaded
    #     result reproduces both values and keeps them distinct.
    d1 = findfirst(!iszero, sp_ser)
    @test d1 !== nothing
    d2 = findfirst(k -> !iszero(sp_ser[k]) && sp_ser[k] != sp_ser[d1],
                   eachindex(sp_ser))
    @test d2 !== nothing
    @test sp_thr[d1] == sp_ser[d1]
    @test sp_thr[d2] == sp_ser[d2]
    @test sp_thr[d1] != sp_thr[d2]

    # (c) zeroing-is-per-element: independent-zeros FIRST, then the call.
    ns_thr2 = _setup_sparse_state(problem)
    ns_thr2.sparse_bmat.sp_ele = SDPA._independent_zeros(
        T, length(ns_thr2.sparse_bmat.sp_ele))
    SDPA.threaded_compute_bmat_sparse_sdp!(ns_thr2, problem, current_pt, work)
    @test all(ns_thr2.sparse_bmat.sp_ele .== sp_ser)
end

# ---------------------------------------------------------------------------
# Trigger config — sparse path + SDP-only + BigFloat + nthreads > 1 —
# threaded sparse assembly bit-identical to serial, in a real
# JULIA_NUM_THREADS=4 subprocess.
#
# Assembly-level (not solve-level) for the same reason as cell 2: the
# block-level density gate in solve.jl rejects every existing fixture
# (block_density > 0.25), so a full solve never reaches the sparse path.
# The subprocess drives the make_bmat! dispatcher, which routes
# SDP-only + use_sparse + nthreads>1 to threaded_compute_bmat_sparse_sdp!.
# The fill! aliasing corrupted sp_ele at ANY thread count the threaded
# path ran at, so the nthreads=4 leg would print false under the bug.
# ---------------------------------------------------------------------------
@testset "SDP-only BigFloat: threaded sparse assembly equals serial at nthreads=4" begin
    fixture = joinpath(_FIXTURE_DIR, "sdp6_blocks.dat-s")
    problem = SDPA.read_dat_s(fixture, BigFloat)
    @test !any(bs -> bs < 2, problem.blockStruct)  # SDP-only -> threaded at nthreads>1

    assembly = """
    using SDPA
    include($(repr(_HELPERS_PATH)))
    problem = SDPA.read_dat_s($(repr(fixture)), BigFloat)
    @assert !any(bs -> bs < 2, problem.blockStruct)
    T = BigFloat
    params = SDPA.Parameters{T}()
    current_pt = SDPA.SDPASolution{T}(problem.m, problem.blockStruct, params.lambdaStar)
    work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)
    ns_ser = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
    ns_thr = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
    for ns in (ns_ser, ns_thr)
        SDPA.compute_formula!(ns, problem)
        ns.use_sparse = true
        ns.ordering = collect(1:problem.m)
        ns.reverse_ordering = collect(1:problem.m)
        dummy = zeros(T, problem.m, problem.m)
        sb, di = _profile_from_dense(dummy, ns.ordering)
        ns.sparse_bmat = sb
        ns.diagonalIndex = di
        SDPA.make_aggrigate_index_sdp!(ns, problem, ns.ordering)
    end
    SDPA.compute_bmat_sparse_sdp!(ns_ser, problem, current_pt, work)
    SDPA.make_bmat!(ns_thr, problem, current_pt, work)  # threaded at nthreads>1
    println(all(ns_ser.sparse_bmat.sp_ele .== ns_thr.sparse_bmat.sp_ele))
    """
    @test _assembly_in_subprocess(1, assembly) == "true"
    @test _assembly_in_subprocess(4, assembly) == "true"
end
