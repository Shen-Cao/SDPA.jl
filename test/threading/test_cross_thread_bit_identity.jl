using SDPA
using Test

# Cross-thread-count bit-identity test.
#
# The other bit-identity tests run within a single process and do NOT
# compare results across thread counts. This test spawns subprocesses
# at JULIA_NUM_THREADS=1 and JULIA_NUM_THREADS=4, solves the same problem
# in each, and compares results byte-identically.
#
# It guards against the entire bug class of
# "works at nthreads=1 but diverges at nthreads>1"
# (e.g. thread-count-dependent kernel dispatch in the iteration loop).

const _PROJECT_ROOT = dirname(dirname(@__DIR__))
const _JULIA_BIN = joinpath(Sys.BINDIR, "julia")
const _FIXTURE = joinpath(_PROJECT_ROOT, "test", "fixtures", "example1.dat-s")
const _FIXTURE_ARCH0 = joinpath(_PROJECT_ROOT, "test", "fixtures", "arch0.dat-s")

function _solve_in_subprocess(threads::Int, force_sparse::Bool)
    fs_str = force_sparse ? "true" : "false"
    script = """
    using SDPA
    problem = SDPA.read_dat_s("$(_FIXTURE)", Float64)
    params = SDPA.Parameters{Float64}()
    log = SDPA.solve!(problem, params; verbose=false, io=devnull,
                      reverse_primal_dual=true, force_sparse=$(fs_str))
    println(log.final_obj_primal)
    println(log.terminal_phase)
    """
    env = copy(ENV)
    env["JULIA_NUM_THREADS"] = string(threads)
    env["JULIA_PKG_SERVER"] = ""  # offline: load the resolved project only
    cmd = Cmd(`$(_JULIA_BIN) --project=$(_PROJECT_ROOT) -e $script`; env=env)
    out = read(cmd, String)
    lines = split(strip(out), "\n")
    return parse(Float64, lines[1]), lines[2]
end

@testset "Cross-thread-count bit-identity: dense path" begin
    obj1, phase1 = _solve_in_subprocess(1, false)
    obj4, phase4 = _solve_in_subprocess(4, false)
    @test phase1 == phase4 == "pdOPT"
    @test obj1 === obj4  # bit-identical
end

@testset "Cross-thread-count bit-identity: sparse path" begin
    # This test specifically guards against the bug class where the
    # sparse path diverges at nthreads > 1.
    obj1, phase1 = _solve_in_subprocess(1, true)
    obj4, phase4 = _solve_in_subprocess(4, true)
    @test phase1 == phase4 == "pdOPT"
    # Sparse path may have tiny float differences vs dense due to F1/F2/F3
    # operation order, but 1-thread vs 4-thread sparse results should be
    # bit-identical (the sparse assembly is sequential regardless of thread
    # count; only the dense path's threaded_compute_bmat_dense! differs).
    @test obj1 === obj4  # bit-identical
end

@testset "Cross-thread-count bit-identity: BigFloat above Rgemm parallel threshold" begin
    # The BigFloat Rgemm! kernels take a parallel @tasks branch when
    # m*n*k > 4096; the task bodies must not share a function-captured
    # BigFloat accumulator — a shared one races at JULIA_NUM_THREADS >= 2
    # and corrupts results non-deterministically.
    #
    # arch0 has a 161x161 SDP block (161³ ≫ 4096), so every iteration
    # crosses the parallel threshold on the BigFloat path (our kernels,
    # not BLAS — BLAS is pinned to 1 thread for BigFloat in solve!).
    # Float64 would NOT work here: it goes through LAPACK/BLAS and its
    # internal reduction order changes with BLAS thread count.
    function _solve_bf_in_subprocess(threads::Int)
        script = """
        using SDPA
        problem = SDPA.read_dat_s("$(_FIXTURE_ARCH0)", BigFloat)
        params = SDPA.Parameters{BigFloat}(precision=200)
        log = SDPA.solve!(problem, params; verbose=false)
        println(log.final_obj_primal)
        println(log.terminal_phase)
        """
        env = copy(ENV)
        env["JULIA_NUM_THREADS"] = string(threads)
        env["JULIA_PKG_SERVER"] = ""  # offline: load the resolved project only
        cmd = Cmd(`$(_JULIA_BIN) --project=$(_PROJECT_ROOT) -e $script`; env=env)
        out = read(cmd, String)
        lines = split(strip(out), "\n")
        # Compare the FULL-precision decimal strings (=== on parsed
        # BigFloats would round both to the same parse precision and
        # mask ulp-level divergence).
        return lines[1], lines[2]
    end
    obj1, phase1 = _solve_bf_in_subprocess(1)
    obj4, phase4 = _solve_bf_in_subprocess(4)
    @test phase1 == phase4 == "pdOPT"
    @test obj1 === obj4  # bit-identical full-precision decimal strings
end

# ---------------------------------------------------------------------------
# Zero-LP-constraint guard: mixed LP+SDP problem with an LP block that
# has ZERO LP constraints, sparse path, BigFloat, JULIA_NUM_THREADS=4.
#
# make_bmat!'s guard is "LP blocks exist at all" (not "any LP constraint
# count > 0"): a mixed problem with a zero-LP-constraint LP block must
# route to the serial sparse path at ANY thread count → bit-identical.
#
# The true tables are (re)built from A's actual nonzeros before the
# mutation (see _sparse_block_index_helpers.jl) — defensively, in case
# the reader inflated any constraint list to m. NOTE: for small fixtures
# the block-level density gate in solve.jl can fall back to dense, so
# the sharp assembly-level regression lives in
# test/algorithm/test_sparse_block_index.jl (cell 2).
# ---------------------------------------------------------------------------
@testset "Cross-thread-count bit-identity: zero-LP-constraint mixed sparse" begin
    _FIXTURE_MIXED = joinpath(_PROJECT_ROOT, "test", "fixtures", "lp_first_mixed.dat-s")
    _HELPERS_PATH = joinpath(_PROJECT_ROOT, "test", "algorithm", "_sparse_block_index_helpers.jl")
    @assert isfile(_FIXTURE_MIXED)

    function _solve_mixed_in_subprocess(threads::Int)
        script = """
        using SDPA
        include($(repr(_HELPERS_PATH)))
        problem = SDPA.read_dat_s("$(_FIXTURE_MIXED)", BigFloat)
        _rebuild_true_tables!(problem)
        problem.LP_nConstraint[1] = 0  # LP block present, ZERO LP constraints
        params = SDPA.Parameters{BigFloat}(precision=200)
        log = SDPA.solve!(problem, params; verbose=false, io=devnull,
                          force_sparse=true)
        println(log.final_obj_primal)
        println(log.terminal_phase)
        """
        env = copy(ENV)
        env["JULIA_NUM_THREADS"] = string(threads)
        env["JULIA_PKG_SERVER"] = ""  # offline: load the resolved project only
        cmd = Cmd(`$(_JULIA_BIN) --project=$(_PROJECT_ROOT) -e $script`; env=env)
        out = read(cmd, String)
        lines = split(strip(out), "\n")
        return lines[1], lines[2]
    end
    obj1, phase1 = _solve_mixed_in_subprocess(1)
    obj4, phase4 = _solve_mixed_in_subprocess(4)
    @test phase1 == phase4
    @test obj1 === obj4  # bit-identical full-precision decimal strings
end
