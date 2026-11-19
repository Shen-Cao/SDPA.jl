using SDPA
using Test

# `useFormula` must be per-(constraint, SDP block), not per-constraint.
#
# Root cause (verified against C++): C++ stores
#   `useFormula = new FormulaType[m * SDP_nBlock]` indexed
#   `useFormula[i * SDP_nBlock + l]` (sdpa_newton.cpp:75, writes at
#   782/785/792/796/800, reads at 1023/1114) — one formula per
#   (constraint i, SDP block l). The dense-block guard at 781-786 selects
#   only F1/F2 for DENSE blocks. Julia stored ONE Symbol per constraint
#   (`useFormula::Vector{Symbol}`), and `compute_formula!` wrote it
#   per-block with the LAST block's selection winning — so a later sparse
#   block's F3 choice leaked into an earlier dense block (either the
#   `_check_no_F3_on_dense!` assert fired, or `cal_f3!` hit a BoundsError
#   on the dense block's empty row_index).
#
# `useFormula[i, l]` — per (constraint i, PROBLEM block l),
# the index in scope at every consumer (compute_bmat_dense!,
# compute_bmat_sparse_sdp!, threaded_compute_bmat_sparse_sdp!,
# threaded_compute_bmat_dense!). Mirrors C++ useFormula[i*SDP_nBlock + l]
# (the C++ l is the SDP-block index; the Julia problem-block index
# coincides with it whenever LP blocks are absent, and the C++ comment is
# purely documentary). The cost model (upNonZeroCount, κ=1.2, f3
# threshold) is UNCHANGED — only the storage layout.

const _FIXTURE_DIR = joinpath(dirname(@__DIR__), "fixtures")
const _F4_FIXTURE = joinpath(_FIXTURE_DIR, "f4_mixed_dense_sparse.dat-s")

# ---------------------------------------------------------------------------
# Local helpers (mirror test_sparse_block_index.jl's idioms).
# ---------------------------------------------------------------------------

function _f4_scale_block!(M::AbstractMatrix{T}, s::T) where {T<:Real}
    @inbounds for i in eachindex(M)
        M[i] *= s
    end
    return M
end

# Test-local construction of a full upper-triangle profile-form sparse
# matrix from a dense `A` + `ordering`, via the LIVE sparse-path pipeline
# (profile_from_skyline! + _fill_sparse_values!).
function _f4_profile_from_dense(A::AbstractMatrix{T}, ordering) where {T<:Real}
    m = size(A, 1)
    sb, di = SDPA.profile_from_skyline!(fill(m, m), T)
    SDPA._fill_sparse_values!(sb, A, ordering, di)
    return sb, di
end

# Build a NewtonState configured for the sparse path (full-profile
# location table, identity ordering) — same idiom as the other
# sparse-path tests.
function _f4_setup_sparse_state(problem::SDPA.SDPAProblem{T}) where {T<:Real}
    ns = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
    SDPA.compute_formula!(ns, problem)
    ns.use_sparse = true
    ns.ordering = collect(1:problem.m)
    ns.reverse_ordering = collect(1:problem.m)
    dummy = zeros(T, problem.m, problem.m)
    sb, di = _f4_profile_from_dense(dummy, ns.ordering)
    ns.sparse_bmat = sb
    ns.diagonalIndex = di
    SDPA.make_aggrigate_index_sdp!(ns, problem, ns.ordering)
    return ns
end

# ---------------------------------------------------------------------------
# Cell 1 (unit): per-(i, l) formula selection.
# Fixture: m=18, blockStruct=[2, 3, 3, 3, 3, 3, 3]. Constraint 1 touches
# the DENSE 2x2 block 1 (25% density -> :dense) AND the sparse 3x3 block 2
# (E11, 11% -> :sparse). The sparse block selects F3 (inz=1, up=1:
# f3 = 3.4 < f1 = 31.8, f2 = 8.4); the dense block must stay F1/F2.
# With a per-constraint slot, block 2's F3 would win for constraint 1,
# tripping `_check_no_F3_on_dense!` at the end of `compute_formula!`.
# ---------------------------------------------------------------------------
@testset "useFormula is per (constraint, block): F1/F2 on dense, F3 on sparse" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            problem = SDPA.read_dat_s(_F4_FIXTURE, T)
            @test problem.m == 18
            @test problem.nBlock == 7
            # Block 1 (2x2) of A_1 must be dense (density 25% > 20% gate).
            @test problem.A[1].SDP_sp_block[1].type === :dense
            # Block 2 (3x3) of A_1 stays sparse (E11, 11% <= 20% gate).
            @test problem.A[1].SDP_sp_block[2].type === :sparse

            ns = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
            SDPA.compute_formula!(ns, problem)

            # Dense block 1: F1/F2 for constraint 1 — the per-(i,l) guard.
            @test ns.useFormula[1, 1] in (:F1, :F2)
            # Sparse block 2: F3 is the cost-model winner for the SAME
            # constraint — legal only because storage is per-(i,l).
            @test ns.useFormula[1, 2] === :F3

            # Invariant: for every (i, l) whose A_i block is DENSE, the
            # formula is F1/F2 (never F3) — the per-(i,l) dense guard.
            for l in 1:problem.nBlock
                problem.blockStruct[l] >= 2 || continue
                for k1 in 1:problem.SDP_nConstraint[l]
                    i = problem.SDP_constraint[l][k1]
                    ib = problem.SDP_blockIndex[l][k1]
                    if problem.A[i].SDP_sp_block[ib].type === :dense
                        @test ns.useFormula[i, l] in (:F1, :F2)
                    end
                end
            end
            # Defense-in-depth assertion must always pass.
            SDPA._check_no_F3_on_dense!(problem, ns)
        end
    end
end

# ---------------------------------------------------------------------------
# Cell 2 (assembly): sparse assembly == dense assembly on the
# mixed fixture at rtol 1e-30 (BigFloat). Exercises the per-(i,l) indexing
# of compute_bmat_dense! AND compute_bmat_sparse_sdp! (via make_bmat! at
# nthreads=1) with F3 formulas in the sparse blocks and F1 in the dense
# block. With a per-constraint slot, compute_formula! would crash with
# the leaked-F3 assert before either assembly runs.
#
# NOTE: the solve!-level sparse path never runs for small fixtures — the
# block-level skyline density gate in solve.jl (METIS ordering) falls
# back to dense — so the sparse Schur assembly is compared directly,
# dense-to-profile vs sparse sp_ele.
# ---------------------------------------------------------------------------
@testset "mixed fixture: sparse assembly matches dense (F3 and F1 blocks)" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            problem = SDPA.read_dat_s(_F4_FIXTURE, T)
            params = SDPA.Parameters{T}()
            current_pt = SDPA.SDPASolution{T}(
                problem.m, problem.blockStruct, params.lambdaStar)
            work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)
            # Distinct per-block X / Z⁻¹ values so a wrong formula/block
            # read is detectable (all blocks start as identical λI).
            for blk in 1:length(current_pt.xMat.SDP_block)
                _f4_scale_block!(current_pt.xMat.SDP_block[blk].de_ele, T(blk))
                _f4_scale_block!(current_pt.invzMat.SDP_block[blk].de_ele, T(blk))
            end

            # Sparse assembly (serial at nthreads=1).
            ns_s = _f4_setup_sparse_state(problem)
            SDPA.make_bmat!(ns_s, problem, current_pt, work)

            # Dense assembly.
            ns_d = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
            SDPA.compute_formula!(ns_d, problem)
            SDPA.make_bmat!(ns_d, problem, current_pt, work)
            sp_ref, _ = _f4_profile_from_dense(ns_d.bmat.de_ele, ns_s.ordering)

            rtol = T === BigFloat ? BigFloat("1e-30") : 1e-12
            @test isapprox(ns_s.sparse_bmat.sp_ele, sp_ref.sp_ele; rtol=rtol)

            # Threaded dense consumer (ThreadedOps.jl) must agree with the
            # serial dense consumer bit-for-bit — exercises the useFormula
            # read in the threaded dense path directly (nthreads=1).
            ns_t = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
            SDPA.compute_formula!(ns_t, problem)
            SDPA.threaded_compute_bmat_dense!(ns_t, problem, current_pt, work)
            @test all(ns_t.bmat.de_ele .== ns_d.bmat.de_ele)
        end
    end
end

# ---------------------------------------------------------------------------
# Cell 3: e2e solves converge and agree; existing behavior
# (example1, control5) unchanged.
# ---------------------------------------------------------------------------
@testset "e2e convergence on mixed fixture + example1/control5 behavior guards" begin
    # Mixed fixture: both force_sparse settings converge to the same
    # optimum (obj = 20). NOTE: for small fixtures the block-level density
    # gate reverts force_sparse=true to the dense path, so the
    # sparse-vs-dense equality is proven by cell 2; this cell guards the
    # full solver (no crash, pdOPT).
    for T in (Float64, BigFloat)
        @testset "mixed fixture solve ($T)" begin
            params = SDPA.Parameters{T}(maxIteration=300)
            p_dense = SDPA.read_dat_s(_F4_FIXTURE, T)
            log_dense = SDPA.solve!(p_dense, params; verbose=false, io=devnull,
                                    force_sparse=false)
            p_sparse = SDPA.read_dat_s(_F4_FIXTURE, T)
            log_sparse = SDPA.solve!(p_sparse, params; verbose=false, io=devnull,
                                     force_sparse=true)

            @test log_dense.terminal_phase == SDPA.pdOPT
            @test log_sparse.terminal_phase == SDPA.pdOPT
            @test isfinite(log_dense.final_obj_primal)
            @test isfinite(log_sparse.final_obj_primal)
            if T <: BigFloat
                @test isapprox(log_sparse.final_obj_primal,
                               log_dense.final_obj_primal;
                               rtol=T("1e-30"))
                @test isapprox(log_sparse.final_obj_dual,
                               log_dense.final_obj_dual;
                               rtol=T("1e-30"))
            else
                @test isapprox(log_sparse.final_obj_primal,
                               log_dense.final_obj_primal; rtol=1e-8)
            end
        end
    end

    # example1 @ BigFloat(200): pdOPT, matches C++ golden at rtol 1e-30
    # (C++ golden: cpp_obj_primal_full = -41.9, 39 iters).
    problem = SDPA.read_dat_s(joinpath(_FIXTURE_DIR, "example1.dat-s"), BigFloat)
    params = SDPA.Parameters{BigFloat}(precision=200)
    log = SDPA.solve!(problem, params; verbose=false, io=devnull)
    @test log.terminal_phase == SDPA.pdOPT
    @test isapprox(log.final_obj_primal, BigFloat("-4.19e+01");
                   rtol=BigFloat("1e-30"))
    @test isapprox(log.final_obj_dual, BigFloat("-4.19e+01");
                   rtol=BigFloat("1e-30"))

    # control5: Float64 sanity (always runs; mirrors test_control5.jl).
    problem = SDPA.read_dat_s(joinpath(_FIXTURE_DIR, "control5.dat-s"), Float64)
    params = SDPA.Parameters{Float64}(maxIteration=100)
    log = SDPA.solve!(problem, params; verbose=false, io=devnull,
                      reverse_primal_dual=true)
    @test abs(Float64(log.final_obj_primal) - 16.883599050955793) < 1e-3

    # control5 @ BigFloat(256): pdOPT + rtol 1e-30 vs the C++ golden
    # (C++ golden: cpp_obj_primal_full). Slow (hours-scale), opt-in —
    # gated like test_control5.jl.
    if get(ENV, "SDPA_RUN_SLOW_TESTS", "") in ("1", "true", "yes")  # grammar: keep in sync with the other slow-gated files
        problem = SDPA.read_dat_s(joinpath(_FIXTURE_DIR, "control5.dat-s"), BigFloat)
        params = SDPA.Parameters{BigFloat}(precision=256, maxIteration=100)
        log = SDPA.solve!(problem, params; verbose=false, io=devnull,
                          reverse_primal_dual=true)
        @test log.terminal_phase == SDPA.pdOPT
        # ------------------------------------------------------------------
        # Golden provenance for the 40-digit literal below.
        #
        # Provenance: copied verbatim from the `cpp_obj_primal_full` field
        # of the archived C++ golden record test/fixtures/golden/control5.json
        # (SDPA-GMP 200-bit run, generated 2026-08-02; that record is no
        # longer shipped — see test/fixtures/golden/README.md). Per its
        # "Precision semantics", the C++ .result summary line prints
        # objValPrimal at only 16 significant digits; the full-precision
        # value is objP = c'x computed from the 40-digit xVec dump at
        # BigFloat(200) — i.e. this literal. It is consistent with the
        # 16-digit summary in test/reference/control5.dat-s.result
        # (objValPrimal = 1.6883599050955793e+01).
        #
        # Why it cannot be re-derived in-repo: test/reference/control5.json
        # (the extract.jl sidecar) quantizes the .result scalars to Float64
        # (16 digits: 16.883599050955794), which is insufficient for the
        # rtol=1e-30 comparison below against a BigFloat(256) solve — the
        # full 40 digits are required, and only the (removed) golden record
        # carried them.
        #
        # ANY change to this literal requires a fresh C++ SDPA-GMP rerun
        # for provenance: regenerate control5.dat-s.result per
        # test/reference/README.md (sdpa_gmp, param.sdpa 200-bit preset),
        # recompute c'x from the 40-digit xVec at BigFloat(200), and
        # re-record per test/fixtures/golden/README.md.
        # ------------------------------------------------------------------
        golden = BigFloat("16.883599050955793324280033368461547775773")
        @test isapprox(log.final_obj_primal, golden; rtol=BigFloat("1e-30"))
        @test isapprox(log.final_obj_dual, golden; rtol=BigFloat("1e-30"))
    else
        @test_skip false  # Set SDPA_RUN_SLOW_TESTS=1 to run (slow, hours-scale)
    end
end
