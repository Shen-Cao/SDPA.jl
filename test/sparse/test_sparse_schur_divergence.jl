# test/sparse/test_sparse_schur_divergence.jl
#
# Regression test for the sparse-schur-solve divergence.
#
# Motivation: `build_skyline_from_problem!` (src/sparse/
# AggregateSparsity.jl) builds the skyline profile from the ORIGINAL
# constraint connectivity but does not compute the CHORDAL / fill-in
# closure. C++ SDPA-GMP derives its profile from the SYMBOLIC FACTORIZATION
# (`symbfacIVL`, sdpa_newton.cpp:225-340), which IS fill-closed; a Julia
# profile without that closure lacks the Cholesky factor's fill-in
# positions, so the profile-form Cholesky cannot be exact no matter how
# the factorization is implemented.
#
# This test locks the fix with a SELF-CONTAINED tiny reproduction: a 4×4
# problem whose aggregate constraint graph is a 4-CYCLE (blocks {1,2},
# {2,3}, {3,4}, {4,1}) — non-chordal, so its Cholesky factor has genuine
# fill-in at (4,2) that the under-closed skyline misses.  The test runs the
# PRODUCTION profile pipeline (`build_skyline_from_problem!` →
# `profile_from_skyline!` → `sparse_cholesky!` → `solve_systems_sparse!`)
# against the dense path (`Rpotrf_with_adjust!` + `ldiv!`, the exact dense
# solve used by the solver) on the same SPD test matrix, asserting:
#
#     1. the produced skyline is fill-closed  (the direct fix property), and
#     2. ||Δy_sparse − Δy_dense||₂ / ||Δy_dense||₂  < 1e-20.
#
# The defect is thread-count-independent (it reproduces at
# JULIA_NUM_THREADS=1).

using SDPA
using Test
using LinearAlgebra

const _FIXTURE = joinpath(@__DIR__, "..", "fixtures", "cycle4_nonchordal.dat-s")

@testset "sparse-schur divergence: fill-in closure of the profile skyline" begin
    setprecision(BigFloat, 256) do
        problem = SDPA.read_dat_s(_FIXTURE, BigFloat)
        m = problem.m
        @assert m == 4

        # Identity ordering keeps the aggregate 4-cycle exactly a 4-cycle
        # (a permuted AMD/METIS order could turn it chordal; the defect is
        # ordering-independent, so a fixed order is the deterministic choice).
        ordering = collect(1:m)
        reverse_ordering = Vector{Int}(undef, m)
        for (pos, orig) in enumerate(ordering)
            reverse_ordering[orig] = pos
        end

        # ---- production profile pipeline (the code path under test) ----
        skyline = SDPA.build_skyline_from_problem!(
            problem, ordering, reverse_ordering)
        sparse_mat, diagonalIndex = SDPA.profile_from_skyline!(skyline, BigFloat)
        nnz = diagonalIndex[m + 1] - 1

        # ---- the fill-closure property (direct regression for the fix) ----
        # A skyline is fill-closed iff for every i < j with skyline[i] ≥ j
        # (column i's envelope reaches row j, so i emits an update to
        # column j), column j's envelope reaches at least skyline[i].
        closed = true
        for j in 1:m
            for i in 1:(j - 1)
                if skyline[i] >= j && skyline[j] < skyline[i]
                    closed = false
                end
            end
        end
        @test closed

        # ---- SPD test matrix with the profile's sparsity pattern ----
        # (deterministic, diagonally dominant, so the dense Rpotrf succeeds
        # and the fill-in at (4,2) is genuinely nonzero)
        for j in 1:m
            for idx in diagonalIndex[j]:(diagonalIndex[j + 1] - 1)
                i = sparse_mat.column_index[idx]
                if i == j
                    sparse_mat.sp_ele[idx] = BigFloat(10) + j
                else
                    sparse_mat.sp_ele[idx] = BigFloat("0.1") * (j + i)
                end
            end
        end

        # RHS in permuted coordinates.
        g = fill(BigFloat(1), m)

        # ---- SPARSE path: sparse_cholesky! + solve_systems_sparse! ----
        sm_sp = SDPA.SDPASparseMatrix{BigFloat}(m, m, :sparse, nnz)
        sm_sp.nonZeroCount = nnz
        sm_sp.nonZeroEffect = nnz
        copyto!(sm_sp.row_index, sparse_mat.row_index)
        copyto!(sm_sp.column_index, sparse_mat.column_index)
        SDPA._copyto_independent!(sm_sp.sp_ele, sparse_mat.sp_ele)
        ok = SDPA.sparse_cholesky!(sm_sp, diagonalIndex)
        @test ok
        x_sparse = SDPA._independent_zeros(BigFloat, m)
        SDPA.solve_systems_sparse!(x_sparse, sm_sp, g, diagonalIndex)

        # ---- DENSE path: Rpotrf_with_adjust! factor + ldiv! ----
        # (mirrors the solver's dense path exactly: cholesky_schur!
        # constructs `Cholesky(bmat, :U, 0)`, compute_dyvec! does
        # `copyto!(DyVec, gvec); ldiv!(F, DyVec)`).
        B = SDPA._independent_zeros(BigFloat, m, m)
        for j in 1:m
            for idx in diagonalIndex[j]:(diagonalIndex[j + 1] - 1)
                i = sparse_mat.column_index[idx]
                v = sparse_mat.sp_ele[idx]
                B[j, i] = v
                if i != j
                    B[i, j] = v
                end
            end
        end
        A = SDPA._independent_zeros(BigFloat, m, m)
        SDPA._copyto_independent!(A, B)
        info = SDPA.Rpotrf_with_adjust!("U", m, A; nb=64)
        @test info == 0
        F = Cholesky(A, :U, 0)
        x_dense = SDPA._independent_zeros(BigFloat, m)
        copyto!(x_dense, g)
        ldiv!(F, x_dense)

        # ---- the regression assertion ----
        denom = norm(x_dense)
        @test denom > 0
        rel = norm(x_sparse - x_dense) / denom
        # 256-bit: a correct solve agrees to ~1e-70; an under-closed
        # profile yields rel ~7e-4. 1e-20 separates them by 50+ orders.
        @test rel < BigFloat("1e-20")
    end
end
