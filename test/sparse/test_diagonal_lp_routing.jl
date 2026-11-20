using LinearAlgebra
using SparseArrays
using SDPA
using Test

function _separable_lp(::Type{T}, m; split=false, empty_row=false,
                       shared_scalar=false) where T
    n = m + 2
    bs = split ? [-m, -2] : [-n]
    C = SDPA.SDPASparseLinearSpace{T}(0, n)
    A = [SDPA.SDPASparseLinearSpace{T}(0, n) for _ in 1:m]
    b = SDPA.SDPAVector{T}(m)
    for i in 1:m
        ai = T(2 + (i % 3))
        SDPA.setelement_lp!(C, i, T(1 + (i % 3)))
        b.ele[i] = ai * T(1 + (i % 4))
        if i == m && empty_row
            b.ele[i] = zero(T)
        else
            lp = (i == m && shared_scalar) ? i - 1 : i
            SDPA.setelement_lp!(A[i], lp, ai)
        end
    end
    for k in 1:2
        SDPA.setelement_lp!(C, m + k, T(4))
        SDPA.setelement_lp!(A[k], m + k, one(T))
    end
    p = SDPA.SDPAProblem{T}(m, length(bs), bs, b, C, A)
    SDPA.initialize_index!(p)
    return p
end

@testset "exact diagonal Schur route for a single packed LP block" begin
    setprecision(BigFloat, 160) do
        for T in (Float64, BigFloat)
            m = 100
            p = _separable_lp(T, m)
            d = _separable_lp(T, m; split=true)  # Same flattened LP, dense default.
            use_dense, ordering, adj = SDPA.analyze_sparsity(p)
            @test !use_dense
            @test ordering == collect(1:m)
            @test nnz(adj) == m
            @test adj == SDPA.build_aggregate_graph(p)
            @test SDPA.analyze_sparsity(d)[1]

            params = SDPA.Parameters{T}(maxIteration=80, precision=160)
            log_sparse = @test_logs (:info, r"Block-level profile: 100 entries") match_mode=:any begin
                SDPA.solve!(p, params; io=devnull, reverse_primal_dual=false)
            end
            log_dense = @test_logs (:info, "Using dense Schur complement path") match_mode=:any begin
                SDPA.solve!(d, params; io=devnull, reverse_primal_dual=false)
            end
            @test log_sparse.terminal_phase == log_dense.terminal_phase == SDPA.pdOPT
            @test log_sparse.exit_reason == log_dense.exit_reason == SDPA.OPTIMAL_TERMINAL
            tol = T === BigFloat ? T("1e-24") : T(1e-6)
            @test log_sparse.final_obj_primal ≈ log_dense.final_obj_primal rtol=tol
            @test log_sparse.final_obj_dual ≈ log_dense.final_obj_dual rtol=tol
            @test log_sparse.xMat.LP_block ≈ log_dense.xMat.LP_block rtol=tol
            @test log_sparse.final_primal_res < tol
            @test log_sparse.final_dual_res < tol
        end
    end

    # The legacy route is unchanged for small LPs, for empty rows, or
    # when any two nonzero rows share an LP scalar.
    @test SDPA.analyze_sparsity(_separable_lp(Float64, 99))[1]
    for key in (:empty_row, :shared_scalar)
        p = _separable_lp(Float64, 100; (; key => true)...)
        @test SDPA.analyze_sparsity(p)[1]
        adj = SDPA.build_aggregate_graph(p)
        @test nnz(adj) == (key === :empty_row ? 100 : 102)
        @test adj[99, 100] == (key === :shared_scalar)
        params = SDPA.Parameters{Float64}(maxIteration=2)
        log = @test_logs (:info, "Using dense Schur complement path") match_mode=:any begin
            SDPA.solve!(p, params; io=devnull, reverse_primal_dual=false)
        end
        @test log.exit_reason == SDPA.CHOLESKY_FAILURE
    end
end
