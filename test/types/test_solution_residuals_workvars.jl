@testset "solution/residuals/workvars" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            # Build the same test problem used by test_SDPAProblem.jl.
            m = 2
            blockStruct = [2, -1]
            b = SDPA.SDPAVector{T}([T(1), T(4)])
            C = SDPA.SDPASparseLinearSpace{T}(1, 1)
            C.SDP_sp_index = [1]
            C.LP_sp_index = [1]
            C.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 2)
            SDPA.setelement_sdp!(C, 1, 1, 1, T(5))
            SDPA.setelement_sdp!(C, 1, 2, 2, T(7))
            SDPA.setelement_lp!(C, 1, T(11))
            A1 = SDPA.SDPASparseLinearSpace{T}(1, 1)
            A1.SDP_sp_index = [1]
            A1.LP_sp_index = [1]
            A1.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
            SDPA.setelement_sdp!(A1, 1, 1, 1, T(1))
            SDPA.setelement_lp!(A1, 1, T(0))
            A2 = SDPA.SDPASparseLinearSpace{T}(1, 1)
            A2.SDP_sp_index = [1]
            A2.LP_sp_index = [1]
            A2.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
            SDPA.setelement_sdp!(A2, 1, 2, 2, T(2))
            SDPA.setelement_lp!(A2, 1, T(3))
            problem = SDPA.SDPAProblem{T}(m, 2, blockStruct, b, C, [A1, A2])
            SDPA.initialize_index!(problem)

            # --- SDPASolution: lambda=1e4 initialization ---
            lambda = T(10000)
            sol = SDPA.SDPASolution{T}(m, blockStruct, lambda)
            # nDim = sum of |bs| over ALL blocks (SDP + LP).
            # Matches C++ Solutions::initialize at sdpa_dataset.cpp:37-44
            # which sums SDP block sizes + LP_nBlock.
            # blockStruct = [2, -1] → nDim = |2| + |-1| = 3
            @test sol.nDim == 3
            @test sol.mDim == 2
            # xMat.SDP_block[1] = lambda * I_2x2
            @test sol.xMat.SDP_block[1][1, 1] == lambda
            @test sol.xMat.SDP_block[1][2, 2] == lambda
            @test sol.xMat.SDP_block[1][1, 2] == zero(T)
            # yVec = 0
            @test all(iszero, sol.yVec.ele)
            # invCholeskyX = I / sqrt(lambda)
            sqrt_inv_lambda = inv(sqrt(lambda))
            @test sol.invCholeskyX.SDP_block[1][1, 1] == sqrt_inv_lambda
            # invzMat = I / lambda
            inv_lambda = inv(lambda)
            @test sol.invzMat.SDP_block[1][1, 1] == inv_lambda

            # --- SDPAResiduals: compute! on iteration-0 state ---
            res = SDPA.SDPAResiduals{T}(problem, sol)
            SDPA.compute!(res, problem, sol)
            # primalVec = b - <A, xMat>
            #   <A1, xMat> = 1*1e4 + 0*1e4 + 0*1e4 (LP) = 1e4
            #   <A2, xMat> = 0*1e4 + 2*1e4 + 3*1e4 (LP) = 5e4
            # primalVec = [1 - 1e4, 4 - 5e4] = [-9999, -49996]
            @test res.primalVec[1] == T(1) - T(10000)
            @test res.primalVec[2] == T(4) - T(50000)
            # max-abs primal norm
            @test res.normPrimalVec == T(50000) - T(4)  # = 49996
            # dualMat = C - sum yVec[i]*A_i - zMat = C - 0 - zMat
            # SDP block: [[5 - 1e4, 0], [0, 7 - 1e4]]
            # LP block: 11 - 0 - 1e4 = -9989
            @test res.dualMat.SDP_block[1][1, 1] == T(5) - T(10000)
            @test res.dualMat.SDP_block[1][2, 2] == T(7) - T(10000)
            @test res.dualMat.LP_block[1] == T(11) - T(10000)
            # max-abs dual norm = 1e4 - 5 = 9995
            @test res.normDualMat == T(10000) - T(5)

                        # --- SDPAWorkVariables: shape sanity ---
            work = SDPA.SDPAWorkVariables{T}(blockStruct)
            # DLS1 has no size() (deliberately removed: mixing SDP block
            # count with LP scalar count is not a meaningful dimension);
            # assert the block layout directly instead.
            @test_throws MethodError size(work.DLS1)
            @test work.DLS1.SDP_nBlock == 1          # one 2×2 SDP block
            @test work.DLS1.LP_nBlock == 1           # -1 → one LP scalar
            @test length(work.DLS1.SDP_block) == 1
            @test length(work.DLS1.LP_block) == 1

            # --- central-path LP fills must be independent elements ---
            # LP block with >= 2 scalars (blockStruct = [2, -2] -> LP_nBlock = 2)
            # so LP_block[1] !== LP_block[2] is checkable.
            lp_m = 2
            lp_blockStruct = [2, -2]
            lp_sol = SDPA.SDPASolution{T}(lp_m, lp_blockStruct, lambda)
            @test lp_sol.xMat.LP_nBlock == 2
            # Values: xMat/zMat = lambda, invCholeskyX/Z = 1/sqrt(lambda),
            # invzMat = 1/lambda (central-path init, all LP scalars).
            @test all(==(lambda), lp_sol.xMat.LP_block)
            @test all(==(lambda), lp_sol.zMat.LP_block)
            @test all(==(sqrt_inv_lambda), lp_sol.invCholeskyX.LP_block)
            @test all(==(sqrt_inv_lambda), lp_sol.invCholeskyZ.LP_block)
            @test all(==(inv_lambda), lp_sol.invzMat.LP_block)
            if T === BigFloat
                # Each LP scalar must be its own BigFloat object (no shared
                # alias from fill!). In-place MPFR mutation of one element
                # must leave the others untouched.
                @test lp_sol.xMat.LP_block[1] !== lp_sol.xMat.LP_block[2]
                @test lp_sol.zMat.LP_block[1] !== lp_sol.zMat.LP_block[2]
                @test lp_sol.invCholeskyX.LP_block[1] !== lp_sol.invCholeskyX.LP_block[2]
                @test lp_sol.invCholeskyZ.LP_block[1] !== lp_sol.invCholeskyZ.LP_block[2]
                @test lp_sol.invzMat.LP_block[1] !== lp_sol.invzMat.LP_block[2]

                SDPA.mp_add!(lp_sol.xMat.LP_block[1], lp_sol.xMat.LP_block[1], BigFloat(1))
                @test lp_sol.xMat.LP_block[1] == lambda + one(T)
                @test lp_sol.xMat.LP_block[2] == lambda
                SDPA.mp_add!(lp_sol.invzMat.LP_block[1], lp_sol.invzMat.LP_block[1], BigFloat(1))
                @test lp_sol.invzMat.LP_block[1] == inv_lambda + one(T)
                @test lp_sol.invzMat.LP_block[2] == inv_lambda
            end
        end
    end

    @testset "LP dual residual with C absent at LP scalars" begin
        # C's LP storage is a compact nonzero list.
        # The dual-residual LP loop must apply the `- zMat` term to EVERY
        # LP scalar, NOT only inside the loop over C's nonzeros — an LP
        # scalar where C has no entry still needs dualMat_LP =
        # C_LP - zMat_LP (its residual is -zMat, not 0).
        for T in (Float64, BigFloat)
            m = 1
            blockStruct = [-1, 2]
            b = SDPA.SDPAVector{T}([T(1)])
            # C touches ONLY the SDP block — NO LP entries at all.
            C = SDPA.SDPASparseLinearSpace{T}(1, 0)
            C.SDP_sp_index = [2]  # problem-block 2 is the SDP block
            C.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
            SDPA.setelement_sdp!(C, 1, 1, 1, T(3))
            # A_1 touches the LP scalar (block 1) and the SDP block (block 2).
            A1 = SDPA.SDPASparseLinearSpace{T}(1, 1)
            A1.SDP_sp_index = [2]
            A1.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
            SDPA.setelement_sdp!(A1, 1, 1, 1, T(1))
            SDPA.setelement_lp!(A1, 1, T(2))
            problem = SDPA.SDPAProblem{T}(m, 2, blockStruct, b, C, [A1])
            SDPA.initialize_index!(problem)

            lambda = T(10000)
            sol = SDPA.SDPASolution{T}(m, blockStruct, lambda)
            res = SDPA.SDPAResiduals{T}(problem, sol)
            SDPA.compute!(res, problem, sol)

            # dualMat_LP[1] = C_LP[1] - zMat_LP[1] = 0 - lambda
            # (C has no LP entry; zMat = lambda * I at initialization).
            @test res.dualMat.LP_block[1] == -lambda
            # SDP block unchanged by the LP fix: C[1,1] - zMat[1,1].
            @test res.dualMat.SDP_block[1][1, 1] == T(3) - lambda
            # norm must dominate: |3 - 1e4| < |0 - 1e4| -> 1e4
            @test res.normDualMat == lambda
        end
    end

    @testset "SDP dual residual with C absent at an SDP block" begin
        # The `- zMat` subtraction must run for EVERY SDP block, NOT
        # nested inside the loop over C's assigned sparse blocks: an SDP
        # block with no C entries (unassigned in sparse C, e.g. a
        # zero-objective block of a manually built problem) must get
        # dualMat_SDP = C_SDP - zMat_SDP too. Otherwise its dual residual
        # omits the -Z term and the phase classification is wrong from
        # iteration 0.
        for T in (Float64, BigFloat)
            m = 1
            blockStruct = [2, 2]
            b = SDPA.SDPAVector{T}([T(1)])
            # C touches ONLY block 1 — block 2 stays unassigned.
            C = SDPA.SDPASparseLinearSpace{T}(2, 0)
            C.SDP_sp_index = [1, 2]
            C.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
            SDPA.setelement_sdp!(C, 1, 1, 1, T(3))
            # A_1 touches ONLY block 2 (the block C does not cover).
            A1 = SDPA.SDPASparseLinearSpace{T}(2, 0)
            A1.SDP_sp_index = [1, 2]
            A1.SDP_sp_block[2] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
            SDPA.setelement_sdp!(A1, 2, 1, 1, T(1))
            problem = SDPA.SDPAProblem{T}(m, 2, blockStruct, b, C, [A1])
            SDPA.initialize_index!(problem)

            lambda = T(10000)
            sol = SDPA.SDPASolution{T}(m, blockStruct, lambda)
            res = SDPA.SDPAResiduals{T}(problem, sol)
            SDPA.compute!(res, problem, sol)

            # Block 1 (has C): C[1,1] - lambda (y = 0 at the start point).
            @test res.dualMat.SDP_block[1][1, 1] == T(3) - lambda
            # Block 2 (no C entry): 0 - lambda, NOT 0 (pre-fix: -0.0).
            @test res.dualMat.SDP_block[2][1, 1] == -lambda
            # The uncovered block dominates the dual norm.
            @test res.normDualMat == lambda
        end
    end
end
