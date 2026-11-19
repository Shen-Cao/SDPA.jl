@testset "NewtonState" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            # Allocate NewtonState for example1's shape (m=3, blockStruct=[2]).
            ns = SDPA.NewtonState{T}(3, [2])
            @test size(ns.bmat.de_ele) == (3, 3)
            @test length(ns.gvec) == 3
            @test ns.DxMat.SDP_nBlock == 1
            @test size(ns.DxMat.SDP_block[1].de_ele) == (2, 2)
            @test length(ns.DyVec) == 3
            @test ns.DzMat.SDP_nBlock == 1
            @test ns.r_zinvMat.SDP_nBlock == 1
            @test ns.x_rd_zinvMat.SDP_nBlock == 1
            @test size(ns.useFormula) == (3, 1)  # per-(constraint, block), nBlock=1
            @test all(f === :F1 for f in ns.useFormula)
            @test ns.cholesky_factorization === nothing

            # Multi-block allocation.
            ns2 = SDPA.NewtonState{T}(5, [2, -3, 4])
            @test size(ns2.bmat.de_ele) == (5, 5)
            @test ns2.DxMat.SDP_nBlock == 2  # 2 SDP blocks
            @test ns2.DxMat.LP_nBlock == 3   # 1 LP block

            # Type-stability: constructor returns the right parametric type.
            @test typeof(ns.bmat) === SDPA.SDPADenseMatrix{T}
            @test typeof(ns.gvec) === SDPA.SDPAVector{T}
            @test typeof(ns.DxMat) === SDPA.SDPADenseLinearSpace{T}
            @test typeof(ns.DyVec) === SDPA.SDPAVector{T}
            @test typeof(ns.DzMat) === SDPA.SDPADenseLinearSpace{T}
        end
    end

    # Lazy dense-bmat allocation: the forced-sparse path must be able to
    # construct a NewtonState WITHOUT the m×m dense bmat (which the sparse
    # path never uses), and allocate it on demand via ensure_dense_bmat!
    # only when the dense path actually runs. The default constructor
    # keeps allocating so all pre-existing tests are unchanged.
    @testset "lazy dense-bmat allocation" begin
        for T in (Float64, BigFloat)
            @testset "$T" begin
                # (a) allocate_dense_bmat=false → no dense bmat allocated,
                #     rest of the state still fully sized.
                ns_lazy = SDPA.NewtonState{T}(3, [2]; allocate_dense_bmat=false)
                @test ns_lazy.bmat === nothing
                @test length(ns_lazy.gvec) == 3
                @test size(ns_lazy.useFormula) == (3, 1)
                @test ns_lazy.cholesky_factorization === nothing

                # (b) ensure_dense_bmat! allocates on demand at the right
                #     size and is idempotent — the second call returns the
                #     SAME object (no reallocation).
                bmat = SDPA.ensure_dense_bmat!(ns_lazy)
                @test bmat isa SDPA.SDPADenseMatrix{T}
                @test size(bmat.de_ele) == (3, 3)
                @test ns_lazy.bmat === bmat
                @test SDPA.ensure_dense_bmat!(ns_lazy) === bmat

                # (c) the default constructor still allocates a real bmat.
                ns_default = SDPA.NewtonState{T}(3, [2])
                @test ns_default.bmat isa SDPA.SDPADenseMatrix{T}
                @test size(ns_default.bmat.de_ele) == (3, 3)
            end
        end
    end
end
