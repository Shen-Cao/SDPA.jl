using LinearAlgebra, SDPA, Test

function _verification_gram_problem(::Type{T}, V; sdp=false) where {T}
    m = size(V,1)
    bs = sdp ? [2,2] : [-size(V,2)]
    n = length(bs)
    b = SDPA.SDPAVector{T}(m)
    b.ele[end] = one(T)
    C = SDPA.SDPASparseLinearSpace{T}(sdp ? 2 : 0,sdp ? 0 : size(V,2))
    A = [SDPA.SDPASparseLinearSpace{T}(sdp ? 2 : 0,sdp ? 0 : size(V,2)) for _ in 1:m]
    if sdp
        for g in 1:2
            C.SDP_sp_block[g] = SDPA.SDPASparseMatrix{T}(2,2,:sparse,2)
            for r in 1:2
                SDPA.setelement_sdp!(C,g,r,r,one(T))
            end
            for a in A
                a.SDP_sp_block[g] = SDPA.SDPASparseMatrix{T}(2,2,:sparse,0)
            end
        end
        for i in 1:m,j in axes(V,2)
            iszero(V[i,j]) || SDPA.setelement_sdp!(A[i],1+(j-1)÷2,1+(j-1)%2,1+(j-1)%2,V[i,j])
        end
    else
        for i in 1:m,j in axes(V,2)
            SDPA.setelement_lp!(A[i],j,V[i,j])
        end
    end
    p=SDPA.SDPAProblem{T}(m,n,bs,b,C,A)
    SDPA.initialize_index!(p)
    return p
end

@testset "original-action enclosure rejects numerically false positive factors" begin
    setprecision(BigFloat,256) do
        for T in (Float64, BigFloat), sdp in (false, true)
            M = T === Float64 ? 16 : 64
            V = T[M 1 0; M+1 1 0; 0 0 1; 0 1 1]
            H = V*V'
            p = _verification_gram_problem(T,V;sdp)
            n = p.m
            plan = SDPA._certify_bordered(p; labels=sdp ? nothing : [1,1,2,0])
            pt = SDPA.SDPASolution{T}(n,p.blockStruct,one(T))
            ns = SDPA.NewtonState{T}(n,p.blockStruct;allocate_dense_bmat=false)
            F = SDPA._BorderedFactors{T}(plan.layout)
            ns.bordered = SDPA._BorderedSchur(F,plan.incidence,p.blockStruct)
            SDPA.compute_formula!(ns,p)
            SDPA.make_bmat!(ns,p,pt,SDPA.SDPAWorkVariables{T}(p.blockStruct))
            g = T[0,0,0,1]
            null = T[-(M+1),M,-1,1]
            use = T[1,-1,2,1]
            verifier = ns.bordered.verifier
            # Prove exact inconsistency separately from the candidate.
            setprecision(BigFloat,512) do
                @test all(iszero,BigFloat.(H)*BigFloat.(null))
                @test dot(BigFloat.(null),BigFloat.(g)) == 1
            end
            @test SDPA._verify_bordered_direction!(verifier,plan.incidence,p,pt,use,H*use)
            @test SDPA._verify_bordered_direction!(verifier,plan.incidence,p,pt,zeros(T,n),zeros(T,n))
            @test !SDPA._verify_bordered_direction!(verifier,plan.incidence,p,pt,null,zeros(T,n))
            # The raw specialized factor may accept a rounded positive
            # pivot. Neither that nor finite components are enough.
            @test SDPA._factor_bordered!(F)
            bad = SDPA._independent_zeros(T,n)
            @test SDPA._solve_bordered!(bad,F,g)
            @test !SDPA._verify_bordered_direction!(verifier,plan.incidence,p,pt,bad,g)
            copyto!(ns.gvec.ele,g)
            ns.DyVec.ele .= T.(1:n)
            previous = SDPA._copy_independent(ns.DyVec.ele)
            @test !SDPA._compute_verified_bordered_dyvec!(ns,false,p,pt)
            @test isequal(ns.DyVec.ele,previous)
            @test F.state === SDPA._BORDERED_FAILED
            @test_throws "original-action verification" SDPA.compute_dyvec!(ns,false)

            if sdp
                # Exercise the actual public predictor, not only private
                # factor/verification methods. C=I makes the initial
                # dual residual zero at the unit warm start.
                params=SDPA.Parameters{T}(precision=256,lambdaStar=one(T),maxIteration=1)
                got=SDPA.solve!(p,params;reverse_primal_dual=false,bordered_schur=true,
                                init=pt,verbose=false)
                @test got.exit_reason === SDPA.CHOLESKY_FAILURE
                @test got.iterations == 0
                @test all(iszero,got.yVec.ele)
                @test all(got.xMat.SDP_block[k].de_ele == pt.xMat.SDP_block[k].de_ele for k in 1:2)
                @test all(got.zMat.SDP_block[k].de_ele == pt.zMat.SDP_block[k].de_ele for k in 1:2)
            end
        end
    end
    @test_throws "exceeds 1000000 guard-precision slots" SDPA._bordered_verify_slots(100,500)
end

@testset "verifier acceptance target is rounded inward" begin
    for p in (128,224,384)
        setprecision(BigFloat,p) do
            v = SDPA._BorderedVerifier(BigFloat,2,0)
            rounded_to_nearest = setprecision(BigFloat,precision(v.tol)) do
                sqrt(ldexp(one(BigFloat),1-p))
            end
            # Here nearest is greater than the mathematical square root.
            # The original action's first component equals the recorded
            # coefficient exactly, while its RHS component is zero.
            @test v.tol < rounded_to_nearest
            V = reshape([rounded_to_nearest,one(BigFloat)],2,1)
            data = _verification_gram_problem(BigFloat,V)
            pt = SDPA.SDPASolution{BigFloat}(2,data.blockStruct,one(BigFloat))
            x = g = BigFloat[0,1]
            inc = SDPA._bordered_incidence(data)
            @test !SDPA._verify_bordered_direction!(v,inc,data,pt,x,g)
            SDPA.setelement_lp!(data.A[1],1,v.tol)
            @test SDPA._verify_bordered_direction!(v,inc,data,pt,x,g)
        end
    end
end
