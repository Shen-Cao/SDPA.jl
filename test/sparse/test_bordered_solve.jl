using LinearAlgebra
using SDPA
using Test

# I in all SDP cones and ones in all LP scalars are strictly primal
# feasible. The objective is the sum of diagonal and border equality
# rows (including the LP coefficients shared by both), so the exact
# objective follows from those right-hand sides.
function _bordered_solve_fixture(::Type{T}, sizes=(3, 4, 5, 6), k=3;
                                  singular=false) where {T}
    ng = length(sizes)
    ds = [max(2, findfirst(d -> d*(d+1)÷2 >= n, 1:n)) for n in sizes]
    nl = (ng + 1) * k
    bs = [-nl; ds]
    m = sum(sizes) + k
    C = SDPA.SDPASparseLinearSpace{T}(ng, nl)
    C.SDP_sp_index .= 2:(ng+1)
    A = [SDPA.SDPASparseLinearSpace{T}(ng, nl) for _ in 1:m]
    for a in A
        a.SDP_sp_index .= 2:(ng+1)
    end
    b = SDPA.SDPAVector{T}(m)
    start = 1
    for g in 1:ng
        d = ds[g]
        coords = vcat([(r,r) for r in 1:d], [(r,c) for r in 1:d for c in (r+1):d])
        C.SDP_sp_block[g] = SDPA.SDPASparseMatrix{T}(d, d, :sparse, d)
        for r in 1:d
            SDPA.setelement_sdp!(C, g, r, r, one(T))
        end
        for ii in 1:sizes[g]
            i = start + ii - 1
            A[i].SDP_sp_block[g] = SDPA.SDPASparseMatrix{T}(d,d,:sparse,1)
            r,c = coords[ii]
            SDPA.setelement_sdp!(A[i], g, r,c,one(T))
            b.ele[i] = T(r == c)
        end
        for j in 1:k
            v = (j-1)*(ng+1)+g
            i = start + (j-1)%d
            SDPA.setelement_lp!(A[i], v, one(T))
            SDPA.setelement_lp!(C, v, T(2))
            b.ele[i] += one(T)
        end
        start += sizes[g]
    end
    for j in 1:k
        for g in 1:(ng+1)
            v = (j-1)*(ng+1)+g
            SDPA.setelement_lp!(A[start+j-1], v, one(T))
            g == ng+1 && SDPA.setelement_lp!(C, v, one(T))
        end
        b.ele[start+j-1] = T(ng+1)
    end
    # The file parser establishes empty shells even for untouched blocks.
    # Our direct test builder must do the same for the legacy graph route.
    for a in A, g in 1:ng
        if !isassigned(a.SDP_sp_block, g)
            d = ds[g]
            a.SDP_sp_block[g] = SDPA.SDPASparseMatrix{T}(d, d, :sparse, 0)
        end
    end
    if singular
        A[2] = A[1]
        b.ele[2] = b.ele[1]
    end
    order = sortperm(1:m; by=i -> mod(17*i,m))
    b.ele .= b.ele[order]
    p = SDPA.SDPAProblem{T}(m,length(bs),bs,b,C,A[order])
    SDPA.initialize_index!(p)
    return p, T(sum(ds) + k*(2ng+1))
end

function _unpack_bordered_schur(F::_F) where {_F<:SDPA._BorderedFactors}
    n = length(F.layout.order)
    order = F.layout.order
    T = eltype(F.rhs)
    A = SDPA._independent_zeros(T,n,n)
    for (k,D) in enumerate([F.interiors; F.separator])
        start = F.starts[k]
        for i in 1:(length(D.offsets)-1), j in i:(length(D.offsets)-1)
            ri,rj = order[start+i-1],order[start+j-1]
            A[ri,rj] = A[rj,ri] = D.sp_ele[D.offsets[i]+(j-i)]
        end
    end
    for (k,E) in enumerate(F.panels)
        for j in axes(E,2), i in axes(E,1)
            ri,rj = order[F.starts[k]+i-1],order[F.starts[end]+j-1]
            A[ri,rj] = A[rj,ri] = E[i,j]
        end
    end
    return A
end

@testset "streamed bordered assembly matches legacy terms in physical order" begin
    setprecision(BigFloat, 160) do
        for T in (Float64, BigFloat), which in (:mixed, :f4)
            p = which === :mixed ? first(_bordered_solve_fixture(T)) :
                SDPA.read_dat_s(joinpath(@__DIR__,"..","fixtures","f4_mixed_dense_sparse.dat-s"),T)
            n = p.m
            plan = SDPA._certify_bordered(p)
            work = SDPA.SDPAWorkVariables{T}(p.blockStruct)
            pt = SDPA.SDPASolution{T}(n,p.blockStruct,T(10))
            # X and Z^{-1} need not be diagonal in actual solves. Add
            # dyadic off-diagonals to exercise dense and sparse formulas.
            for k in 1:pt.xMat.SDP_nBlock, mat in (pt.xMat,pt.invzMat)
                D = mat.SDP_block[k].de_ele
                D[1,2] = D[2,1] = T(1)/T(128)
            end
            ns = SDPA.NewtonState{T}(n,p.blockStruct;allocate_dense_bmat=false)
            ns.bordered = SDPA._BorderedSchur(SDPA._BorderedFactors{T}(plan.layout),plan.incidence)
            SDPA.compute_formula!(ns,p)
            SDPA.make_bmat!(ns,p,pt,work)
            @test ns.bmat === ns.sparse_bmat === nothing
            @test isempty(ns.SDP_location_sparse_bMat)
            @test isempty(ns._lp_all_values)
            @test isempty(ns._sdp_all_values)
            @test ns.bordered.factors.state === SDPA._BORDERED_ASSEMBLED
            B = _unpack_bordered_schur(ns.bordered.factors)

            legacy = SDPA.NewtonState{T}(n,p.blockStruct;allocate_dense_bmat=false)
            legacy.use_sparse = true
            legacy.ordering = legacy.reverse_ordering = collect(1:n)
            legacy.sparse_bmat = SDPA._SchurSkyline{T}(fill(n,n))
            SDPA.compute_formula!(legacy,p)
            SDPA.make_aggrigate_index_sdp!(legacy,p,legacy.ordering)
            SDPA.make_bmat!(legacy,p,pt,work)
            out = legacy.sparse_bmat
            @test all(isequal(B[i,j],out.sp_ele[out.offsets[i]+(j-i)]) for i in 1:n for j in i:n)
        end
    end
end

@testset "public bordered opt-in success and refusal" begin
    setprecision(BigFloat,160) do
        for T in (Float64,BigFloat)
            p, expected = _bordered_solve_fixture(T)
            params = SDPA.Parameters{T}(maxIteration=100,precision=160)
            caps = SDPA._certify_bordered(p).layout.slots
            logs, got = Test.collect_test_logs() do
                SDPA.solve!(p,params; bordered_schur=true,bordered_max_slots=caps,
                            reverse_primal_dual=false, io=devnull)
            end
            msgs = string.(getfield.(logs,:message))
            @test "Using certified bordered Schur complement path" in msgs
            @test !any(s -> occursin("Using dense Schur",s) || occursin("Using sparse Schur",s),msgs)
            ref = SDPA.solve!(p,params;reverse_primal_dual=false,verbose=false)
            tol = T === BigFloat ? T("1e-24") : T(1e-6)
            @test got.iterations > 1
            @test got.exit_reason === ref.exit_reason === SDPA.OPTIMAL_TERMINAL
            @test got.terminal_phase === ref.terminal_phase === SDPA.pdOPT
            for result in (ref,got)
                @test result.final_obj_primal ≈ expected rtol=tol
                @test result.final_obj_dual ≈ expected rtol=tol
                @test result.final_primal_res < tol
                @test result.final_dual_res < tol
            end
            @test_throws "factor/panel slots" SDPA.solve!(p,params; bordered_schur=true,
                        bordered_max_slots=caps-1,reverse_primal_dual=false)
            @test_throws "mutually exclusive" SDPA.solve!(p,params; bordered_schur=true,force_sparse=true)
            @test_throws "requires bordered_schur=true" SDPA.solve!(p,params; bordered_max_slots=caps)

            L = SDPA._certify_bordered(p).layout
            a = first(i for i in 1:p.m if L.island_of[i] == 1 && !isempty(p.A[i].LP_sp_index))
            b = findfirst(==(2), L.island_of)
            SDPA.setelement_lp!(p.A[b],p.A[a].LP_sp_index[1],one(T))
            SDPA.initialize_index!(p)
            @test_throws "different islands" SDPA.solve!(p,params; bordered_schur=true,
                        reverse_primal_dual=false)
        end
        p,_ = _bordered_solve_fixture(BigFloat)
        params = SDPA.Parameters{BigFloat}(precision=160)
        setrounding(BigFloat,RoundDown) do
            @test_throws "nearest rounding" SDPA.solve!(p,params; bordered_schur=true,
                        reverse_primal_dual=false)
        end
    end
end

@testset "numeric rejection returns unmodified last accepted iterate" begin
    setprecision(BigFloat,160) do
        for T in (Float64,BigFloat)
            p,_ = _bordered_solve_fixture(T,(3,4,5,6),3;singular=true)
            params=SDPA.Parameters{T}(precision=160)
            initial=SDPA.SDPASolution{T}(p.m,p.blockStruct,params.lambdaStar)
            snapshot=copy(initial)
            result=SDPA.solve!(p,params; bordered_schur=true,reverse_primal_dual=false,
                               init=initial,verbose=false)
            @test result.exit_reason === SDPA.CHOLESKY_FAILURE
            @test result.iterations == 0
            @test isempty(result.trajectory)
            @test result.yVec.ele == initial.yVec.ele == snapshot.yVec.ele
            for (R,V,O) in ((result.xMat,initial.xMat,snapshot.xMat),(result.zMat,initial.zMat,snapshot.zMat))
                @test R.LP_block == V.LP_block == O.LP_block
                @test all(R.SDP_block[i].de_ele == V.SDP_block[i].de_ele == O.SDP_block[i].de_ele
                          for i in eachindex(R.SDP_block))
            end
        end
    end
end
