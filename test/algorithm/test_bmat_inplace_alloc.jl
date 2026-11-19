# test/algorithm/test_bmat_inplace_alloc.jl
# In-place `mp_add!` for the DENSE bmat assembly at BOTH sites —
# compute_bmat_dense! (SchurComplement.jl) and
# threaded_compute_bmat_dense! (ThreadedOps.jl).
#
# `Matrix{BigFloat}` getindex returns the MUTABLE object reference
# (test/mpfr/test_aliasing.jl), and `bmat.de_ele` is allocated via
# `_independent_zeros` (SDPADenseMatrix.jl), so
# `mp_add!(bmat[i,j], bmat[i,j], value_buf)` mutates the array element in
# place with ZERO allocation and the SAME single rounding as `+=`
# (bit-identical results) — the BigFloat path thus avoids the one ~96 B
# allocation per (k1,k2) pair that `bmat[i,j] += value_buf` would cost.
# The Float64 path is unchanged (`+=` is optimal for isbits).

@testset "dense bmat assembly: allocation gate and bit-identical paths" begin
    # ---------------------------------------------------------------------
    # SDP-only fixture for the allocation gate (NO LP blocks — the LP
    # path `_compute_bmat_lp!` still allocates 3 BigFloats/pair and is
    # tracked separately). A single 3x3 SDP block with 400
    # constraints, all full 3x3 matrices (equal nonzero counts -> the
    # (inz < jnz) skip never fires, so every pair with k1 >= k2 accumulates:
    # N = m^2 = 160,000 accumulation operations).
    # ---------------------------------------------------------------------
    function _build_alloc_fixture(::Type{T}) where {T}
        bs = 3
        n = 400
        blockStruct = [bs]
        b = SDPA.SDPAVector{T}(ones(T, n))
        C = SDPA.SDPASparseLinearSpace{T}(1, 0)
        C.SDP_sp_index = [1]
        C.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(bs, bs, :sparse, 9)
        for (r, c, v) in ((1,1,T(1)), (1,2,T(2)), (1,3,T(3)), (2,1,T(2)),
                          (2,2,T(4)), (2,3,T(5)), (3,1,T(3)), (3,2,T(5)),
                          (3,3,T(6)))
            SDPA.setelement_sdp!(C, 1, r, c, v)
        end
        A = Vector{typeof(C)}(undef, n)
        for k in 1:n
            Ak = SDPA.SDPASparseLinearSpace{T}(1, 0)
            Ak.SDP_sp_index = [1]
            Ak.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(bs, bs, :sparse, 9)
            s = T(k)  # distinct values so results are nontrivial
            for (r, c, v) in ((1,1,T(1)+s), (1,2,T(2)), (1,3,T(3)), (2,1,T(2)),
                              (2,2,T(4)+s), (2,3,T(5)), (3,1,T(3)), (3,2,T(5)),
                              (3,3,T(6)))
                SDPA.setelement_sdp!(Ak, 1, r, c, v)
            end
            A[k] = Ak
        end
        problem = SDPA.SDPAProblem{T}(n, 1, blockStruct, b, C, A)
        SDPA.initialize_index!(problem)
        return problem
    end

    function _setup_alloc_case(problem::SDPA.SDPAProblem{T}) where {T}
        params = SDPA.Parameters{T}()
        current_pt = SDPA.SDPASolution{T}(problem.m, problem.blockStruct, params.lambdaStar)
        work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)
        ns = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
        SDPA.compute_formula!(ns, problem)
        return current_pt, work, ns
    end

    @testset "allocation gate at BigFloat(200): serial + threaded" begin
        setprecision(BigFloat, 200) do
            T = BigFloat
            problem = _build_alloc_fixture(T)
            current_pt, work, ns = _setup_alloc_case(problem)
            nst = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
            SDPA.compute_formula!(nst, problem)

            # Warmup (compile) both sites.
            SDPA.compute_bmat_dense!(ns, problem, current_pt, work)
            SDPA.threaded_compute_bmat_dense!(nst, problem, current_pt, work)

            # Allocation budgets (bytes) for the two gates below —
            # ~10% headroom over the expected steady-state allocation:
            #   serial   gate < 1,552,038
            #   threaded gate < 1,648,132
            # Typical actuals: serial ≈ 160 KB, threaded ≈ 1.1 MB.
            # BOTH assertions require BOTH sites in-place.
            GC.gc()
            serial_alloc = @allocated SDPA.compute_bmat_dense!(ns, problem, current_pt, work)
            GC.gc()
            threaded_alloc = @allocated SDPA.threaded_compute_bmat_dense!(nst, problem, current_pt, work)

            @test serial_alloc < round(Int, 1_552_038 * 1.1)    # serial budget +10% headroom
            @test threaded_alloc < round(Int, 1_648_132 * 1.1)  # threaded budget +10% headroom

            # Sanity: the fixture actually exercised the accumulation.
            @test any(!iszero, ns.bmat.de_ele)
            @test any(!iszero, nst.bmat.de_ele)
        end
    end

    # threaded_compute_bmat_dense! must acquire its work1/work2 scratch
    # (plus the BigFloat scalar accumulators) ONCE per task via an
    # OhMyThreads @local TLV (BigFloatPool) and reuse it across all k1
    # iterations of the task's chunk, instead of allocating
    # _independent_zeros(bs, bs) per iteration.
    #
    # Allocation budget for the gate below (bs=10 fixture):
    # threaded gate < 1,008,132 bytes; typical actual ~235 KB.
    #
    # Fixture note: bs=10 (not the 3x3 fixture above) because the bs=3
    # fixture's residual is dominated by per-call allocations INSIDE the
    # math kernels (Rgemm! ~336 B + _sparse_dense_mul! accumulator ~112 B
    # per k1 call — pre-existing). Scratch allocation scales with bs^2
    # while kernel allocation is per-call, so the bs=10 fixture isolates
    # the pool wiring's effect.
    @testset "threaded bmat assembly reuses per-task workspace pool" begin
        setprecision(BigFloat, 200) do
            T = BigFloat
            bs = 10
            n = 400
            blockStruct = [bs]
            b = SDPA.SDPAVector{T}(ones(T, n))
            C = SDPA.SDPASparseLinearSpace{T}(1, 0)
            C.SDP_sp_index = [1]
            C.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(bs, bs, :sparse, bs * bs)
            for r in 1:bs, c in 1:bs
                SDPA.setelement_sdp!(C, 1, r, c, T(1))
            end
            A = Vector{typeof(C)}(undef, n)
            for k in 1:n
                Ak = SDPA.SDPASparseLinearSpace{T}(1, 0)
                Ak.SDP_sp_index = [1]
                Ak.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(bs, bs, :sparse, bs * bs)
                s = T(k)  # distinct values so results are nontrivial
                for r in 1:bs, c in 1:bs
                    SDPA.setelement_sdp!(Ak, 1, r, c, T(r * bs + c) + s)
                end
                A[k] = Ak
            end
            problem = SDPA.SDPAProblem{T}(n, 1, blockStruct, b, C, A)
            SDPA.initialize_index!(problem)
            params = SDPA.Parameters{T}()
            current_pt = SDPA.SDPASolution{T}(problem.m, problem.blockStruct, params.lambdaStar)
            work = SDPA.SDPAWorkVariables{T}(problem.blockStruct)
            nst = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
            SDPA.compute_formula!(nst, problem)

            # Warmup (compile).
            SDPA.threaded_compute_bmat_dense!(nst, problem, current_pt, work)

            GC.gc()
            p2_alloc = @allocated SDPA.threaded_compute_bmat_dense!(nst, problem, current_pt, work)
            @test p2_alloc < round(Int, 1_008_132 * 1.1)  # budget +10% headroom

            # Correctness: fixture actually exercised the accumulation.
            @test any(!iszero, nst.bmat.de_ele)
        end
    end

    @testset "correctness: in-place mp_add! bit-identical to old semantics" begin
        for T in (Float64, BigFloat)
            @testset "$T" begin
                for (name, path) in (("example1", "example1.dat-s"),
                                     ("sdp6_blocks", "sdp6_blocks.dat-s"))
                    problem = SDPA.read_dat_s(
                        joinpath(@__DIR__, "..", "fixtures", path), T)
                    current_pt, work, ns = _setup_alloc_case(problem)
                    nst = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
                    SDPA.compute_formula!(nst, problem)
                    nsm = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
                    SDPA.compute_formula!(nsm, problem)

                    SDPA.compute_bmat_dense!(ns, problem, current_pt, work)
                    SDPA.threaded_compute_bmat_dense!(nst, problem, current_pt, work)
                    SDPA.make_bmat!(nsm, problem, current_pt, work)  # dense dispatcher

                    B_s = ns.bmat.de_ele
                    B_t = nst.bmat.de_ele
                    B_m = nsm.bmat.de_ele

                    # Serial == threaded == make_bmat! bit-for-bit (each
                    # bmat element is written by exactly one (k1,k2) pair
                    # with a single mp_add! rounding — identical to the old
                    # `+=` semantics).
                    @test B_s == B_t
                    @test B_s == B_m
                    # Symmetry (the i!=j branch writes both triangles).
                    m = problem.m
                    @test all(B_s[i, j] == B_s[j, i] for i in 1:m for j in 1:m)
                    # Fixture actually exercised.
                    @test any(!iszero, B_s)
                end

                # Hand-built gate fixture: serial == threaded at both T.
                problem = _build_alloc_fixture(T)
                current_pt, work, ns = _setup_alloc_case(problem)
                nst = SDPA.NewtonState{T}(problem.m, problem.blockStruct)
                SDPA.compute_formula!(nst, problem)
                SDPA.compute_bmat_dense!(ns, problem, current_pt, work)
                SDPA.threaded_compute_bmat_dense!(nst, problem, current_pt, work)
                @test ns.bmat.de_ele == nst.bmat.de_ele
                @test any(!iszero, ns.bmat.de_ele)
            end
        end
    end
end
