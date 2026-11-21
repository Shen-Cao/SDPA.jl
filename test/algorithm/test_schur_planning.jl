# A literal small-n oracle for the old cost model, deliberately using its
# quadratic comparisons rather than the production prefix computation.
function _schur_planning_legacy(problem::SDPA.SDPAProblem{T}, kappa::T) where T
    formulas = fill(:F1, problem.m, problem.nBlock)
    counts = [zeros(Int, problem.m) for _ in 1:problem.nBlock]
    for l in 1:problem.nBlock
        problem.blockStruct[l] >= 2 || continue
        cs, bs = problem.SDP_constraint[l], problem.SDP_blockIndex[l]
        active = problem.SDP_nConstraint[l]
        for k in 1:active
            i = cs[k]
            a = problem.A[i].SDP_sp_block[bs[k]]
            w = a.nonZeroEffect
            up = w
            for q in 1:active
                j = cs[q]
                v = problem.A[j].SDP_sp_block[bs[q]].nonZeroEffect
                if v < w || (v == w && j < i)
                    up += v
                end
            end
            counts[l][i] = up
            n = a.nRow
            F1 = kappa * n * w + n * n * n + kappa * T(up)
            F2 = kappa * n * w + kappa * (n + 1) * T(up)
            F3 = (2 * kappa * w + 1) * T(up)
            formulas[i, l] = if a.type === :dense || w >= n * a.nCol
                F1 < F2 ? :F1 : :F2
            elseif F1 < F2 && F1 < F3
                :F1
            elseif F2 < F3
                :F2
            else
                :F3
            end
        end
    end
    return formulas, counts
end

# LP first, multiple SDP positions (some A_i store only a subset), and
# an unused SDP block. Weight-list order is not canonical after reversing.
function _schur_planning_fixture(::Type{T}) where T
    shapes = [-2, 3, 2, 2, 2, 5]
    m = 16
    A = SDPA.SDPASparseLinearSpace{T}[]
    for i in 1:m
        specs = [(2, :sparse, 1)]
        i <= 2 && push!(specs, (3, :dense, 4))
        i in (3, 4, 7) && push!(specs, (4, :sparse, i == 7 ? 5 : 2))
        i in (5, 6) && push!(specs, (5, :sparse, i == 5 ? 3 : 1))
        a = SDPA.SDPASparseLinearSpace{T}(length(specs), 2)
        for (k, (l, storage, w)) in enumerate(specs)
            n = shapes[l]
            a.SDP_sp_index[k] = l
            a.SDP_sp_block[k] = SDPA.SDPASparseMatrix{T}(n, n, storage, w)
            if storage === :dense
                SDPA.setelement_sdp!(a, k, 1, 1, one(T))
            else
                # Intentionally allow repeated coordinates in sparse storage.
                # Only duplicate *constraint incidence IDs* are invalid.
                for j in 1:w
                    row = mod1(j, n)
                    SDPA.setelement_sdp!(a, k, row, row, one(T))
                end
            end
        end
        i == 8 && SDPA.setelement_lp!(a, 2, one(T))
        push!(A, a)
    end
    p = SDPA.SDPAProblem{T}(m, length(shapes), shapes, SDPA.SDPAVector{T}(m),
                           SDPA.SDPASparseLinearSpace{T}(0, 2), A)
    return SDPA.initialize_index!(p)
end

@testset "Schur formula planning: original-ID ordering and exact cost ties" begin
    for T in (Float64, BigFloat)
        p = _schur_planning_fixture(T)
        ns = SDPA.NewtonState{T}(p.m, p.blockStruct; allocate_dense_bmat=false)
        for reverse_lists in (false, true)
            if reverse_lists
                for (cs, bs) in zip(p.SDP_constraint, p.SDP_blockIndex)
                    reverse!(cs)
                    reverse!(bs)
                end
                ns.ordering = collect(p.m:-1:1)
                ns.reverse_ordering = copy(ns.ordering)
            end
            # Inactive tails are not interpreted as current incidence entries.
            push!(p.SDP_constraint[end], typemax(Int))
            for kappa in (one(T), T(1.2), T(0.17))
                expected, counts = _schur_planning_legacy(p, kappa)
                before = deepcopy((p.SDP_constraint, p.SDP_blockIndex))
                @test SDPA.compute_formula!(ns, p; kappa=kappa) === ns
                @test ns.useFormula == expected
                @test (p.SDP_constraint, p.SDP_blockIndex) == before
                @test all(==(:F1), ns.useFormula[:, 1])  # LP column untouched
                @test all(==(:F1), ns.useFormula[:, end]) # inactive SDP column
                @test ns._up_counts_scratch[[5,6]] == counts[5][[5,6]]
                if kappa == one(T)
                    @test ns.useFormula[15, 2] === :F3  # F1 == F3 < F2
                    @test ns.useFormula[16, 2] === :F1
                    @test ns.useFormula[1, 3] === :F2   # dense F1 == F2
                    @test ns.useFormula[2, 3] === :F1
                    @test ns.useFormula[3, 4] === :F3   # sparse F2 == F3 < F1
                    @test ns.useFormula[5, 5] === :F2   # sparse F1 == F2 < F3
                    @test ns.useFormula[7, 4] !== :F3   # repeated triplets, w >= n²
                end
            end
            pop!(p.SDP_constraint[end])
        end
    end
end

@testset "Schur formula planning rejects invalid active data and overflow" begin
    function expect_error(change, exception)
        p = _schur_planning_fixture(Float64)
        ns = SDPA.NewtonState{Float64}(p.m, p.blockStruct; allocate_dense_bmat=false)
        change(p)
        @test_throws exception SDPA.compute_formula!(ns, p)
    end
    expect_error(ArgumentError) do p
        p.SDP_nConstraint[2] += 1
    end
    expect_error(ArgumentError) do p
        p.SDP_nConstraint[2] = -1
    end
    expect_error(ArgumentError) do p
        p.SDP_constraint[2][1] = p.m + 1
    end
    expect_error(ArgumentError) do p
        p.SDP_blockIndex[2][1] = typemax(Int)
    end
    expect_error(ArgumentError) do p
        p.SDP_blockIndex[2][1] = 2  # valid local position, wrong global SDP block
    end
    expect_error(ArgumentError) do p
        p.A[1].SDP_sp_block[1].nonZeroEffect = -1
    end
    expect_error(ArgumentError) do p
        p.SDP_constraint[2][2] = p.SDP_constraint[2][1] # identical key
    end
    expect_error(ArgumentError) do p
        a = p.A[1]
        push!(a.SDP_sp_index, 2)
        block = SDPA.SDPASparseMatrix{Float64}(3, 3, :sparse, 2)
        block.nonZeroEffect = 2
        push!(a.SDP_sp_block, block)
        a.SDP_sp_nBlock += 1
        p.SDP_constraint[2][2] = 1
        p.SDP_blockIndex[2][2] = a.SDP_sp_nBlock # same ID, different weight
    end
    expect_error(OverflowError) do p
        # Stress metadata directly; allocating this many triplets is not needed.
        p.A[1].SDP_sp_block[1].nonZeroEffect = typemax(Int)
    end
    expect_error(OverflowError) do p
        # Square still fits, but its cube is not a native-Int cost term.
        d = Int(1) << (8 * sizeof(Int) ÷ 3 + 1)
        p.blockStruct[2] = d
        for a in p.A
            block = a.SDP_sp_block[1]
            block.nRow = block.nCol = d
        end
    end
end

@testset "Schur formula planning preserves assembled values in both backends" begin
    for T in (Float64, BigFloat)
        p = SDPA.read_dat_s(joinpath(@__DIR__, "..", "fixtures", "lp_first_mixed.dat-s"), T)
        for (cs, bs) in zip(p.SDP_constraint, p.SDP_blockIndex)
            reverse!(cs)
            reverse!(bs)
        end
        parameters = SDPA.Parameters{T}()
        legacy, _ = _schur_planning_legacy(p, T(SDPA.KAPPA_FORMULA))
        for sparse in (false, true)
            function assemble(use_old)
                ns = SDPA.NewtonState{T}(p.m, p.blockStruct; allocate_dense_bmat=!sparse)
                if use_old
                    ns.useFormula .= legacy
                else
                    SDPA.compute_formula!(ns, p)
                end
                ns.use_sparse = sparse
                if sparse
                    ns.ordering = collect(p.m:-1:1)
                    ns.reverse_ordering = copy(ns.ordering)
                    reaches = SDPA.build_skyline_from_problem!(p, ns.ordering, ns.reverse_ordering)
                    ns.sparse_bmat = SDPA._SchurSkyline{T}(reaches)
                    SDPA.make_aggrigate_index_sdp!(ns, p, ns.ordering)
                end
                point = SDPA.SDPASolution{T}(p.m, p.blockStruct, parameters.lambdaStar)
                work = SDPA.SDPAWorkVariables{T}(p.blockStruct)
                SDPA.make_bmat!(ns, p, point, work)
                return sparse ? ns.sparse_bmat.sp_ele : vec(ns.bmat.de_ele)
            end
            @test all(isequal.(assemble(false), assemble(true)))
        end
    end
end

@testset "Sparse Schur table count preserves every emitted record in order" begin
    for T in (Float64, BigFloat)
        p = _schur_planning_fixture(T)
        ns = SDPA.NewtonState{T}(p.m, p.blockStruct; allocate_dense_bmat=false)
        ns.ordering = vcat(2:2:p.m, 1:2:p.m)
        ns.reverse_ordering = invperm(ns.ordering)
        reaches = SDPA.build_skyline_from_problem!(p, ns.ordering, ns.reverse_ordering)
        ns.sparse_bmat = SDPA._SchurSkyline{T}(reaches)
        slots = ns.sparse_bmat.offsets

        for reverse_lists in (false, true)
            if reverse_lists
                for (cs, bs) in zip(p.SDP_constraint, p.SDP_blockIndex)
                    reverse!(cs)
                    reverse!(bs)
                end
            end
            # Not part of the active prefix, and never used as an index.
            push!(p.SDP_constraint[2], typemax(Int))
            push!(p.SDP_blockIndex[2], 0)
            push!(p.SDP_constraint[end], typemax(Int))
            expected = NTuple{5,Int}[]
            block_lengths = zeros(Int, p.nBlock)
            for l in 1:p.nBlock
                cs, bs = p.SDP_constraint[l], p.SDP_blockIndex[l]
                for k1 in 1:p.SDP_nConstraint[l], k2 in 1:p.SDP_nConstraint[l]
                    i, j = cs[k1], cs[k2]
                    ib, jb = bs[k1], bs[k2]
                    inz = p.A[i].SDP_sp_block[ib].nonZeroEffect
                    jnz = p.A[j].SDP_sp_block[jb].nonZeroEffect
                    if inz < jnz || (inz == jnz && i < j)
                        continue
                    end
                    ii, jj = minmax(ns.reverse_ordering[i], ns.reverse_ordering[j])
                    push!(expected, (i, j, ib, jb, slots[ii] + jj - ii))
                    block_lengths[l] += 1
                end
            end
            # No compute_formula! call establishes any hidden marks.
            @test SDPA.make_aggrigate_index_sdp!(ns, p, ns.ordering) === ns
            actual = collect(zip(ns.SDP_constraint1, ns.SDP_constraint2, ns.blockIndex1,
                                 ns.blockIndex2, ns.SDP_location_sparse_bMat))
            @test actual == expected
            @test ns.SDP_block_table_len == block_lengths == [0, 136, 3, 6, 3, 0]
            pop!(p.SDP_constraint[2])
            pop!(p.SDP_blockIndex[2])
            pop!(p.SDP_constraint[end])
        end
    end
end

@testset "Sparse Schur count requires valid unique incidences and native capacity" begin
    function bad_incidence(change)
        p = _schur_planning_fixture(Float64)
        ns = SDPA.NewtonState{Float64}(p.m, p.blockStruct; allocate_dense_bmat=false)
        ns.ordering = collect(1:p.m)
        ns.reverse_ordering = copy(ns.ordering)
        ns.sparse_bmat = SDPA._SchurSkyline{Float64}(fill(p.m, p.m))
        change(p)
        @test_throws ArgumentError SDPA.make_aggrigate_index_sdp!(ns, p, ns.ordering)
        @test isempty(ns.SDP_constraint1)   # no records from the invalid first SDP block
    end
    bad_incidence() do p
        p.SDP_constraint[2][2] = p.SDP_constraint[2][1]
    end
    bad_incidence() do p
        p.SDP_constraint[2][2] = 0
    end
    bad_incidence() do p
        p.SDP_blockIndex[2][2] = 2 # belongs to a different original block
    end
    bad_incidence() do p
        p.SDP_nConstraint[2] += 1
    end
    @test SDPA._checked_sdp_pair_count(Int32(0)) === Int32(0)
    @test SDPA._checked_sdp_pair_count(Int32(65_535)) === Int32(2_147_450_880)
    @test_throws OverflowError SDPA._checked_sdp_pair_count(Int32(65_536))
    @test_throws ArgumentError SDPA._checked_sdp_pair_count(Int32(-1))
    n64 = Int64(4_294_967_295)
    @test SDPA._checked_sdp_pair_count(n64) == big(n64) * (big(n64) + 1) ÷ 2
    @test_throws OverflowError SDPA._checked_sdp_pair_count(n64 + 1)
end
