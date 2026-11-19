@testset "read_dat (dense format)" begin
    example1_dat_path = joinpath(@__DIR__, "..", "fixtures", "example1.dat")
    example1_dats_path = joinpath(@__DIR__, "..", "fixtures", "example1.dat-s")

    @testset "dense parser matches sparse parser on example1" begin
        for T in (Float64, BigFloat)
            pd = SDPA.read_dat(example1_dat_path, T)
            ps = SDPA.read_dat_s(example1_dats_path, T)

            @test pd.m == ps.m
            @test pd.nBlock == ps.nBlock
            @test pd.blockStruct == ps.blockStruct

            # b vector must match exactly.
            @test pd.b.ele == ps.b.ele

            # C: compare dense-form de_ele (both should be dense after
            # change_to_dense! at density 0.5 > 0.20).
            C_d = pd.C.SDP_sp_block[1].de_ele
            C_s = ps.C.SDP_sp_block[1].de_ele
            @test size(C_d) == size(C_s)
            @test C_d ≈ C_s

            # A_1, A_2, A_3
            for k in 1:3
                A_d = pd.A[k].SDP_sp_block[1].de_ele
                A_s = ps.A[k].SDP_sp_block[1].de_ele
                @test size(A_d) == size(A_s)
                @test A_d ≈ A_s
            end
        end
    end

    @testset "Float64 vs BigFloat agreement" begin
        # The same dense file parsed at Float64 and BigFloat should agree
        # to within eps(Float64).
        pf = SDPA.read_dat(example1_dat_path, Float64)
        pb = SDPA.read_dat(example1_dat_path, BigFloat)
        @test pf.b.ele ≈ [Float64(x) for x in pb.b.ele]
    end

    @testset "read_dat LP block count for multi-element LP blocks" begin
        # Synthetic dense .dat problem with blockStruct = [2, -3].
        # The LP block has dimension 3 (three LP scalars), not 1.
        # nLP must count LP scalars, not LP blocks:
        # nLP = sum(abs(bs) for bs < 0) = 3.
        for T in (Float64, BigFloat)
            mktemp() do path, io
                # Write a valid .dat file: mDim=1, nBlock=2, blockStruct=[2,-3]
                # b=[1], C=[1 0; 0 1]+[1,2,3], A1=[1 0; 0 1]+[0.5,0,0]
                println(io, "\"test M13\"")
                println(io, "1")
                println(io, "2")
                println(io, "2 -3")
                println(io, "1.0")
                # C SDP block (2×2 = 4 entries, row-major)
                println(io, "1.0 0.0 0.0 1.0")
                # C LP block (3 entries)
                println(io, "1.0 2.0 3.0")
                # A_1 SDP block (2×2 = 4 entries)
                println(io, "1.0 0.0 0.0 1.0")
                # A_1 LP block (3 entries)
                println(io, "0.5 0.0 0.0")
                close(io)
                problem = SDPA.read_dat(path, T)
                @test problem.nBlock == 2
                @test problem.blockStruct == [2, -3]
                @test problem.m == 1
                @test length(problem.b.ele) == 1
                # CRITICAL: 3 LP scalars, not 1
                @test problem.C.LP_sp_nBlock == 3
                # Compact LP storage: only nonzero entries stored.
                @test problem.C.LP_nnz == 3          # C LP = [1, 2, 3]
                @test problem.A[1].LP_nnz == 1       # A1 LP = [0.5, 0, 0]
                @test SDPA.get_lp_value(problem.C, 3) == T(3)
                @test SDPA.get_lp_value(problem.A[1], 1) == T(0.5)
                @test SDPA.get_lp_value(problem.A[1], 2) == T(0)
            end
        end
    end

    @testset "trailing token rejected" begin
        # The dense parser reads fixed counts (bs*bs per SDP block, |bs| per
        # LP block), so a mid-block truncation is caught by the token-stream
        # reader. Here we append one extra garbage token after a complete
        # file: 1 token remains unconsumed and the trailing-token guard must
        # fire instead of silently ignoring it.
        mktemp() do path, io
            # mDim=1, nBlock=1, blockStruct=[2], b=[1.0],
            # C = [1 0; 0 1], A1 = [2 0; 0 3], then one stray token.
            println(io, "1")
            println(io, "1")
            println(io, "2")
            println(io, "1.0")
            println(io, "1.0 0.0 0.0 1.0")
            println(io, "2.0 0.0 0.0 3.0")
            println(io, "99")
            close(io)
            err = try
                SDPA.read_dat(path, Float64)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin("trailing", msg)
            @test occursin("1 trailing token(s)", msg)
        end
    end

    @testset "line-strict header" begin
        mktemp() do path, io
            # Two tokens on the mDIM line.
            println(io, "1 2")
            println(io, "1")
            println(io, "2")
            println(io, "1.0")
            println(io, "1.0 0.0 0.0 1.0")
            println(io, "2.0 0.0 0.0 3.0")
            close(io)
            # "1 2" is a valid 2-token header (mDIM + nBLOCK on one
            # line), so mDIM=1 nBLOCK=2 and the single-token line 2 then
            # fails the bLOCKsTRUCT count check.
            err = try; SDPA.read_dat(path, Float64); nothing; catch ex; ex; end
            @test err isa ErrorException
            @test occursin("line 2: expected 2 token(s) for bLOCKsTRUCT, got 1", sprint(showerror, err))
        end
    end

    @testset "huge mDIM with truncated file fails fast" begin
        # Mirrors read_dat_s: a corrupt header declaring mDIM=1e9 must be
        # rejected by the token-availability guard before mDim-sized
        # storage is allocated.
        mktemp() do path, io
            println(io, "1000000000")
            println(io, "1")
            println(io, "1")
            println(io, "0.5")
            close(io)
            err = try; SDPA.read_dat(path, Float64); nothing; catch ex; ex; end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin("truncated", msg)
            @test occursin("mDIM=1000000000", msg)
            @test occursin(path, msg)
        end
    end

    @testset "token errors carry file/field context" begin
        mktempdir() do tmpdir
            # Non-integer mDIM token ("3.0"): the error must name the file
            # and the field.
            p = joinpath(tmpdir, "bad_mdim.dat")
            write(p, "3.0\n1\n2\n1.0\n1.0 0.0 0.0 1.0\n2.0 0.0 0.0 3.0\n")
            err = try; SDPA.read_dat(p, Float64); nothing; catch e; e; end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin(p, msg)
            @test occursin("mDIM", msg)

            # Truncated b vector: the token-availability guard must name
            # the b-vector range.
            p2 = joinpath(tmpdir, "short_b.dat")
            write(p2, "3\n1\n2\n1.0\n2.0\n")
            err = try; SDPA.read_dat(p2, Float64); nothing; catch e; e; end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin(p2, msg)
            @test occursin("b[", msg)
        end
    end

    @testset "degenerate headers (mDIM=0 / nBLOCK=0) are accepted" begin
        mktempdir() do tmpdir
            # (a) mDIM=0: no b vector, but C blocks are still written.
            # Symmetric C so the asymmetry warning stays silent here.
            p1 = joinpath(tmpdir, "m0.dat")
            write(p1, "0\n1\n2\n1.0 2.0\n2.0 4.0\n")
            problem = SDPA.read_dat(p1, Float64)
            @test problem.m == 0
            @test problem.nBlock == 1
            @test isempty(problem.b.ele)
            @test isempty(problem.A)
            @test problem.C.SDP_sp_block[1].de_ele[1, 2] == 2.0

            # (b) nBLOCK=0: no blockStruct line and no matrix body, only b.
            p2 = joinpath(tmpdir, "nb0.dat")
            write(p2, "1\n0\n1.0\n")
            problem = SDPA.read_dat(p2, Float64)
            @test problem.m == 1
            @test problem.nBlock == 0
            @test problem.blockStruct == Int[]
            @test problem.b.ele == [1.0]

            # (c) the conjunction mDIM=0 AND nBLOCK=0 — a fully-empty
            # problem in 2 data lines.
            p3 = joinpath(tmpdir, "empty.dat")
            write(p3, "0\n0\n")
            problem = SDPA.read_dat(p3, Float64)
            @test problem.m == 0
            @test problem.nBlock == 0
            @test isempty(problem.b.ele)

            # (d) truncations still fail descriptively.
            p4 = joinpath(tmpdir, "one_line.dat")
            write(p4, "3\n")
            err = try
                SDPA.read_dat(p4, Float64)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin(">= 2 non-comment lines", sprint(showerror, err))
        end
    end

    @testset "asymmetric dense input warns; upper triangle is kept" begin
        mktempdir() do tmpdir
            # C = [1 2; 3 4] (asymmetric: lower 3 ≠ upper 2) → one warn;
            # the parsed value at (1,2)/(2,1) must be the UPPER 2.0.
            p = joinpath(tmpdir, "asym.dat")
            write(p, "1\n1\n2\n1.0\n1.0 2.0\n3.0 4.0\n1.0 0.0\n0.0 1.0\n")
            logs, problem = Test.collect_test_logs() do
                SDPA.read_dat(p, Float64)
            end
            warns = filter(l -> occursin("asymmetric", l.message), logs)
            @test length(warns) == 1
            @test occursin(p, warns[1].message)
            @test occursin("taking the upper triangle", warns[1].message)
            # The warning reports the first violating pair + a count.
            @test any(kv -> first(kv) === :first_violation, warns[1].kwargs)
            @test any(kv -> first(kv) === :n_violations, warns[1].kwargs)
            # Semantics unchanged: upper triangle wins (mirrored on
            # densification), the discarded lower value 3.0 is absent.
            C_blk = problem.C.SDP_sp_block[1]
            @test C_blk.de_ele[1, 2] == 2.0
            @test C_blk.de_ele[2, 1] == 2.0  # mirrored upper, not the lower 3.0
            @test C_blk.de_ele[2, 2] == 4.0

            # Symmetric input stays completely silent.
            ps = joinpath(tmpdir, "sym.dat")
            write(ps, "1\n1\n2\n1.0\n1.0 2.0\n2.0 4.0\n1.0 0.0\n0.0 1.0\n")
            logs2, problem2 = Test.collect_test_logs() do
                SDPA.read_dat(ps, Float64)
            end
            @test isempty(logs2)
            @test problem2.C.SDP_sp_block[1].de_ele[1, 2] == 2.0
        end
    end
end
