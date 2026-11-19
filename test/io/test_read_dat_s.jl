@testset "read_dat_s" begin
    example1_path = joinpath(@__DIR__, "..", "fixtures", "example1.dat-s")

    @testset "Float64 parse" begin
        problem = SDPA.read_dat_s(example1_path, Float64)
        @test problem.m == 3
        @test problem.nBlock == 1
        @test problem.blockStruct == [2]
        @test problem.b.ele == Float64[48, -8, 20]

        # C has 2 entries: (1,1,-11) and (2,2,23). Density = 2/4 = 0.5 > 0.20,
        # so the parser auto-converts to dense form via change_to_dense!.
        C_block = problem.C.SDP_sp_block[1]
        @test C_block.type === :dense
        @test size(C_block.de_ele) == (2, 2)
        @test C_block.de_ele[1, 1] == -11.0
        @test C_block.de_ele[2, 2] == 23.0
        @test C_block.de_ele[1, 2] == 0.0
        @test C_block.de_ele[2, 1] == 0.0

        # A1 = (1,1,10) + (1,2,4): density 2/4 = 0.5 > 0.20 → dense
        A1 = problem.A[1].SDP_sp_block[1]
        @test A1.type === :dense
        @test A1.de_ele[1, 1] == 10.0
        @test A1.de_ele[1, 2] == 4.0

        # A2 = (2,2,-8): density 1/4 = 0.25 > 0.20 → dense
        A2 = problem.A[2].SDP_sp_block[1]
        @test A2.type === :dense
        @test A2.de_ele[2, 2] == -8.0

        # A3 = (1,2,-8) + (2,2,-2): density 2/4 = 0.5 > 0.20 → dense
        A3 = problem.A[3].SDP_sp_block[1]
        @test A3.type === :dense
        @test A3.de_ele[1, 2] == -8.0
        @test A3.de_ele[2, 2] == -2.0

        # No LP entries in example1.
        @test problem.C.LP_sp_nBlock == 0
        @test problem.C.LP_nnz == 0
        @test isempty(problem.C.LP_sp_index)

        # Index tables populated correctly.
        @test problem.SDP_nConstraint == [3]  # all 3 constraints touch block 1
        @test problem.SDP_constraint[1] == [1, 2, 3]
    end

    @testset "BigFloat parse preserves precision" begin
        problem = SDPA.read_dat_s(example1_path, BigFloat)
        @test typeof(problem.b.ele[1]) === BigFloat
        @test problem.b.ele[1] == BigFloat(48)
        # Non-trivial values: ensure no Float64 rounding was introduced.
        @test problem.b.ele[2] == BigFloat(-8)
        C_block = problem.C.SDP_sp_block[1]
        @test typeof(C_block.de_ele[1, 1]) === BigFloat
        @test C_block.de_ele[1, 1] == BigFloat(-11)
    end

    @testset "multi-block parse (synthetic)" begin
        # Build a small multi-block file: m=1, nBlock=2, blockStruct=[2, -1]
        # C: SDP block (1,1,=5), LP block (=11)
        # A1: SDP block (2,2,=7), LP block (=3)
        tmpdir = mktempdir()
        multi_path = joinpath(tmpdir, "multi.dat-s")
        write(
            multi_path,
            """
            "multi-block synthetic"
              1  =  mDIM
              2  =  nBLOCK
              2 -1 = bLOCKsTRUCT
            100
            0 1 1 1 5
            0 2 1 1 11
            1 1 2 2 7
            1 2 1 1 3
            """
        )
        problem = SDPA.read_dat_s(multi_path, Float64)
        @test problem.m == 1
        @test problem.nBlock == 2
        @test problem.blockStruct == [2, -1]
        @test problem.b.ele == [100.0]
        # SDP block 1 (size 2): C[1,1]=5, A1[2,2]=7
        # LP block 2 (size 1):  C=11, A1=3
        @test problem.C.SDP_sp_nBlock == 1
        @test problem.C.LP_sp_nBlock == 1
        @test problem.C.SDP_sp_index == [1]
        @test problem.C.LP_sp_index == [1]

        # Density gate: C SDP block has 1 nonzero in 4 cells = 0.25 > 0.20,
        # so it should auto-convert to dense.
        @test problem.C.SDP_sp_block[1].type === :dense
        @test problem.C.SDP_sp_block[1].de_ele[1, 1] == 5.0

        # Index tables should reflect multi-block structure.
        @test problem.SDP_nConstraint == [1, 0]  # only constraint 1 touches SDP block 1
        @test problem.LP_nConstraint == [1]   # 1 LP scalar, touched by 1 constraint
    end

    @testset "malformed entries rejected with context" begin
        tmpdir = mktempdir()
        function _assert_rejected(path, msg_pieces)
            err = try
                SDPA.read_dat_s(path, Float64)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin(path, msg)
            for piece in msg_pieces
                @test occursin(piece, msg)
            end
        end

        # (a) SDP entry with i > j on a 2x2 block.
        p_ij = joinpath(tmpdir, "sdp_igtj.dat-s")
        write(p_ij, "2\n1\n2\n1.0\n2.0\n0 1 2 1 1.0\n")
        _assert_rejected(p_ij, ["SDP entry", "i=2", "j=1", "mat=0", "block=1"])

        # (b) SDP entry with i out of range (i=5 on a 2x2 block).
        p_oob = joinpath(tmpdir, "sdp_oob.dat-s")
        write(p_oob, "2\n1\n2\n1.0\n2.0\n0 1 5 1 1.0\n")
        _assert_rejected(p_oob, ["SDP entry", "i=5", "j=1", "mat=0", "block=1"])

        # (c) LP entry with j > |bs| (LP block size 1).
        p_lp = joinpath(tmpdir, "lp_oob.dat-s")
        write(p_lp, "1\n1\n-1\n1.0\n0 1 2 2 1.0\n")
        _assert_rejected(p_lp, ["LP entry", "j=2", "mat=0", "block=1"])
    end

    @testset "truncated trailing entry rejected" begin
        # A valid final entry `1 1 2 2 7` cut to its first 3 tokens leaves 2
        # unconsumed tokens after the entry loop; the parser must not treat
        # the truncated file as valid.
        tmpdir = mktempdir()
        p = joinpath(tmpdir, "truncated.dat-s")
        write(
            p,
            """
            "truncated"
              1  =  mDIM
              2  =  nBLOCK
              2 -1 = bLOCKsTRUCT
            100
            0 1 1 1 5
            0 2 1 1 11
            1 1 2 2 7
            1 1
            """,
        )
        err = try
            SDPA.read_dat_s(p, Float64)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        msg = sprint(showerror, err)
        # The per-line entry validation reports the partial tuple with
        # its line number.
        @test occursin("whole 5-tuples", msg)
        @test occursin("line 8: 2 token(s)", msg)
    end

    @testset "line-anchored validation" begin
        mktempdir() do tmpdir
            # Two tokens on the mDIM line.
            p = joinpath(tmpdir, "hdr.dat-s")
            write(p, """
            "bad header"
            1 2 = mDIM
            1 = nBLOCK
            2 = bLOCKsTRUCT
            1.0
            0 1 1 1 5.0
            """)
            # "1 2 = mDIM" is a valid 2-token header (mDIM + nBLOCK on
            # one line), so mDIM=1 nBLOCK=2 and the single-token line 2
            # then fails the bLOCKsTRUCT count check.
            err = try; SDPA.read_dat_s(p, Float64); nothing; catch ex; ex; end
            @test err isa ErrorException
            @test occursin("line 2: expected 2 token(s) for bLOCKsTRUCT, got 1", sprint(showerror, err))

            # Wrong token count on the bLOCKsTRUCT line.
            p2 = joinpath(tmpdir, "bs.dat-s")
            write(p2, """
            "bad bs"
            1 = mDIM
            1 = nBLOCK
            2 2 = bLOCKsTRUCT
            1.0
            0 1 1 1 5.0
            """)
            err = try; SDPA.read_dat_s(p2, Float64); nothing; catch ex; ex; end
            @test err isa ErrorException
            @test occursin("line 3: expected 1 token(s)", sprint(showerror, err))

            # Entry line carrying a partial 5-tuple.
            p3 = joinpath(tmpdir, "part.dat-s")
            write(p3, """
            "partial entry"
            1 = mDIM
            1 = nBLOCK
            2 = bLOCKsTRUCT
            1.0
            0 1 1 1 5.0
            0 1
            """)
            err = try; SDPA.read_dat_s(p3, Float64); nothing; catch ex; ex; end
            @test err isa ErrorException
            @test occursin("whole 5-tuples", sprint(showerror, err))
        end
    end

    @testset "duplicate entry positions are rejected" begin
        # A repeated (mat, block, i, j) position produces path-dependent
        # results (sparse sums, dense last-wins), so the reader must fail
        # loudly instead of silently picking one semantics.
        mktemp() do tmp_path, io
            write(io, """
            "duplicate test"
            1 = mDIM
            1 = nBLOCK
            2 = bLOCKsTRUCT
            1.0
            0 1 1 1 1.0
            0 1 1 1 2.0
            """)
            close(io)
            err = try
                SDPA.read_dat_s(tmp_path, Float64)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("duplicate entry", sprint(showerror, err))
            @test occursin("i=1 j=1", sprint(showerror, err))
        end
    end

    @testset "brace-packed b vector parses" begin
        # Parity with read_dat brace stripping: brace-packed b values
        # must parse.
        mktemp() do tmp_path, io
            write(io, """
            2
            2
            2 -1
            {0.5, 0.25}
            0 1 1 1 1.0
            1 2 1 1 1.0
            """)
            close(io)
            problem = SDPA.read_dat_s(tmp_path, Float64)
            @test problem.b.ele == [0.5, 0.25]
        end
    end

    @testset "huge mDIM with truncated file fails fast" begin
        # A corrupt header declaring mDIM=1e9 with a short file must be
        # rejected by the token-availability guard before mDim-sized
        # storage is allocated, failing fast with a descriptive,
        # file-naming message.
        mktemp() do tmp_path, io
            write(io, """
            1000000000
            1
            1
            0.5
            """)
            close(io)
            err = try
                SDPA.read_dat_s(tmp_path, Float64)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin("truncated", msg)
            @test occursin("mDIM=1000000000", msg)
            @test occursin(tmp_path, msg)
        end
    end

    @testset "LP offsets identical via precomputed prefix table" begin
        # LP scalar position = (LP scalars in earlier blocks) + j.
        # Pin the VALUES across mixed SDP/LP blocks.
        mktempdir() do tmpdir
            p = joinpath(tmpdir, "lp_offsets.dat-s")
            open(p, "w") do io
                # blockStruct: SDP2, LP3, SDP3, LP174, LP4 → nLP = 181.
                # LP prefix offsets per block: [0, 0, 3, 3, 177, 181].
                println(io, "2")
                println(io, "5")
                println(io, "2 -3 3 -174 -4")
                println(io, "1.0 2.0")
                # LP entries at the first/last scalar of every LP block.
                for (blk, sz, off) in [(2, 3, 0), (4, 174, 3), (5, 4, 177)]
                    for j in (1, sz)
                        println(io, "0 $(blk) $(j) $(j) $(10.0 + off + j)")
                        println(io, "1 $(blk) $(j) $(j) $(20.0 + off + j)")
                    end
                end
                # SDP entries on the interleaved SDP blocks.
                println(io, "0 1 1 1 5.0")
                println(io, "1 1 2 2 6.0")
                println(io, "0 3 1 1 7.0")
                println(io, "1 3 2 2 8.0")
            end
            problem = SDPA.read_dat_s(p, Float64)
            @test problem.nBlock == 5
            @test problem.C.LP_sp_nBlock == 181
            # block 2 (LP, |bs|=3, offset 0): positions 1..3
            @test SDPA.get_lp_value(problem.C, 1) == 11.0
            @test SDPA.get_lp_value(problem.C, 3) == 13.0
            # block 4 (LP, |bs|=174, offset 3): positions 4..177
            @test SDPA.get_lp_value(problem.C, 4) == 14.0
            @test SDPA.get_lp_value(problem.C, 177) == 187.0
            # block 5 (LP, |bs|=4, offset 177): positions 178..181
            @test SDPA.get_lp_value(problem.C, 178) == 188.0
            @test SDPA.get_lp_value(problem.C, 181) == 191.0
            @test SDPA.get_lp_value(problem.A[1], 181) == 201.0
        end
    end

    @testset "token errors carry file/field context" begin
        mktempdir() do tmpdir
            # (a) Non-integer mDIM token ("3.0"): parse(Int, ...) fails;
            # the error must name the file and the field.
            p1 = joinpath(tmpdir, "bad_mdim.dat-s")
            write(p1, "3.0\n1\n2\n1.0\n0 1 1 1 5.0\n")
            err = try; SDPA.read_dat_s(p1, Float64); nothing; catch e; e; end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin(p1, msg)
            @test occursin("mDIM", msg)

            # (b) Truncated b vector (file ends mid-b): caught by the
            # token-availability guard, which must name the file and the
            # b-vector range.
            p2 = joinpath(tmpdir, "short_b.dat-s")
            write(p2, "3\n1\n2\n1.0\n2.0\n")  # mDIM=3, only 2 b tokens
            err = try; SDPA.read_dat_s(p2, Float64); nothing; catch e; e; end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin(p2, msg)
            @test occursin("b[", msg)
        end
    end

    @testset "degenerate headers (mDIM=0 / nBLOCK=0) are accepted" begin
        mktempdir() do tmpdir
            # (a) mDIM=0: no b-vector tokens at all; C entries may exist.
            p1 = joinpath(tmpdir, "m0.dat-s")
            write(p1, "0\n1\n2\n0 1 1 1 1.0\n")
            problem = SDPA.read_dat_s(p1, Float64)
            @test problem.m == 0
            @test problem.nBlock == 1
            @test isempty(problem.b.ele)
            @test isempty(problem.A)
            # The C entry survived; density 1/4 = 0.25 > 0.20 → dense.
            @test problem.C.SDP_sp_block[1].de_ele[1, 1] == 1.0

            # (b) nBLOCK=0: no blockStruct line, only the b vector.
            p2 = joinpath(tmpdir, "nb0.dat-s")
            write(p2, "1\n0\n1.0\n")
            problem = SDPA.read_dat_s(p2, Float64)
            @test problem.m == 1
            @test problem.nBlock == 0
            @test problem.blockStruct == Int[]
            @test problem.b.ele == [1.0]

            # (c) the conjunction mDIM=0 AND nBLOCK=0 — a fully-empty
            # problem in 2 data lines.
            p3 = joinpath(tmpdir, "empty.dat-s")
            write(p3, "0\n0\n")
            problem = SDPA.read_dat_s(p3, Float64)
            @test problem.m == 0
            @test problem.nBlock == 0
            @test isempty(problem.b.ele)
            # And it round-trips through solve! as the empty problem.
            log = SDPA.solve!(problem, SDPA.Parameters{Float64}();
                              verbose=false, reverse_primal_dual=false)
            @test log.terminal_phase == SDPA.pdOPT
            @test log.iterations == 0

            # (d) truncations still fail descriptively — a 1-line file
            # cannot carry mDIM AND nBLOCK on separate lines.
            p4 = joinpath(tmpdir, "one_line.dat-s")
            write(p4, "3\n")
            err = try
                SDPA.read_dat_s(p4, Float64)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin(">= 2 non-comment lines", sprint(showerror, err))
        end
    end

    @testset "stray b token on the completion line is rejected" begin
        # The b line carries ONE extra token after the single b value,
        # followed by lines of exact 5-tuples. The completion line is
        # exempt from the per-line %5 check, so its tokens must be
        # attributed to b: the non-5 remainder is rejected with the
        # line number.
        mktempdir() do tmpdir
            p = joinpath(tmpdir, "stray_b.dat-s")
            write(p, """
            1
            1
            2
            10.0 0
            0 1 1 1 5.0
            1 1 2 2 7.0
            """)
            err = try
                SDPA.read_dat_s(p, Float64)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin(p, msg)
            @test occursin("line 4", msg)
            @test occursin("whole 5-tuples", msg)
            @test occursin("1 remaining", msg)
        end
    end

    @testset "b ending mid-line with whole tuples after still parses" begin
        # Entries legitimately start mid-line on the b-completion line;
        # a full 5-tuple glued after the b tail must keep parsing.
        mktempdir() do tmpdir
            p = joinpath(tmpdir, "midline_b.dat-s")
            write(p, """
            1
            1
            2
            10.0 0 1 1 1 5.0
            1 1 2 2 7.0
            """)
            problem = SDPA.read_dat_s(p, Float64)
            @test problem.b.ele == [10.0]
            @test problem.C.SDP_sp_block[1].de_ele[1, 1] == 5.0
            @test problem.A[1].SDP_sp_block[1].de_ele[2, 2] == 7.0
        end
    end
end
