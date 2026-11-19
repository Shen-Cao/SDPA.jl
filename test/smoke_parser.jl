@testset "smoke_parser" begin
    fixture = joinpath(@__DIR__, "fixtures", "example1.dat-s")
    header = SDPA.parse_dat_s_header(fixture)
    @test header.mDim == 3
    @test header.nBlock == 1
    @test header.blockStruct == [2]
    @test startswith(header.comment, "\"Example 1")

    @testset "plain SDPLIB layout parses" begin
        tmpdir = mktempdir()
        p = joinpath(tmpdir, "plain.dat-s")
        # SDPLIB layout: no comment line — mDIM, nBLOCK, bLOCKsTRUCT are
        # three bare lines, then the b vector and entries (unread here).
        write(p, "2\n1\n2\n1.0 2.0\n0 1 1 1 5.0\n1 1 2 2 7.0\n")
        h = SDPA.parse_dat_s_header(p)
        @test h.mDim == 2
        @test h.nBlock == 1
        @test h.blockStruct == [2]
        @test h.comment == ""
    end

    @testset "plain layout bLOCKsTRUCT count mismatch errors" begin
        tmpdir = mktempdir()
        p = joinpath(tmpdir, "bs_count.dat-s")
        # nBLOCK=1 but the bLOCKsTRUCT line carries 2 tokens.
        write(p, "2\n1\n2 -1\n1.0 2.0\n")
        err = try
            SDPA.parse_dat_s_header(p)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        msg = sprint(showerror, err)
        @test occursin(p, msg)
        @test occursin("bLOCKsTRUCT", msg)
        @test occursin("expected 1", msg)
        @test occursin("got 2", msg)
    end

    @testset "short file errors as too short" begin
        tmpdir = mktempdir()
        p = joinpath(tmpdir, "short.dat-s")
        write(p, "3\n1\n")
        err = try
            SDPA.parse_dat_s_header(p)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        msg = sprint(showerror, err)
        @test occursin(p, msg)
        @test occursin("too short", msg)
    end
end
