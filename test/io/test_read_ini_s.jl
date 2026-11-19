# read_ini_s: initial-point (.ini-s) reader tests.
#
# Format under test (mirrors the C++ IO::read initial-point reader,
# sdpa_io.cpp:470-560, with the benchmark-exporter layout):
#   line 1:  yVec (m tokens, stored NEGATED)
#   then:    Z entries, then X entries, each `l i j value` with
#            l = matrix block label (1..nBlock for Z, nBlock+1..2nBlock
#            for X; problem-block = (l-1) % nBlock + 1).

using SDPA
using Test
using Logging

const _FIXTURE_DIR = joinpath(dirname(@__DIR__), "fixtures")

function _tiny_problem(T)
    m = 2
    blockStruct = [2, -1]  # SDP block first, then LP block
    b = SDPA.SDPAVector{T}([T(1), T(4)])
    C = SDPA.SDPASparseLinearSpace{T}(1, 1)
    C.SDP_sp_index = [1]
    C.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
    SDPA.setelement_sdp!(C, 1, 1, 1, T(5))
    SDPA.setelement_lp!(C, 1, T(11))
    A1 = SDPA.SDPASparseLinearSpace{T}(1, 1)
    A1.SDP_sp_index = [1]
    A1.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
    SDPA.setelement_sdp!(A1, 1, 1, 1, T(1))
    SDPA.setelement_lp!(A1, 1, T(0))
    A2 = SDPA.SDPASparseLinearSpace{T}(1, 1)
    A2.SDP_sp_index = [1]
    A2.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
    SDPA.setelement_sdp!(A2, 1, 2, 2, T(2))
    SDPA.setelement_lp!(A2, 1, T(3))
    problem = SDPA.SDPAProblem{T}(m, 2, blockStruct, b, C, [A1, A2])
    SDPA.initialize_index!(problem)
    return problem
end

function _write_ini(path, text)
    open(path, "w") do io
        write(io, text)
    end
end

@testset "read_ini_s" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            problem = _tiny_problem(T)
            ini = joinpath(mktempdir(), "tiny.ini-s")
            # blockStruct [2, -1]: problem block 1 = SDP(2x2), block 2 = LP.
            # Z: SDP[1,1]=5, SDP[1,2]=-1, SDP[2,2]=7; LP=3
            # X: SDP[1,1]=4; LP=2
            # y: (1.5, 2.5) -> stored negated
            _write_ini(ini, """
            1.5, 2.5
            1 1 1 5.0
            1 1 2 -1.0
            1 2 2 7.0
            2 1 1 3.0
            3 1 1 4.0
            3 2 2 9.0
            4 1 1 2.0
            """)
            sol = SDPA.read_ini_s(ini, problem)

            # yVec negated
            @test sol.yVec.ele[1] == -T(1.5)
            @test sol.yVec.ele[2] == -T(2.5)

            # Z: SDP block (problem block 1 -> SDP position 1), symmetric fill
            @test sol.zMat.SDP_block[1].de_ele[1, 1] == T(5)
            @test sol.zMat.SDP_block[1].de_ele[1, 2] == -T(1)
            @test sol.zMat.SDP_block[1].de_ele[2, 1] == -T(1)  # symmetric
            @test sol.zMat.SDP_block[1].de_ele[2, 2] == T(7)
            @test sol.zMat.LP_block[1] == T(3)

            # X: SDP[1,1]=4 (off-diagonals zero), LP=2
            @test sol.xMat.SDP_block[1].de_ele[1, 1] == T(4)
            @test sol.xMat.SDP_block[1].de_ele[1, 2] == T(0)
            @test sol.xMat.SDP_block[1].de_ele[2, 1] == T(0)
            @test sol.xMat.SDP_block[1].de_ele[2, 2] == T(9)
            @test sol.xMat.LP_block[1] == T(2)

            # Inverses computed: X SDP = diag(4,9) -> invCholeskyX =
            # diag(1/2, 1/3); Z SDP = [[5,-1],[-1,7]] -> invCholeskyZ
            # lower-triangular with [2,2] = sqrt(5/34), Z⁻¹ = [[7,1],[1,5]]/34
            @test sol.invCholeskyX.SDP_block[1].de_ele[1, 1] == inv(sqrt(T(4)))
            @test sol.invCholeskyX.SDP_block[1].de_ele[2, 2] == inv(sqrt(T(9)))
            @test sol.invCholeskyZ.SDP_block[1].de_ele[1, 1] ≈ inv(sqrt(T(5)))
            @test sol.invCholeskyZ.SDP_block[1].de_ele[2, 2] ≈ sqrt(T(5) / T(34))
            @test sol.invzMat.SDP_block[1].de_ele[1, 1] ≈ T(7) / T(34)
            @test sol.invzMat.SDP_block[1].de_ele[2, 2] ≈ T(5) / T(34)
            @test sol.invzMat.SDP_block[1].de_ele[1, 2] ≈ T(1) / T(34)
            @test sol.invCholeskyX.LP_block[1] ≈ inv(sqrt(T(2)))
            @test sol.invzMat.LP_block[1] == inv(T(3))

            # mu = <X,Z>/nDim (nDim = |2| + |-1| = 3)
            # <X,Z> = tr(Xs*Zs) + Xlp*Zlp = (4*5 + 9*7) + 2*3 = 89
            mu = SDPA._frobenius_inner_product_dls(sol.xMat, sol.zMat) / sol.nDim
            @test mu ≈ T(89) / T(3)

            # Regularization: negative LP Z entry clipped to 1e-6. The
            # clip warns once, naming the file and the count
            # (the point is otherwise PD, so no z-shift warning fires).
            _write_ini(ini, """
            1.5, 2.5
            1 1 1 5.0
            1 2 2 7.0
            2 1 1 -1.0
            3 1 1 4.0
            3 2 2 9.0
            4 1 1 2.0
            """)
            logs, sol2 = Test.collect_test_logs() do
                SDPA.read_ini_s(ini, problem)
            end
            @test sol2.zMat.LP_block[1] == T(1e-6)
            reg_warnings = filter(l -> l.level == Logging.Warn, logs)
            @test length(reg_warnings) == 1
            @test occursin("regularized", reg_warnings[1].message)
            @test occursin("n_lp_floored=1", reg_warnings[1].message)
            @test occursin(ini, reg_warnings[1].message)
        end
    end

    @testset "BigFloat warm-start off-diagonal mirror independent" begin
        T = BigFloat
        problem = _tiny_problem(T)
        ini = joinpath(mktempdir(), "tiny.ini-s")
        # Same warm start as above: Z SDP block has off-diagonal entry
        # (1,2) = -1 that is mirrored to (2,1).
        _write_ini(ini, """
        1.5, 2.5
        1 1 1 5.0
        1 1 2 -1.0
        1 2 2 7.0
        2 1 1 3.0
        3 1 1 4.0
        3 2 2 9.0
        4 1 1 2.0
        """)
        sol = SDPA.read_ini_s(ini, problem)
        blk = sol.zMat.SDP_block[1].de_ele
        @test blk[1, 2] == -T(1)
        @test blk[2, 1] == -T(1)
        # The transposed mirror must be a distinct BigFloat object (an
        # in-place MPFR op on one must not corrupt the other).
        @test blk[1, 2] !== blk[2, 1]
        # In-place mutation of one mirror leaves the other unchanged.
        SDPA.mp_add!(blk[1, 2], blk[1, 2], T(2))
        @test blk[1, 2] == T(1)
        @test blk[2, 1] == -T(1)
    end

    @testset "C++ 5-token layout" begin
        for T in (Float64, BigFloat)
            problem = _tiny_problem(T)
            ini = joinpath(mktempdir(), "cpp.ini-s")
            # C++ sdpa_io.cpp layout: target l i j value, target 1 → zMat,
            # 2 → xMat, l in 1..nBlock. Same point as the exporter test.
            _write_ini(ini, """
            1.5, 2.5
            1 1 1 1 5.0
            1 1 1 2 -1.0
            1 1 2 2 7.0
            1 2 1 1 3.0
            2 1 1 1 4.0
            2 1 2 2 9.0
            2 2 1 1 2.0
            """)
            sol = SDPA.read_ini_s(ini, problem)
            @test sol.yVec.ele[1] == -T(1.5)
            @test sol.yVec.ele[2] == -T(2.5)
            @test sol.zMat.SDP_block[1].de_ele[1, 1] == T(5)
            @test sol.zMat.SDP_block[1].de_ele[1, 2] == -T(1)
            @test sol.zMat.SDP_block[1].de_ele[2, 2] == T(7)
            @test sol.zMat.LP_block[1] == T(3)
            @test sol.xMat.SDP_block[1].de_ele[1, 1] == T(4)
            @test sol.xMat.SDP_block[1].de_ele[2, 2] == T(9)
            @test sol.xMat.LP_block[1] == T(2)
        end
    end

    @testset "duplicate entries warn once, last value wins" begin
        for T in (Float64, BigFloat)
            problem = _tiny_problem(T)
            ini = joinpath(mktempdir(), "dup.ini-s")
            # (1,1,1) written twice into Z: 5.0 then 6.0 — the last
            # value must win and exactly one warning must fire.
            _write_ini(ini, """
            1.5, 2.5
            1 1 1 5.0
            1 1 1 6.0
            1 2 2 7.0
            2 1 1 3.0
            3 1 1 4.0
            3 2 2 9.0
            4 1 1 2.0
            """)
            logs, sol = Test.collect_test_logs() do
                SDPA.read_ini_s(ini, problem)
            end
            @test sol.zMat.SDP_block[1].de_ele[1, 1] == T(6)
            dup_warnings = filter(l -> l.level == Logging.Warn, logs)
            @test length(dup_warnings) == 1
            @test occursin("duplicate", dup_warnings[1].message)
            @test occursin("1 duplicate", dup_warnings[1].message)
            @test occursin("last value wins", dup_warnings[1].message)
            @test occursin(ini, dup_warnings[1].message)
            # No regularization was needed for this point.
            @test !any(l -> occursin("regularized", l.message), logs)
        end
    end

    @testset "malformed entries rejected with context" begin
        for T in (Float64, BigFloat)
            problem = _tiny_problem(T)
            ini = joinpath(mktempdir(), "bad.ini-s")

            # SDP entry out of range (i=5 on a 2x2 SDP block).
            _write_ini(ini, """
            1.5, 2.5
            1 1 1 5.0
            1 5 1 9.0
            """)
            err = try
                SDPA.read_ini_s(ini, problem)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin(ini, msg)
            @test occursin("SDP entry", msg)
            @test occursin("i=5", msg)
            @test occursin("j=1", msg)
            @test occursin("l=1", msg)

            # SDP entry with i > j on a 2x2 SDP block.
            _write_ini(ini, """
            1.5, 2.5
            1 1 1 5.0
            1 2 1 9.0
            """)
            err = try
                SDPA.read_ini_s(ini, problem)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin(ini, msg)
            @test occursin("SDP entry", msg)
            @test occursin("i=2", msg)
            @test occursin("j=1", msg)
            @test occursin("l=1", msg)

            # LP entry with i=0 on the 1-scalar LP block (block 2) must
            # be rejected with a descriptive error naming the field —
            # not a raw BoundsError or a silent write to the wrong slot.
            _write_ini(ini, """
            1.5, 2.5
            1 1 1 5.0
            2 0 0 3.0
            """)
            err = try
                SDPA.read_ini_s(ini, problem)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin(ini, msg)
            @test occursin("LP entry", msg)
            @test occursin("i=0", msg)
            @test occursin("l=2", msg)

            # LP entry with i > |bs| (i=2 on the 1-scalar LP block).
            _write_ini(ini, """
            1.5, 2.5
            1 1 1 5.0
            2 2 2 3.0
            """)
            err = try
                SDPA.read_ini_s(ini, problem)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin(ini, msg)
            @test occursin("LP entry", msg)
            @test occursin("i=2", msg)
            @test occursin("l=2", msg)
        end
    end
end

@testset "y-only ini-s fails fast naming X" begin
    # An ini-s with only the y line leaves X = 0. X is never regularized
    # (only Z is), so this must fail immediately with an error naming X —
    # not after five futile εI shifts of Z with a misleading z_shift warn.
    for T in (Float64, BigFloat)
        problem = _tiny_problem(T)
        ini = joinpath(mktempdir(), "y_only.ini-s")
        _write_ini(ini, "1.5, 2.5\n")
        err = try
            SDPA.read_ini_s(ini, problem)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("X", err.msg)
        @test occursin("not positive definite", err.msg)
    end
end

@testset "x-only ini-s (Z = 0) is regularized and succeeds" begin
    # Locks the asymmetry: Z = 0 IS salvageable (LP floor + εI shift on
    # Z only), X = 0 is not.
    for T in (Float64, BigFloat)
        problem = _tiny_problem(T)
        ini = joinpath(mktempdir(), "x_only.ini-s")
        _write_ini(ini, "0.5, 0.5\n3 1 1 4.0\n3 2 2 9.0\n4 1 1 2.0\n")
        sol = SDPA.read_ini_s(ini, problem)
        @test sol.xMat.SDP_block[1].de_ele[1, 1] == T(4)
        @test sol.xMat.SDP_block[1].de_ele[2, 2] == T(9)
        # Z was εI-shifted to a strictly PD diagonal.
        @test sol.zMat.SDP_block[1].de_ele[1, 1] > 0
        @test sol.zMat.SDP_block[1].de_ele[2, 2] > 0
        @test sol.zMat.LP_block[1] > 0
    end
end
