@testset "Parameters" begin
    @testset "default preset (BigFloat = C++-GMP; Float64 = upstream double)" begin
        # BigFloat keeps the C++ SDPA-GMP presets byte-identical
        # (parity goldens); Float64/Float32 get upstream SDPA's
        # double-precision defaults (1e-7 tolerances are reachable).
        p = SDPA.Parameters{BigFloat}()
        @test p.maxIteration == 200
        @test p.precision == 200  # matches C++ SDPA-GMP default
        @test p.epsilonStar == BigFloat(1.0e-30)
        @test p.lambdaStar == BigFloat(1.0e+4)
        @test p.omegaStar == BigFloat(2.0)
        @test p.lowerBound == BigFloat(-1.0e+5)
        @test p.upperBound == BigFloat(1.0e+5)
        @test p.betaStar == BigFloat(0.1)
        @test p.betaBar == BigFloat(0.3)
        @test p.gammaStar == BigFloat(0.9)
        @test p.epsilonDash == BigFloat(1.0e-30)

        p = SDPA.Parameters{Float64}()
        @test p.maxIteration == 100
        @test p.precision == 200  # unused for Float64; kept for field completeness
        @test p.epsilonStar == 1.0e-7
        @test p.lambdaStar == 1.0e+2
        @test p.omegaStar == 2.0
        @test p.lowerBound == -1.0e+5
        @test p.upperBound == 1.0e+5
        @test p.betaStar == 0.1
        @test p.betaBar == 0.2
        @test p.gammaStar == 0.9
        @test p.epsilonDash == 1.0e-7
    end

    @testset "UNSTABLE_BUT_FAST preset (matches C++ source; double eps = 1e-7)" begin
        p = SDPA.Parameters{BigFloat}(setting=SDPA.UNSTABLE_BUT_FAST)
        @test p.maxIteration == 100
        @test p.precision == 100
        @test p.lambdaStar == BigFloat(1.0e+2)
        @test p.betaStar == BigFloat(0.01)
        @test p.betaBar == BigFloat(0.02)
        @test p.gammaStar == BigFloat(0.98)
        @test p.epsilonStar == BigFloat(1.0e-30)
        @test p.epsilonDash == BigFloat(1.0e-30)

        p = SDPA.Parameters{Float64}(setting=SDPA.UNSTABLE_BUT_FAST)
        @test p.maxIteration == 100
        @test p.precision == 100
        @test p.lambdaStar == 1.0e+2
        @test p.betaStar == 0.01
        @test p.betaBar == 0.02
        @test p.gammaStar == 0.98
        @test p.epsilonStar == 1.0e-7
        @test p.epsilonDash == 1.0e-7
    end

    @testset "STABLE_BUT_SLOW preset (matches C++ source; double eps = 1e-7)" begin
        p = SDPA.Parameters{BigFloat}(setting=SDPA.STABLE_BUT_SLOW)
        @test p.maxIteration == 1000
        @test p.precision == 300
        @test p.lambdaStar == BigFloat(1.0e+2)
        @test p.betaStar == BigFloat(0.2)
        @test p.betaBar == BigFloat(0.4)
        @test p.gammaStar == BigFloat(0.5)
        @test p.epsilonStar == BigFloat(1.0e-30)
        @test p.epsilonDash == BigFloat(1.0e-30)

        p = SDPA.Parameters{Float64}(setting=SDPA.STABLE_BUT_SLOW)
        @test p.maxIteration == 1000
        @test p.precision == 300
        @test p.lambdaStar == 1.0e+2
        @test p.betaStar == 0.2
        @test p.betaBar == 0.4
        @test p.gammaStar == 0.5
        @test p.epsilonStar == 1.0e-7
        @test p.epsilonDash == 1.0e-7
    end

    @testset "kw override preserves T" begin
        p = SDPA.Parameters{BigFloat}(epsilonStar=BigFloat("1e-200"), precision=2000)
        @test p.epsilonStar == BigFloat("1e-200")
        @test p.precision == 2000
        @test p.betaStar == BigFloat(0.1)  # default retained
        @test typeof(p.epsilonStar) == BigFloat
        @test typeof(p.betaStar) == BigFloat
    end

    @testset "validators reject bad inputs" begin
        @test_throws ArgumentError SDPA.Parameters{Float64}(betaStar=-0.1)
        @test_throws ArgumentError SDPA.Parameters{Float64}(betaStar=0.5, betaBar=0.3)  # star > bar
        @test_throws ArgumentError SDPA.Parameters{Float64}(gammaStar=1.5)
        @test_throws ArgumentError SDPA.Parameters{Float64}(epsilonStar=-1)
        @test_throws ArgumentError SDPA.Parameters{Float64}(omegaStar=0.5)
        @test_throws ArgumentError SDPA.Parameters{Float64}(maxIteration=0)
        @test_throws ArgumentError SDPA.Parameters{Float64}(lowerBound=1, upperBound=-1)
        # Non-float T must be rejected with a descriptive ArgumentError.
        @test_throws ArgumentError SDPA.Parameters{Int}()
        @test_throws ArgumentError SDPA.Parameters{Rational{Int}}()
        # Positional construction with a non-float T is caught by _validate.
        @test_throws ArgumentError SDPA.Parameters{Int}(
            200, 53, 1, 10000, 2, -100000, 100000, 0, 0, 0, 1,
        )
    end

    @testset "positional constructor validates" begin
        # The inner constructor must route positional construction
        # through _validate. maxIteration=0 must throw, not construct
        # silently.
        @test_throws Exception SDPA.Parameters{Float64}(
            0, 53, 1e-7, 1e5, 2.0, -1e5, 1e5, 0.5, 0.9, 0.9, 1e-7,
        )
        # Valid positional construction still works and preserves field order.
        p = SDPA.Parameters{Float64}(
            200, 53, 1e-7, 1e5, 2.0, -1e5, 1e5, 0.5, 0.9, 0.9, 1e-7,
        )
        @test p.maxIteration == 200
        @test p.precision == 53
        @test p.epsilonStar == 1e-7
        @test p.betaBar == 0.9
        @test p.epsilonDash == 1e-7
    end

    @testset "positional constructor round-trips all 11 fields" begin
        # Every field gets a pairwise-distinct value and is asserted
        # after positional construction, so any argument-position swap
        # (transposition) fails. Values are valid (they pass _validate).
        for T in (Float64, BigFloat)
            raw = (111, 222, 1.2e-7, 3.4e4, 2.5, -6.7e5, 8.9e5,
                   0.125, 0.375, 0.675, 9.8e-8)
            tv = (raw[1], raw[2], map(T, raw[3:end])...)
            p = SDPA.Parameters{T}(tv...)
            @test p.maxIteration == 111
            @test p.precision == 222
            @test p.epsilonStar == T(1.2e-7)
            @test p.lambdaStar == T(3.4e4)
            @test p.omegaStar == T(2.5)
            @test p.lowerBound == T(-6.7e5)
            @test p.upperBound == T(8.9e5)
            @test p.betaStar == T(0.125)
            @test p.betaBar == T(0.375)
            @test p.gammaStar == T(0.675)
            @test p.epsilonDash == T(9.8e-8)
            # T-typed args land in T-typed fields (no Float64 fallback).
            @test p.epsilonStar isa T
            @test p.lambdaStar isa T
            @test p.omegaStar isa T
            @test p.lowerBound isa T
            @test p.upperBound isa T
            @test p.betaStar isa T
            @test p.betaBar isa T
            @test p.gammaStar isa T
            @test p.epsilonDash isa T
        end
    end

    @testset "Float64 path" begin
        p = SDPA.Parameters{Float64}()
        @test typeof(p.epsilonStar) === Float64
    end

    @testset "read_parameter_file round-trip" begin
        param_path = joinpath(@__DIR__, "fixtures", "param.sdpa")
        for T in (Float64, BigFloat)
            p = SDPA.read_parameter_file(param_path, T)
            # The fixture file is the C++ 200-bit preset.
            @test p.maxIteration == 200
            @test p.precision == 200
            # Numeric literals: parse(T, "1.0e-30") gives a clean decimal
            # expansion at full BigFloat precision; T(1.0e-30) first rounds
            # to Float64 (relative error ~1e-16). Use rtol=1e-15 to allow
            # the Float64-rounding noise without accepting real value drift.
            @test isapprox(p.epsilonStar, T(1.0e-30); rtol=1e-15)
            @test p.lambdaStar == T(1.0e+4)  # exact in both paths
            @test p.omegaStar == T(2.0)       # exact
            @test isapprox(p.betaStar, T(0.1); rtol=1e-15)
            @test isapprox(p.betaBar, T(0.3); rtol=1e-15)
            @test isapprox(p.gammaStar, T(0.9); rtol=1e-15)
            @test typeof(p.epsilonStar) === T
        end
    end

    @testset "read_parameter_file blank line errors with context" begin
        # A blank/whitespace-only line must produce a descriptive error
        # naming the 1-based line number and the expected field — not a
        # raw BoundsError from split(line)[1].
        # Same 11-line format as test/fixtures/param.sdpa; line 5
        # (lowerBound) is whitespace-only.
        values = ["200", "1.0E-30", "1.0E4", "2.0", "   ",
                  "1.0E5", "0.1", "0.3", "0.9", "1.0E-30", "200"]
        mktemp() do path, io
            for v in values
                println(io, v)
            end
            close(io)
            err = try
                SDPA.read_parameter_file(path, Float64)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("line 5", err.msg)
            @test occursin("lowerBound", err.msg)
        end
        # A valid file still parses (happy path, same shape as fixture).
        mktemp() do path, io
            println(io, "200\tunsigned int maxIteration;")
            println(io, "1.0E-30\tepsilonStar")
            println(io, "1.0E4\tlambdaStar")
            println(io, "2.0\tomegaStar")
            println(io, "-1.0E5\tlowerBound")
            println(io, "1.0E5\tupperBound")
            println(io, "0.1\tbetaStar")
            println(io, "0.3\tbetaBar")
            println(io, "0.9\tgammaStar")
            println(io, "1.0E-30\tepsilonDash")
            println(io, "200\tprecision")
            close(io)
            p = SDPA.read_parameter_file(path, Float64)
            @test p.maxIteration == 200
            @test p.precision == 200
            @test p.lowerBound == -1.0e5
        end
    end
end
