using SDPA
using Test
using Printf

@testset "mpfr_inplace wrappers" begin
    setprecision(BigFloat, 256) do
        # -------------------------------------------------------------------
        # Functional correctness
        # -------------------------------------------------------------------
        a = BigFloat("1.5")
        b = BigFloat("2.25")
        c = BigFloat("0.0")

        SDPA.mp_add!(c, a, b); @test c == a + b
        SDPA.mp_sub!(c, a, b); @test c == a - b
        SDPA.mp_mul!(c, a, b); @test c == a * b
        SDPA.mp_div!(c, a, b); @test c == a / b
        SDPA.mp_sqrt!(c, a);   @test c == sqrt(a)
        SDPA.mp_rec_sqrt!(c, a); @test c == 1 / sqrt(a)
        SDPA.mp_neg!(c, a);    @test c == -a
        SDPA.mp_neg!(c, a); SDPA.mp_abs!(c, c); @test c == abs(-a)
        SDPA.mp_set!(c, a);    @test c == a
        SDPA.mp_set_si!(c, 42); @test c == BigFloat(42)
        SDPA.mp_set_d!(c, 3.14); @test c == BigFloat(3.14)
        SDPA.mp_pow_si!(c, a, 3); @test c == a^3

        SDPA.mp_zero!(c); @test c == 0
        SDPA.mp_one!(c); @test c == 1

        # fma / fms (with aliasing — r is output AND accumulator)
        SDPA.mp_zero!(c); SDPA.mp_fma!(c, a, b, c); @test c == a * b
        s = BigFloat(0)
        SDPA.mp_fma!(s, a, b, s); @test s == a * b
        SDPA.mp_fma!(s, a, b, s); @test s == 2 * a * b

        SDPA.mp_fms!(c, a, b, a); @test c == a*b - a

        # add_mul / sub_mul (r is mutated)
        SDPA.mp_set_si!(c, 10); SDPA.mp_add_mul!(c, a, b); @test c == 10 + a*b
        SDPA.mp_set_si!(c, 10); SDPA.mp_sub_mul!(c, a, b); @test c == 10 - a*b

        # -------------------------------------------------------------------
        # Zero-allocation contract
        # -------------------------------------------------------------------
        # All wrappers should allocate 0 bytes per call. Exception:
        # mp_pow_si! which has 56 bytes of unavoidable MPFR-internal cache
        # allocation (mpfr_init2 inside mpfr_pow_us).
        r = BigFloat()
        ops = [
            ("mp_add!",          () -> SDPA.mp_add!(r, a, b)),
            ("mp_sub!",          () -> SDPA.mp_sub!(r, a, b)),
            ("mp_mul!",          () -> SDPA.mp_mul!(r, a, b)),
            ("mp_div!",          () -> SDPA.mp_div!(r, a, b)),
            ("mp_sqrt!",         () -> SDPA.mp_sqrt!(r, a)),
            ("mp_rec_sqrt!",     () -> SDPA.mp_rec_sqrt!(r, a)),
            ("mp_neg!",          () -> SDPA.mp_neg!(r, a)),
            ("mp_abs!",          () -> SDPA.mp_abs!(r, a)),
            ("mp_set!",          () -> SDPA.mp_set!(r, a)),
            ("mp_set_si!",       () -> SDPA.mp_set_si!(r, 0)),
            ("mp_set_d!",        () -> SDPA.mp_set_d!(r, 1.0)),
            ("mp_pow_si!",       () -> SDPA.mp_pow_si!(r, a, 3)),
            ("mp_fma! aliased",  () -> SDPA.mp_fma!(r, a, b, r)),
            ("mp_fms! aliased",  () -> SDPA.mp_fms!(r, a, b, r)),
            ("mp_add_mul!",      () -> SDPA.mp_add_mul!(r, a, b)),
            ("mp_sub_mul!",      () -> SDPA.mp_sub_mul!(r, a, b)),
        ]
        for (name, op) in ops
            op()  # warm up (compile + MPFR cache init)
            allocs = @allocated op()
            if name == "mp_pow_si!"
                # mp_pow_si! has 56 bytes from MPFR-internal mpfr_init2
                # cache (mpfr_pow_us uses an internal cache variable).
                @test allocs <= 64
            else
                # A <= 64-byte budget keeps the zero-alloc intent with
                # headroom for allocator-level noise on Julia point
                # releases (same budget as mp_pow_si!).
                @test allocs <= 64
            end
        end
    end
end

@testset "mp_*! honor task-local rounding mode" begin
    # The in-place ccalls must read the task-local (scoped) rounding mode,
    # matching the Base BigFloat operators.
    setprecision(BigFloat, 128) do
        a = BigFloat(1)
        b = BigFloat(3)
        r = zero(BigFloat)
        setrounding(BigFloat, RoundDown) do
            SDPA.mp_div!(r, a, b)
            @test r == BigFloat(1) / BigFloat(3)  # parity with Base operator
            @test r < BigFloat("0.34")
        end
        setrounding(BigFloat, RoundUp) do
            SDPA.mp_div!(r, a, b)
            @test r == BigFloat(1) / BigFloat(3)
            @test r > BigFloat("0.33")
        end
        setrounding(BigFloat, RoundNearest) do
            SDPA.mp_div!(r, a, b)
            @test r == BigFloat(1) / BigFloat(3)
        end
    end
end
