@testset "BigFloatPool" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            # Basic pool operations.
            pool = SDPA.BigFloatPool{T}(;
                precision=256, max_mats=3, max_vecs=2,
                mat_size=(4, 4), vec_size=4,
            )
            @test pool.precision == 256
            @test length(pool.mats) == 3
            @test length(pool.vecs) == 2

            # Acquire a matrix.
            mat = SDPA.acquire_mat!(pool, 4, 4)
            @test size(mat) == (4, 4)
            @test all(iszero, mat)
            @test length(pool.mats) == 2  # one was taken

            # Write into it and release.
            mat[1, 1] = T(42)
            SDPA.release_mat!(pool, mat)
            @test length(pool.mats) == 3  # returned

            # Re-acquire: should be zeroed.
            mat2 = SDPA.acquire_mat!(pool, 4, 4)
            @test all(iszero, mat2)  # zeroed on acquire

            # Acquire a vector.
            vec = SDPA.acquire_vec!(pool, 4)
            @test length(vec) == 4
            @test all(iszero, vec)
            SDPA.release_vec!(pool, vec)

            # Max-bytes cap: should error when exceeded.
            pool_small = SDPA.BigFloatPool{T}(;
                max_mats=1, max_vecs=0, mat_size=(2, 2),
            )
            # Pool starts with 1 mat; acquiring it is fine.
            m1 = SDPA.acquire_mat!(pool_small, 2, 2)
            # Allocating another exceeds max_bytes.
            # The default max_bytes is min(max(4*requested, 64 MiB),
            # Sys.free_memory()÷2).
            # For a tiny pool, acquiring a huge matrix should error.
            @test_throws ErrorException SDPA.acquire_mat!(pool_small, 10000, 10000)
        end
    end

    # Every pool path must hand out matrices/vectors with INDEPENDENT
    # BigFloat elements. `zeros(BigFloat, ...)`
    # shares ONE BigFloat object across all elements (aliased Memory), which
    # corrupts in-place MPFR writes (mp_add!/mp_fma! on one element clobbers
    # the others). Only meaningful for BigFloat — Float64 is isbits, so `!==`
    # would be false for equal values; guarded to BigFloat.
    @testset "BigFloat independent elements" begin
        # Constructor pre-allocated path (pool built with mat_size/vec_size):
        # the free-list entries themselves must carry independent elements
        # (zeros(BigFloat, ...) would alias one shared BigFloat across the
        # whole container).
        pool = SDPA.BigFloatPool{BigFloat}(;
            precision=256, max_mats=1, max_vecs=1,
            mat_size=(3, 3), vec_size=3,
        )
        @test pool.mats[1][1, 1] !== pool.mats[1][1, 2]
        @test pool.vecs[1][1] !== pool.vecs[1][2]

        # Free-list hit path: acquire re-fills with independent elements.
        m = SDPA.acquire_mat!(pool, 3, 3)
        @test m[1, 1] !== m[1, 2]
        @test m[1, 1] !== m[2, 3]
        SDPA.release_mat!(pool, m)
        # Reuse path (free-list hit): F1 re-acquisition zeroes the SAME
        # independent element objects in place — independence is a
        # property of the objects (created independent; kernels mutate
        # by value or assign fresh objects), preserved by in-place zeroing.
        m2 = SDPA.acquire_mat!(pool, 3, 3)
        @test m2[1, 1] !== m2[1, 2]
        # Fresh-alloc path (size mismatch -> no free-list hit).
        m3 = SDPA.acquire_mat!(pool, 4, 4)
        @test m3[1, 1] !== m3[1, 2]
        @test m3[1, 1] !== m3[4, 4]

        v = SDPA.acquire_vec!(pool, 3)
        @test v[1] !== v[2]
        SDPA.release_vec!(pool, v)
        v2 = SDPA.acquire_vec!(pool, 3)
        @test v2[1] !== v2[2]
        v3 = SDPA.acquire_vec!(pool, 7)  # fresh-alloc path
        @test v3[1] !== v3[2]
        @test v3[1] !== v3[7]
    end

    # max_bytes accounting must be precision-aware for BigFloat.
    # sizeof(BigFloat) is only the 32-byte C struct; each element
    # also carries a ~16-byte Julia object header plus the MPFR mantissa
    # allocation (~8·⌈precision/64⌉ B), so the per-element cost is
    # 64 + 8·cld(precision, 64) = 96 B at 256-bit.
    @testset "BigFloat byte accounting is precision-aware" begin
        # The `max_bytes` constructor kwarg pins the cap exactly, so the
        # budget can be sized directly.
        pool = SDPA.BigFloatPool{BigFloat}(;
            precision=256, max_mats=0, max_vecs=0, max_bytes=400,
        )  # 4·96 = 384 fits; a second 2×2 does not

        m1 = SDPA.acquire_mat!(pool, 2, 2)
        @test size(m1) == (2, 2)
        @test pool.bytes_allocated == 384  # 4 elems × 96 B
        # 384 + 384 = 768 > 400 → must error.
        @test_throws ErrorException SDPA.acquire_mat!(pool, 2, 2)

        pool2 = SDPA.BigFloatPool{BigFloat}(;
            precision=256, max_mats=0, max_vecs=0, max_bytes=200,
        )  # 2·96 = 192 fits; a second length-2 vec does not
        v1 = SDPA.acquire_vec!(pool2, 2)
        @test length(v1) == 2
        @test pool2.bytes_allocated == 192
        # 192 + 192 = 384 > 200 → must error.
        @test_throws ErrorException SDPA.acquire_vec!(pool2, 2)

        # Float64 accounting is unchanged: sizeof(Float64) = 8.
        pool3 = SDPA.BigFloatPool{Float64}(;
            max_mats=0, max_vecs=0, max_bytes=20,
        )  # 8·2 = 16 fits; a second length-2 vec does not
        v2 = SDPA.acquire_vec!(pool3, 2)
        @test pool3.bytes_allocated == 16
        @test_throws ErrorException SDPA.acquire_vec!(pool3, 2)
    end

    # The reacquire (free-list hit) path must zero the EXISTING
    # independent element objects IN PLACE (mp_zero! per element) — not
    # replace them with fresh BigFloat(0)s (that would allocate n²
    # objects per acquire: the pool would recycle containers but not
    # objects). Values are bit-identical
    # (zero is exact at any MPFR precision).
    @testset "reacquire zeroes in place" begin
        pool = SDPA.BigFloatPool{BigFloat}(;
            precision=256, max_mats=1, max_vecs=1,
            mat_size=(3, 3), vec_size=3,
        )
        m = SDPA.acquire_mat!(pool, 3, 3)
        m[1, 1] = BigFloat(42)
        m[2, 3] = BigFloat(-7)
        SDPA.release_mat!(pool, m)
        m2 = SDPA.acquire_mat!(pool, 3, 3)
        @test all(iszero, m2)
        # Object identity: the SAME container and the SAME element
        # objects come back (zeroed by value, not replaced).
        @test m2 === m
        @test m2[1, 1] === m[1, 1]
        @test m2[3, 3] === m[3, 3]
        # In-place zeroing does not introduce aliasing.
        @test m2[1, 1] !== m2[1, 2]
        SDPA.release_mat!(pool, m2)

        # Allocation gate: after warmup, an acquire/release cycle must not
        # allocate: the in-place mp_zero! loop allocates 0. Bound is far
        # below one-object-per-element cost.
        mw = SDPA.acquire_mat!(pool, 3, 3)
        SDPA.release_mat!(pool, mw)
        allocs = @allocated begin
            mm = SDPA.acquire_mat!(pool, 3, 3)
            SDPA.release_mat!(pool, mm)
        end
        @test allocs < 160  # @allocated counts fluctuate with Julia version/GC

        # Vector path: same in-place zeroing + identity.
        v = SDPA.acquire_vec!(pool, 3)
        v[2] = BigFloat(7)
        SDPA.release_vec!(pool, v)
        v2 = SDPA.acquire_vec!(pool, 3)
        @test all(iszero, v2)
        @test v2 === v
        @test v2[1] === v[1]
        SDPA.release_vec!(pool, v2)

        # Float64 (isbits) reuse path: fill! — zero values, container
        # identity, alloc-free.
        pool64 = SDPA.BigFloatPool{Float64}(;
            max_mats=1, max_vecs=0, mat_size=(3, 3),
        )
        m64 = SDPA.acquire_mat!(pool64, 3, 3)
        m64[1, 1] = 42.0
        SDPA.release_mat!(pool64, m64)
        m64b = SDPA.acquire_mat!(pool64, 3, 3)
        @test all(iszero, m64b)
        @test m64b === m64
        SDPA.release_mat!(pool64, m64b)
        mw64 = SDPA.acquire_mat!(pool64, 3, 3)
        SDPA.release_mat!(pool64, mw64)
        allocs64 = @allocated begin
            mm64 = SDPA.acquire_mat!(pool64, 3, 3)
            SDPA.release_mat!(pool64, mm64)
        end
        @test allocs64 < 160  # @allocated counts fluctuate with Julia version/GC
    end

    # Exception safety of the checkout/checkin discipline used in
    # threaded_min_eigen_dls. The helper replicates its nested try/finally
    # structure; a simulated kernel throw between acquire and release must
    # still return BOTH mats to the free list — the next acquires hand
    # back the SAME objects (===), and the pool state (free-list length)
    # is exactly as after a clean cycle.
    @testset "exception-safe checkout/checkin" begin
        # Mirrors threaded_min_eigen_dls: second acquire + use region
        # wrapped so that a throw releases BOTH mats (outer finally
        # releases tmp even if the acquire of `out` itself throws).
        function _acquire_use_release(pool, m, n, throw_mid::Bool)
            tmp = SDPA.acquire_mat!(pool, m, n)
            try
                out = SDPA.acquire_mat!(pool, m, n)
                try
                    out[1, 1] = one(eltype(out))
                    throw_mid && error("simulated kernel failure")
                    return out[1, 1]
                finally
                    SDPA.release_mat!(pool, out)
                end
            finally
                SDPA.release_mat!(pool, tmp)
            end
        end

        for T in (Float64, BigFloat)
            pool = SDPA.BigFloatPool{T}(;
                precision=256, max_mats=2, max_vecs=0, mat_size=(2, 2),
            )
            # acquire pops from the END of the free list first.
            mat_a = pool.mats[end]
            mat_b = pool.mats[end - 1]

            # Throw mid-use: the error propagates, but both mats are
            # released on the way out.
            @test_throws ErrorException _acquire_use_release(pool, 2, 2, true)
            @test length(pool.mats) == 2  # both back in the free list
            x = SDPA.acquire_mat!(pool, 2, 2)
            y = SDPA.acquire_mat!(pool, 2, 2)
            # SAME objects handed back out (release actually ran).
            @test (x === mat_a || x === mat_b)
            @test (y === mat_a || y === mat_b)
            @test x !== y
            @test all(iszero, x) && all(iszero, y)  # zeroed on reacquire
            SDPA.release_mat!(pool, x)
            SDPA.release_mat!(pool, y)

            # Success path unchanged.
            @test _acquire_use_release(pool, 2, 2, false) == one(T)
        end
    end

    # Release guard. acquire registers the handed-out object;
    # release_mat!/release_vec! error on a double release and on a foreign
    # release (an object never acquired from this pool), leaving the free
    # list untouched in both cases.
    @testset "release guard" begin
        for T in (Float64, BigFloat)
            pool = SDPA.BigFloatPool{T}(;
                precision=256, max_mats=1, max_vecs=1,
                mat_size=(2, 2), vec_size=2,
            )
            m = SDPA.acquire_mat!(pool, 2, 2)
            v = SDPA.acquire_vec!(pool, 2)

            # Double release errors (the second time).
            SDPA.release_mat!(pool, m)
            @test_throws ErrorException SDPA.release_mat!(pool, m)
            @test length(pool.mats) == 1  # bad release did not touch the free list
            SDPA.release_vec!(pool, v)
            @test_throws ErrorException SDPA.release_vec!(pool, v)
            @test length(pool.vecs) == 1

            # Foreign release (never acquired from this pool) errors.
            foreign_m = SDPA._independent_zeros(T, 2, 2)
            @test_throws ErrorException SDPA.release_mat!(pool, foreign_m)
            @test_throws ErrorException SDPA.release_vec!(pool, T[0, 0])
            @test length(pool.mats) == 1 && length(pool.vecs) == 1

            # Legitimate re-acquire/re-release cycles still work after
            # the guarded rejections.
            m2 = SDPA.acquire_mat!(pool, 2, 2)
            SDPA.release_mat!(pool, m2)
            v2 = SDPA.acquire_vec!(pool, 2)
            SDPA.release_vec!(pool, v2)
            @test length(pool.mats) == 1 && length(pool.vecs) == 1
        end
    end

    # The max_bytes cap is applied BEFORE pre-allocation, and the
    # `max_bytes` constructor kwarg pins the cap exactly (nothing = the
    # auto behavior: 4×requested, floored at
    # _DEFAULT_POOL_BUDGET, clamped at Sys.free_memory()÷2). If the
    # pre-allocation alone would exceed the cap, the constructor errors
    # descriptively.
    @testset "constructor cap" begin
        # Explicit tiny cap with NO pre-allocation: construction succeeds
        # (requested = 0 ≤ 64); the FIRST acquire errors (a 2×2 BigFloat
        # matrix @256-bit needs 4·96 = 384 B > 64).
        pool = SDPA.BigFloatPool{BigFloat}(;
            precision=256, max_mats=0, max_vecs=0, max_bytes=64,
        )
        @test pool.max_bytes == 64
        @test_throws ErrorException SDPA.acquire_mat!(pool, 2, 2)

        # Pre-allocation alone exceeds the cap: the CONSTRUCTOR errors
        # (4 mats of 2×2 @256-bit = 4·(4·96) = 1536 B > 64).
        err = try
            SDPA.BigFloatPool{BigFloat}(;
                precision=256, max_mats=4, max_vecs=0,
                mat_size=(2, 2), max_bytes=64,
            )
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("pre-allocation", err.msg)
        @test occursin("max_bytes=64", err.msg)

        # max_bytes=nothing (default): auto behavior preserved — floored
        # at _DEFAULT_POOL_BUDGET (64 MiB), ordinary pools unaffected.
        pool_auto = SDPA.BigFloatPool{BigFloat}(;
            precision=256, max_mats=1, max_vecs=0, mat_size=(2, 2),
        )
        @test pool_auto.max_bytes >= SDPA._DEFAULT_POOL_BUDGET
    end
end
