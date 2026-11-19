using SDPA
using Test
using LinearAlgebra

@testset "Rblas" begin
    setprecision(BigFloat, 256) do
        @testset "Rdot" begin
            x = BigFloat.([1, 2, 3, 4])
            y = BigFloat.([10, 20, 30, 40])
            @test SDPA.Rdot(4, x, 1, y, 1) == 300  # 10+40+90+160
            @test SDPA.Rdot(0, x, 1, y, 1) == 0
            @test SDPA.Rdot(2, x, 1, y, 1) == 50  # 10+40
            @test SDPA.Rdot(2, x, 2, y, 2) == 100  # x[1]*y[1]+x[3]*y[3]=10+90
        end

        @testset "Raxpy!" begin
            x = BigFloat.([1, 2, 3, 4])
            y = BigFloat.([10, 20, 30, 40])
            SDPA.Raxpy!(4, BigFloat(2), x, 1, y, 1)
            @test y == BigFloat.([12, 24, 36, 48])
            # da=0 is no-op
            y2 = BigFloat.([10, 20, 30, 40])
            SDPA.Raxpy!(4, BigFloat(0), x, 1, y2, 1)
            @test y2 == BigFloat.([10, 20, 30, 40])
            # n=0 is no-op
            y3 = BigFloat.([10, 20, 30, 40])
            SDPA.Raxpy!(0, BigFloat(2), x, 1, y3, 1)
            @test y3 == BigFloat.([10, 20, 30, 40])
        end

        @testset "Rscal!" begin
            x = BigFloat.([1, 2, 3, 4])
            SDPA.Rscal!(4, BigFloat(3), x, 1)
            @test x == BigFloat.([3, 6, 9, 12])
            # da=1 is no-op
            x2 = BigFloat.([1, 2, 3, 4])
            SDPA.Rscal!(4, BigFloat(1), x2, 1)
            @test x2 == BigFloat.([1, 2, 3, 4])
            # Stride
            x3 = BigFloat.([1, 2, 3, 4, 5])
            SDPA.Rscal!(3, BigFloat(2), x3, 2)
            @test x3 == BigFloat.([2, 2, 6, 4, 10])
        end

        @testset "Rcopy!" begin
            x = BigFloat.([1, 2, 3, 4])
            # Rcopy! writes into the destination via mp_set! (in-place
            # value copy), so the destination must hold INDEPENDENT BigFloat
            # objects — `zeros(BigFloat, n)` shares one MPFR buffer across
            # all slots (Julia 1.12), which mp_set! would overwrite.
            y = SDPA._independent_zeros(BigFloat, 4)
            SDPA.Rcopy!(4, x, 1, y, 1)
            @test y == x
            # Partial copy with stride: x[1], x[3] → y2[1], y2[2]
            y2 = SDPA._independent_zeros(BigFloat, 4)
            SDPA.Rcopy!(2, x, 2, y2, 1)
            @test y2[1] == 1
            @test y2[2] == 3
        end

        @testset "Rcopy! BigFloat independence" begin
            # Rcopy! must VALUE-copy: `dy[i] = dx[i]` would store the SAME
            # BigFloat object in both vectors, so an in-place MPFR op on
            # dy[i] would silently corrupt dx[i].
            dx = BigFloat.([1, 2, 3])
            dy = SDPA._independent_zeros(BigFloat, 3)
            SDPA.Rcopy!(3, dx, 1, dy, 1)
            @test dy == dx
            for i in 1:3
                @test dy[i] !== dx[i]  # distinct objects
            end
            # In-place mutation of dy[1] must leave dx[1] unchanged.
            SDPA.mp_add!(dy[1], dy[1], BigFloat(1))
            @test dy[1] == 2
            @test dx[1] == 1
            @test dx[2] == 2
            @test dx[3] == 3
        end

        @testset "Rgemv!" begin
            # y = alpha * A * x + beta * y
            A = BigFloat.([1 2; 3 4])
            x = BigFloat.([1, 1])
            y = zeros(BigFloat, 2)
            SDPA.Rgemv!("N", 2, 2, one(BigFloat), A, 2, x, 1,
                        zero(BigFloat), y, 1)
            @test y == BigFloat.([3, 7])  # [1*1+2*1, 3*1+4*1]
            # Transpose
            y2 = zeros(BigFloat, 2)
            SDPA.Rgemv!("T", 2, 2, one(BigFloat), A, 2, x, 1,
                        zero(BigFloat), y2, 1)
            @test y2 == BigFloat.([4, 6])  # [1*1+3*1, 2*1+4*1]
            # With beta
            y3 = BigFloat.([10, 20])
            SDPA.Rgemv!("N", 2, 2, one(BigFloat), A, 2, x, 1,
                        BigFloat(2), y3, 1)
            @test y3 == BigFloat.([23, 47])  # [3+2*10, 7+2*20]
        end

        @testset "Rgemv! non-unit stride incx=2" begin
            # BLAS contract: x is read at x[1 + (j-1)*incx].
            m, n = 3, 4
            A = rand(BigFloat, m, n) .* BigFloat(10)
            alpha = BigFloat(1.5)
            beta = BigFloat(0.5)
            for (transa, len_in, len_out) in (("N", n, m), ("T", m, n))
                x = rand(BigFloat, 1 + (len_in - 1) * 2) .* BigFloat(10)
                x_used = x[1:2:(1 + (len_in - 1) * 2)]
                y0 = rand(BigFloat, len_out) .* BigFloat(10)
                opA = transa == "N" ? A : transpose(A)
                # Reference on a PRISTINE copy BEFORE the in-place call:
                # BigFloat `copy` shares element objects, and Rgemv!
                # beta-scales y in place.
                y_ref = SDPA._copy_independent(y0)
                mul!(y_ref, opA, x_used, alpha, beta)
                y = SDPA._copy_independent(y0)
                SDPA.Rgemv!(transa, m, n, alpha, A, m, x, 2, beta, y, 1)
                @test y ≈ y_ref
            end
        end

        @testset "Rgemv! negative stride incy=-1 non-square" begin
            # The beta-scaling (and alpha-add) start index for a reversed
            # output must be derived from the branch's output length
            # (m for "N", n for "T"), NOT max(m, n). On a non-square
            # system max() starts the reversed write at the wrong slot.
            #
            # BLAS incy=-1 convention: physical slot j holds logical output
            # len_out-j+1, so the reference is reverse(alpha*op(A)*x) added
            # to the (physically reversed) beta*y0.
            for (transa, m, n, len_in, len_out) in
                    (("N", 3, 5, 5, 3), ("T", 5, 3, 5, 3))
                A = rand(BigFloat, m, n) .* BigFloat(10)
                x = rand(BigFloat, len_in) .* BigFloat(10)
                y0 = rand(BigFloat, len_out) .* BigFloat(10)
                alpha = BigFloat(1.5)
                beta = BigFloat(0.5)
                opA = transa == "N" ? A : transpose(A)
                vals = reverse(alpha * opA * x) + beta * SDPA._copy_independent(y0)
                y = SDPA._copy_independent(y0)
                SDPA.Rgemv!(transa, m, n, alpha, A, m, x, 1, beta, y, -1)
                @test y ≈ vals
            end
        end

        @testset "Rgemv! negative incx" begin
            # BLAS convention: incx=-1 reads x from the FAR end
            # (x[n], x[n-1], ...).
            for (transa, m, n, len_in, len_out) in
                    (("N", 3, 4, 4, 3), ("T", 4, 3, 4, 3))
                A = rand(BigFloat, m, n) .* BigFloat(10)
                x = rand(BigFloat, len_in) .* BigFloat(10)
                alpha = BigFloat(1.5)
                opA = transa == "N" ? A : transpose(A)
                for beta in (BigFloat(0), BigFloat(0.5))
                    y0 = rand(BigFloat, len_out) .* BigFloat(10)
                    vals = alpha * opA * reverse(x) + beta * SDPA._copy_independent(y0)
                    y = SDPA._copy_independent(y0)
                    SDPA.Rgemv!(transa, m, n, alpha, A, m, x, -1, beta, y, 1)
                    @test y ≈ vals
                end
            end
        end

        @testset "Rgemv! dimension/stride validation" begin
            A = rand(BigFloat, 2, 2)
            x = rand(BigFloat, 2)
            y = rand(BigFloat, 2)
            @test_throws DimensionMismatch SDPA.Rgemv!(
                "N", 2, 3, BigFloat(1), A, 2, x, 1, BigFloat(0), y, 1)
            @test_throws DimensionMismatch SDPA.Rgemv!(
                "N", 2, 2, BigFloat(1), A, 2, BigFloat[1.0], 1, BigFloat(0), y, 1)
            @test_throws DimensionMismatch SDPA.Rgemv!(
                "N", 2, 2, BigFloat(1), A, 2, x, 1, BigFloat(0), BigFloat[1.0], 1)
            @test_throws ArgumentError SDPA.Rgemv!(
                "N", 2, 2, BigFloat(1), A, 2, x, 0, BigFloat(0), y, 1)
            @test_throws ArgumentError SDPA.Rgemv!(
                "N", 2, 2, BigFloat(1), A, 2, x, 1, BigFloat(0), y, 0)
        end

        @testset "Float64 dispatch" begin
            x = [1.0, 2.0, 3.0]
            y = [10.0, 20.0, 30.0]
            SDPA.Raxpy!(3, 2.0, x, 1, y, 1)
            @test y == [12.0, 24.0, 36.0]
            d = SDPA.Rdot(3, x, 1, y, 1)
            @test d ≈ 12.0*1.0 + 24.0*2.0 + 36.0*3.0
        end

        @testset "Rgemv! Float64" begin
            # y = alpha * op(A) * x + beta * y (Float64 fallback)
            A = Float64[1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15]  # 3×5
            m, n = 3, 5
            # Stride-aware reference (plain loops, BLAS convention:
            # x index = 1 + (j-1)*incx; y index = 1 + (i-1)*incy;
            # negative strides start at 1 + (len-1)*incx).
            function ref_gemv(transa, m, n, alpha, A, x, incx, beta, y, incy)
                y_ref = copy(y)
                if transa == "N"
                    iy = incy < 0 ? (1 - m) * incy + 1 : 1
                    for i in 1:m
                        s = 0.0
                        ix = incx < 0 ? (1 - n) * incx + 1 : 1
                        for j in 1:n
                            s += A[i, j] * x[ix]
                            ix += incx
                        end
                        y_ref[iy] = alpha * s + beta * y_ref[iy]
                        iy += incy
                    end
                else
                    iy = incy < 0 ? (1 - n) * incy + 1 : 1
                    for j in 1:n
                        s = 0.0
                        ix = incx < 0 ? (1 - m) * incx + 1 : 1
                        for i in 1:m
                            s += A[i, j] * x[ix]
                            ix += incx
                        end
                        y_ref[iy] = alpha * s + beta * y_ref[iy]
                        iy += incy
                    end
                end
                return y_ref
            end

            for (transa, len_out, len_in) in (("N", m, n), ("T", n, m))
                for (incx, incy) in ((1, 1), (2, 1), (1, 2), (2, 2), (1, -1), (-1, -1))
                    # Size x/y for the stride (BLAS x index = 1+(j-1)*incx;
                    # y index = 1+(i-1)*incy; negative strides start at the
                    # far end, so the vectors span 1 + (len-1)*abs(stride)).
                    x = collect(1.0:Float64(1 + (len_in - 1) * abs(incx)))
                    alpha = 2.0
                    beta = 0.5
                    y0 = collect(Float64(1 + (len_out - 1) * abs(incy)):-1.0:1.0)
                    y = copy(y0)
                    SDPA.Rgemv!(transa, m, n, alpha, A, m, x, incx, beta, y, incy)
                    y_ref = ref_gemv(transa, m, n, alpha, A, x, incx, beta, y0, incy)
                    @test y ≈ y_ref
                end
            end
        end

        @testset "Rgemv! aliasing rejection" begin
            # BLAS GEMV contract: y must not overlap x or A. Mirrors the
            # Rgemm! alias guard in test_Rgemm.jl; the in-place
            # loops silently corrupt results otherwise. mightalias also
            # catches view-aliasing, not just ===.
            A = BigFloat.([1 2; 3 4])
            x = BigFloat.([1, 1])
            @test_throws ArgumentError SDPA.Rgemv!(  # y === x
                "N", 2, 2, one(BigFloat), A, 2, x, 1, zero(BigFloat), x, 1)
            @test_throws ArgumentError SDPA.Rgemv!(  # transposed path
                "T", 2, 2, one(BigFloat), A, 2, x, 1, zero(BigFloat), x, 1)
            @test_throws ArgumentError SDPA.Rgemv!(  # y aliases A
                "N", 2, 2, one(BigFloat), A, 2, x, 1, zero(BigFloat),
                view(A, :, 1), 1)
            # View aliasing into the same parent is detected too.
            parent = rand(BigFloat, 5)
            @test_throws ArgumentError SDPA.Rgemv!(
                "N", 2, 2, one(BigFloat), A, 2, view(parent, 1:2), 1,
                zero(BigFloat), view(parent, 2:3), 1)
            # Disjoint views are NOT aliases: the call must succeed.
            y = SDPA._independent_zeros(BigFloat, 2)
            SDPA.Rgemv!("N", 2, 2, one(BigFloat), A, 2,
                        view(parent, 1:2), 1, zero(BigFloat), y, 1)
            @test y == A * view(parent, 1:2)
            # Positive control: disjoint operands give
            # y = alpha*op(A)*x + beta*y for both transa values.
            for transa in ("N", "T")
                A2 = rand(BigFloat, 3, 4) .* BigFloat(10)
                x2 = rand(BigFloat, transa == "N" ? 4 : 3) .* BigFloat(10)
                y2 = rand(BigFloat, transa == "N" ? 3 : 4) .* BigFloat(10)
                alpha = BigFloat(1.5)
                beta = BigFloat(0.5)
                opA = transa == "N" ? A2 : transpose(A2)
                # Reference on a PRISTINE copy (Rgemv! mutates y in place).
                y_ref = SDPA._copy_independent(y2)
                mul!(y_ref, opA, x2, alpha, beta)
                SDPA.Rgemv!(transa, 3, 4, alpha, A2, 3, x2, 1, beta, y2, 1)
                @test y2 ≈ y_ref
            end
            # Float64 path enforces the same contract.
            Af = Float64[1 2; 3 4]
            xf = [1.0, 1.0]
            @test_throws ArgumentError SDPA.Rgemv!(
                "N", 2, 2, 1.0, Af, 2, xf, 1, 0.0, xf, 1)
            for transa in ("N", "T")
                A2 = rand(3, 4) .* 10.0
                x2 = rand(transa == "N" ? 4 : 3) .* 10.0
                y2 = rand(transa == "N" ? 3 : 4) .* 10.0
                alpha = 1.5
                beta = 0.5
                opA = transa == "N" ? A2 : transpose(A2)
                y_ref = copy(y2)
                mul!(y_ref, opA, x2, alpha, beta)
                SDPA.Rgemv!(transa, 3, 4, alpha, A2, 3, x2, 1, beta, y2, 1)
                @test y2 ≈ y_ref
            end
        end
    end
end
