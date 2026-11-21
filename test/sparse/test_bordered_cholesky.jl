using LinearAlgebra
using SDPA
using Test

function _load_bordered_kernel!(F, A)
    n = length(F.layout.order)
    SDPA._reset_bordered!(F)
    for i in 1:n, j in i:n
        iszero(A[i, j]) || SDPA._bordered_add!(F, i, j, A[i, j])
    end
    F.state = SDPA._BORDERED_ASSEMBLED
    return F
end

function _bordered_kernel_example(::Type{T}) where {T}
    ns = [63, 64, 65, 67]
    labels = append!(reduce(vcat, [fill(g, n) for (g, n) in enumerate(ns)]), zeros(Int, 5))
    # Multiplication by 35 modulo 264 is a bijection. The physical row
    # order therefore interleaves panels, islands and nonzero border rows.
    lookup = [1 + mod(35 * i, length(labels)) for i in 0:(length(labels) - 1)]
    labels = labels[lookup]
    m = length(labels)
    A = SDPA._independent_zeros(T, m, m)
    for i in 1:m
        A[i, i] = T(8 + i % 4)
        for j in (i + 1):m
            if iszero(labels[i]) || iszero(labels[j]) || labels[i] == labels[j]
                v = T(mod(31i + 11j, 29) - 14) / T(4096)
                A[i, j] = A[j, i] = v
            end
        end
    end
    return A, labels
end

@testset "packed bordered factor and two independent RHS" begin
    setprecision(BigFloat, 256) do
        for T in (Float64, BigFloat)
            A, labels = _bordered_kernel_example(T)
            n = length(labels)
            layout = SDPA._bordered_layout(labels, 100_000)
            @test layout.sizes == [63,64,65,67]
            @test layout.border == 5
            @test layout.order != collect(1:n)
            F = SDPA._BorderedFactors{T}(layout)
            @test layout.slots == sum(D -> length(D.sp_ele), F.interiors) +
                    sum(length, F.panels) + length(F.separator.sp_ele)
            @test_throws "requires valid factors" SDPA._solve_bordered!(zeros(T,n), F, zeros(T,n))
            slot = F.interiors[1].sp_ele[1]
            _load_bordered_kernel!(F, A)
            if T === BigFloat
                @test F.interiors[1].sp_ele[1] === slot
            end
            @test SDPA._factor_bordered!(F)
            snapshot = SDPA._copy_independent(vcat(F.separator.sp_ele,
                        [D.sp_ele for D in F.interiors]..., vec.(F.panels)...))
            @test_throws "requires fresh assembly" SDPA._factor_bordered!(F)
            y = SDPA._independent_zeros(T, n)
            # Different exact patterns, both nonzero on the border and
            # all interiors. Compare to the pristine matrix in original
            # order, not the permuted factor reconstructed from the kernel.
            for want in (T.(mod.(1:n, 11) .- 3), T.(mod.(13 .* (1:n), 23) .- 9))
                b = A * want
                original_rhs = SDPA._copy_independent(b)
                @test SDPA._solve_bordered!(y, F, b)
                @test isequal(b, original_rhs)
                @test norm(y - want, Inf) <= 250 * eps(one(T)) * norm(want, Inf)
                scale = norm(b, Inf)
                err = setprecision(BigFloat, 512) do
                    norm(BigFloat.(A) * BigFloat.(y) - BigFloat.(b), Inf)
                end
                @test err <= 500 * eps(one(T)) * scale
            end
            @test snapshot == vcat(F.separator.sp_ele, [D.sp_ele for D in F.interiors]...,
                                   vec.(F.panels)...)
            fail_rhs = T.(1:n)
            fail_rhs[3] = T(Inf)
            save = SDPA._copy_independent(y)
            @test !SDPA._solve_bordered!(y, F, fail_rhs)
            @test isequal(y, save)
            @test SDPA._solve_bordered!(y, F, A * T.(1:n)) # still reusable
            @test_throws "crosses certified islands" SDPA._bordered_add!(F, layout.order[1],
                layout.order[layout.sizes[1] + 1], one(T))
        end
    end
end

@testset "nonfinite, singular, and scale-invariant rejection" begin
    for T in (Float64, BigFloat)
        layout = SDPA._bordered_layout([1,2,0], 10)
        F = SDPA._BorderedFactors{T}(layout)
        for (A, bad) in ((T[1 0 1; 0 1 1; 1 1 2], true),
                         (T[1 0 1; 0 1 1; 1 1 1], true),
                         (T[1 0 T(Inf); 0 1 0; T(Inf) 0 2], true),
                         (T[T(NaN) 0 1; 0 1 0; 1 0 2], true),
                         (T[1 0 1; 0 1 0; 1 0 T(Inf)], true))
            _load_bordered_kernel!(F, A)
            @test SDPA._factor_bordered!(F) != bad
            @test F.state === SDPA._BORDERED_FAILED
            @test_throws "requires valid factors" SDPA._solve_bordered!(
                SDPA._independent_zeros(T, 3), F, T[0,0,1])
        end
        for power in (-400, 400)
            a = ldexp(one(T), power)
            A = a .* T[4 0 1; 0 3 1; 1 1 3]
            _load_bordered_kernel!(F, A)
            @test SDPA._factor_bordered!(F)
            y = SDPA._independent_zeros(T, 3)
            @test SDPA._solve_bordered!(y, F, A * T[1,2,-3])
            @test y ≈ T[1,2,-3]
        end
    end
    F = SDPA._BorderedFactors{Float64}(SDPA._bordered_layout([1,2], 10))
    _load_bordered_kernel!(F, [floatmin() 0; 0 1])
    @test SDPA._factor_bordered!(F)
    destination = [9.0, 9.0]
    @test !SDPA._solve_bordered!(destination, F, [floatmax(), 1.0])
    @test destination == [9.0, 9.0]
end

@testset "pristine 512-bit oracle distinguishes the dyadic border pivots" begin
    setprecision(BigFloat, 256) do
        L = SDPA._bordered_layout([1,2,3,0], 10)
        F = SDPA._BorderedFactors{BigFloat}(L)
        q, u = ldexp(BigFloat(1), -128), ldexp(BigFloat(1), -255)
        w = BigFloat[1, q, q]
        g = BigFloat[0,0,0,1]
        gram = BigFloat(0)
        for a in w
            gram += a * a
        end
        @test gram == 1  # rounded, naive left-to-right sum
        for step in 0:2
            f = BigFloat(1) + step * u
            A = [Matrix{BigFloat}(I,3,3) w; w' f]
            @test f - gram == step * u
            setprecision(BigFloat, 512) do
                H = BigFloat.(A)
                v = BigFloat[-1,-q,-q,1]
                pivot = H[4,4] - sum(abs2, H[1:3,4])
                @test pivot == (step - 1) * u
                if step == 1
                    @test all(iszero, H * v)
                    @test dot(v, BigFloat.(g)) == 1 # RHS lies outside range
                    false_x = v ./ u
                    @test norm(H * false_x - g, Inf) == 1
                elseif step == 2
                    @test H * (v ./ u) == g
                end
            end
            _load_bordered_kernel!(F, A)
            @test !SDPA._factor_bordered!(F) # also declines unresolved SPD neighbor
            @test F.state === SDPA._BORDERED_FAILED
        end
    end
end
