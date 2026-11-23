using SDPA
using Test

# The source-row order is deliberately interleaved. Each interior has
# a distinct SDP block, and the border joins both only through LP scalars.
function _bordered_plan_fixture(::Type{T}) where {T}
    m = 5
    bs = [-3, 2, 2]
    C = SDPA.SDPASparseLinearSpace{T}(2, 3)
    b = SDPA.SDPAVector{T}(m)
    A = [SDPA.SDPASparseLinearSpace{T}(2, 3) for _ in 1:m]
    for i in 1:m
        A[i].SDP_sp_index .= [2, 3]
        i == 3 && continue
        k = isodd(i) ? 2 : 1
        A[i].SDP_sp_block[k] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 1)
        SDPA.setelement_sdp!(A[i], k, i <= 2 ? 1 : 2, i <= 2 ? 1 : 2, T(1))
    end
    for (row, lp) in ((1, 1), (2, 2), (3, 1), (3, 2), (3, 3))
        SDPA.setelement_lp!(A[row], lp, T(1))
    end
    p = SDPA.SDPAProblem{T}(m, length(bs), bs, b, C, A)
    SDPA.initialize_index!(p)
    return p
end

@testset "bordered incidence certificate and checked storage" begin
    for T in (Float64, BigFloat)
        p = _bordered_plan_fixture(T)
        plan = SDPA._certify_bordered(p)
        L = plan.layout
        @test L.sizes == [2, 2]
        @test L.border == 1
        @test L.slots == 11
        @test L.slots < SDPA._schur_triangle_slots(p.m)
        @test L.order == [2, 4, 1, 5, 3]
        @test L.inverse == invperm(L.order)
        @test L.island_of == [2, 1, 0, 1, 2]
        @test L.local_index == [1, 1, 1, 2, 2]
        @test_throws "requires 11 factor/panel slots; cap is 10" SDPA._certify_bordered(p; max_slots=10)
        @test_throws "slot cap must be positive" SDPA._certify_bordered(p; max_slots=0)
        @test SDPA._certify_bordered(p; max_slots=11).layout.order == L.order
        @test_throws "partition does not cover all rows" SDPA._certify_bordered(p; labels=[1,2,0])
        @test_throws "negative partition label" SDPA._certify_bordered(p; labels=[-1,2,0,2,1])
        @test_throws "SDP block 2 connects interior rows" SDPA._certify_bordered(
            p; labels=[2,1,0,2,1])
        labels = [2,1,0,1,2]
        owned = SDPA._certify_bordered(p; labels)
        labels .= 0
        @test owned.layout.order == [2,4,1,5,3]
        @test owned.layout.island_of == [2,1,0,1,2]

        # An LP scalar cannot be inferred from SDP-only support. A tiny
        # nonzero bridge invalidates the old partition even if the numeric
        # contribution would round to zero at a given iteration.
        SDPA.setelement_lp!(p.A[2], 1, T === Float64 ? nextfloat(0.0) : ldexp(T(1), -1000))
        @test_throws "stale LP incidence table" SDPA._certify_bordered(p)
        SDPA.initialize_index!(p)
        @test_throws "LP scalar 1 connects interior rows 1 and 2" SDPA._certify_bordered(p)
        SDPA.setelement_lp!(p.A[2], 1, T(0))
        SDPA.initialize_index!(p)
        @test SDPA._certify_bordered(p).layout.slots == 11

        SDPA.setelement_lp!(p.A[2], 1, T(Inf))
        @test_throws "non-finite LP coefficient" SDPA._certify_bordered(p)
        SDPA.setelement_lp!(p.A[2], 1, T(0))
        p.A[1].SDP_sp_block[2].sp_ele[1] = T(0)
        @test_throws "stale SDP incidence table" SDPA._certify_bordered(p)
    end
    @test_throws "border has 65 rows; limit is 64" SDPA._bordered_layout([1,2,zeros(Int,65)...],30000)
    @test_throws "expected 2 to 32" SDPA._bordered_layout(collect(1:33),100)
    @test_throws "expected 2 to 32" SDPA._bordered_layout([0,1,1,0],100)
    # The builder refuses a layout with no slot saving against the packed
    # full triangle. With the required >= 2 nonempty islands that refusal is
    # unreachable: tri(sum n_i) - sum tri(n_i) = sum_{i<j} n_i*n_j >= 1. Pin
    # the invariant it protects on every admissible partition of a small m.
    for m in 2:7
        admissible = (collect(Int, labels) for labels in Iterators.product(fill(0:3, m)...)
                      if length(unique(filter(!iszero, labels))) >= 2)
        @test all(SDPA._bordered_layout(labels, typemax(Int)).slots <
                  SDPA._schur_triangle_slots(m) for labels in admissible)
    end
end

@testset "bordered refuses cancelling mirrored SDP records" begin
    for T in (Float64, BigFloat)
        p = _bordered_plan_fixture(T)
        A = p.A[1]
        A.SDP_sp_block[2] = SDPA.SDPASparseMatrix{T}(2, 2, :sparse, 2)
        SDPA.setelement_sdp!(A, 2, 1, 2, one(T))
        SDPA.setelement_sdp!(A, 2, 2, 1, -one(T))
        SDPA.initialize_index!(p)
        @test_throws "duplicate normalized SDP coordinate in row 1" SDPA._certify_bordered(p)
    end
end
