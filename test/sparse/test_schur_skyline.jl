using Random
using SDPA
using Test

# Source-equivalence oracle: enumerate clique edges, then propagate each
# source range to all its targets (the pre-optimization implementation).
function _skyline_pairwise_reference(m, reverse_ordering, constraint_lists...)
    reach = collect(1:m)
    for lists in constraint_lists, cs in lists
        for a in eachindex(cs), b in (a + 1):length(cs)
            i, j = minmax(reverse_ordering[cs[a]], reverse_ordering[cs[b]])
            reach[i] = max(reach[i], j)
        end
    end
    for i in eachindex(reach)
        for j in (i + 1):reach[i]
            reach[j] = max(reach[j], reach[i])
        end
    end
    return reach
end

@testset "linear skyline equals pairwise construction and closure" begin
    # Explicit separated edges: closed skyline extends columns that
    # weren't themselves in an original edge.
    @test _skyline_pairwise_reference(7, 1:7, [[1, 4], [3, 6]]) ==
        [4, 4, 6, 6, 6, 6, 7]
    rng = MersenneTwister(10259)
    for T in (Float64, BigFloat)
        # Include zero rows and singleton cases, both separate and
        # overlapping SDP/LP incidences, and many non-identity orders.
        for m in (0, 1, 2, 9, 35), repetition in 1:12
            nlp = 7
            bs = [2, -nlp, 3]
            C = SDPA.SDPASparseLinearSpace{T}(2, nlp)
            A = [SDPA.SDPASparseLinearSpace{T}(2, nlp) for _ in 1:m]
            for ai in A
                ai.SDP_sp_index .= (1, 3)
                for k in 1:2
                    if rand(rng) < 0.30
                        n = bs[ai.SDP_sp_index[k]]
                        ai.SDP_sp_block[k] = SDPA.SDPASparseMatrix{T}(n, n, :sparse, 1)
                        SDPA.setelement_sdp!(ai, k, 1, 1, one(T))
                    end
                end
                for k in 1:nlp
                    rand(rng) < 0.18 && SDPA.setelement_lp!(ai, k, one(T))
                end
            end
            p = SDPA.SDPAProblem{T}(m, length(bs), bs, SDPA.SDPAVector{T}(m), C, A)
            SDPA.initialize_index!(p)
            for perm in (collect(1:m), randperm(rng, m), collect(m:-1:1))
                rev = invperm(perm)
                actual = SDPA.build_skyline_from_problem!(p, perm, rev)
                @test actual == _skyline_pairwise_reference(
                    m, rev, p.SDP_constraint, p.LP_constraint)
            end
        end
    end
end
