# The candidate is a hypergraph heuristic, not a proof. The certificate
# verifies every coefficient incidence without forming the clique graph.
const _BORDERED_MAX_BORDER = 64
const _BORDERED_MAX_ISLANDS = 32
const _BORDERED_MAX_MODEL_BLOCKS = 64
const _BORDERED_DEFAULT_MAX_SLOTS = 16_000_000

struct _BorderedSDPEntry
    row::Int
    block_index::Int
end

struct _BorderedLPEntry
    row::Int
    coeff_index::Int
end

struct _BorderedIncidence
    sdp::Vector{Vector{_BorderedSDPEntry}} # problem block order
    lp::Vector{Vector{_BorderedLPEntry}}   # scalar order, not LP block order
end

_bordered_refuse(reason) = throw(ArgumentError("bordered Schur refused: $reason"))

# Independently recover structural supports from the *values* of A.
# Then check the initialized tables which the existing formula selector
# uses. A model modified after initialization must not be certified on
# the basis of a stale table.
function _bordered_incidence(problem::SDPAProblem)
    nlp = lp_scalar_count(problem.blockStruct)
    sdp = [_BorderedSDPEntry[] for _ in 1:problem.nBlock]
    lp = [_BorderedLPEntry[] for _ in 1:nlp]
    seen = zeros(Int, problem.nBlock)
    for i in 1:problem.m
        A = problem.A[i]
        length(A.SDP_sp_index) == length(A.SDP_sp_block) == A.SDP_sp_nBlock ||
            _bordered_refuse("invalid SDP block storage in row $i")
        active = false
        for j in 1:A.SDP_sp_nBlock
            isassigned(A.SDP_sp_block, j) || continue
            M = A.SDP_sp_block[j]
            k = A.SDP_sp_index[j]
            1 <= k <= problem.nBlock && problem.blockStruct[k] >= 2 ||
                _bordered_refuse("invalid SDP block index $k in row $i")
            n = problem.blockStruct[k]
            M.nRow == M.nCol == n ||
                _bordered_refuse("invalid SDP dimensions for block $k in row $i")
            nonzero = false
            if M.type === :dense
                size(M.de_ele) == (n, n) && M.nonZeroEffect == length(M.de_ele) ||
                    _bordered_refuse("invalid dense SDP storage for block $k in row $i")
                issymmetric(M.de_ele) ||
                    _bordered_refuse("nonsymmetric dense SDP coefficients in row $i")
                for v in M.de_ele
                    isfinite(v) || _bordered_refuse("non-finite SDP coefficient in row $i")
                    nonzero |= !iszero(v)
                end
            elseif M.type === :sparse
                q = M.nonZeroEffect
                0 <= q <= min(length(M.row_index), length(M.column_index), length(M.sp_ele)) ||
                    _bordered_refuse("invalid sparse SDP storage for block $k in row $i")
                for l in 1:q
                    1 <= M.row_index[l] <= n && 1 <= M.column_index[l] <= n ||
                        _bordered_refuse("invalid SDP coordinate in row $i")
                    v = M.sp_ele[l]
                    isfinite(v) || _bordered_refuse("non-finite SDP coefficient in row $i")
                    nonzero |= !iszero(v)
                end
            else
                _bordered_refuse("unsupported SDP storage in row $i")
            end
            if nonzero
                seen[k] != i || _bordered_refuse("duplicate SDP block $k in row $i")
                seen[k] = i
                push!(sdp[k], _BorderedSDPEntry(i, j))
                active = true
            end
        end
        A.LP_sp_nBlock == nlp &&
            length(A.LP_sp_index) == length(A.LP_sp_val) == A.LP_nnz ||
            _bordered_refuse("invalid LP storage in row $i")
        last = 0
        for l in 1:A.LP_nnz
            k = A.LP_sp_index[l]
            last < k <= nlp || _bordered_refuse("LP indices must be strictly increasing in row $i")
            last = k
            v = A.LP_sp_val[l]
            isfinite(v) || _bordered_refuse("non-finite LP coefficient in row $i")
            # Opt-in is conservative: a canonical compact LP list never
            # contains a stored zero. Require consistency with its index.
            iszero(v) && _bordered_refuse("stored zero in compact LP storage in row $i")
            push!(lp[k], _BorderedLPEntry(i, l))
            active = true
        end
        active || _bordered_refuse("empty constraint row $i")
    end
    return _BorderedIncidence(sdp, lp)
end

function _check_bordered_indices(inc::_BorderedIncidence, problem::SDPAProblem)
    nsdp, nlp = length(inc.sdp), max(1, length(inc.lp))
    length(problem.SDP_nConstraint) == length(problem.SDP_constraint) ==
        length(problem.SDP_blockIndex) == nsdp ||
        _bordered_refuse("inconsistent SDP index table dimensions")
    length(problem.LP_nConstraint) == length(problem.LP_constraint) ==
        length(problem.LP_blockIndex) == nlp ||
        _bordered_refuse("inconsistent LP index table dimensions")
    for (k, entries) in enumerate(inc.sdp)
        n = length(entries)
        n == problem.SDP_nConstraint[k] == length(problem.SDP_constraint[k]) ==
            length(problem.SDP_blockIndex[k]) ||
            _bordered_refuse("stale SDP incidence table at block $k")
        for (j, e) in enumerate(entries)
            e.row == problem.SDP_constraint[k][j] && e.block_index == problem.SDP_blockIndex[k][j] ||
                _bordered_refuse("stale SDP incidence table at block $k")
        end
    end
    for k in 1:nlp
        n = isempty(inc.lp) ? 0 : length(inc.lp[k])
        n == problem.LP_nConstraint[k] == length(problem.LP_constraint[k]) ==
            length(problem.LP_blockIndex[k]) ||
            _bordered_refuse("stale LP incidence table at scalar $k")
        for j in 1:n
            inc.lp[k][j].row == problem.LP_constraint[k][j] && problem.LP_blockIndex[k][j] == k ||
                _bordered_refuse("stale LP incidence table at scalar $k")
        end
    end
    return nothing
end

function _bordered_candidate(inc::_BorderedIncidence, m::Int)
    # Rows visiting exactly one SDP block seed islands. Multiple/no SDP
    # incidences are potential border rows, not evidence of separation.
    owner = zeros(Int, m)
    for (k, entries) in enumerate(inc.sdp), e in entries
        i = e.row
        owner[i] = iszero(owner[i]) ? k : -1
    end
    replace!(owner, -1 => 0)
    return owner
end

struct _BorderedLayout
    sizes::Vector{Int}
    border::Int
    order::Vector{Int}        # permuted position -> original row
    inverse::Vector{Int}      # original row -> permuted position
    island_of::Vector{Int}    # original row -> island, 0 for border
    local_index::Vector{Int}  # original row -> position in its island/border
    slots::Int               # interiors + full coupling panels + border
end

function _bordered_layout(labels::AbstractVector{Int}, max_slots::Int)
    Base.require_one_based_indexing(labels)
    max_slots > 0 || _bordered_refuse("the factor/panel slot cap must be positive")
    all(>=(0), labels) || _bordered_refuse("negative partition label")
    names = sort!(unique(filter(!iszero, labels)))
    ni = length(names)
    2 <= ni <= _BORDERED_MAX_ISLANDS ||
        _bordered_refuse("expected 2 to $_BORDERED_MAX_ISLANDS nonempty SDP islands; found $ni")
    group = Dict(label => k for (k, label) in enumerate(names))
    rows = [Int[] for _ in 1:(ni + 1)]
    m = length(labels)
    island_of = Vector{Int}(undef, m)
    local_index = Vector{Int}(undef, m)
    for (i, label) in enumerate(labels)
        g = iszero(label) ? ni + 1 : group[label]
        push!(rows[g], i)
        island_of[i] = iszero(label) ? 0 : g
        local_index[i] = length(rows[g])
    end
    border = length(rows[end])
    border <= _BORDERED_MAX_BORDER ||
        _bordered_refuse("border has $border rows; limit is $_BORDERED_MAX_BORDER")
    sizes = [length(rows[k]) for k in 1:ni]

    # Count the dense packed factor/panels before any numeric allocation.
    # With m native-width rows, a full triangle bounds this sum.
    slots = _schur_triangle_slots(border)
    for n in sizes
        slots = Base.checked_add(slots, _schur_triangle_slots(n))
        slots = Base.checked_add(slots, Base.widemul(n, border))
    end
    slots <= max_slots ||
        _bordered_refuse("requires $slots factor/panel slots; cap is $max_slots")
    slots < _schur_triangle_slots(m) ||
        _bordered_refuse("no slot savings versus the packed full triangle")
    order = sizehint!(Int[], m)
    for ids in rows
        append!(order, ids)
    end
    return _BorderedLayout(sizes, border, order, invperm(order),
                           island_of, local_index, Int(slots))
end

struct _BorderedPlan
    layout::_BorderedLayout
    incidence::_BorderedIncidence
end

function _check_bordered_separation(entries, layout::_BorderedLayout, cone, k)
    first_row = 0
    island = 0
    for e in entries
        g = layout.island_of[e.row]
        iszero(g) && continue
        if iszero(island)
            island, first_row = g, e.row
        elseif island != g
            _bordered_refuse("$cone $k connects interior rows $first_row and $(e.row) in different islands")
        end
    end
    return nothing
end

function _certify_bordered(
    problem::SDPAProblem; max_slots::Int=_BORDERED_DEFAULT_MAX_SLOTS,
    labels::Union{Nothing,AbstractVector{Int}}=nothing,
)
    inc = _bordered_incidence(problem)
    _check_bordered_indices(inc, problem)
    partition = labels === nothing ? _bordered_candidate(inc, problem.m) : labels
    length(partition) == problem.m || _bordered_refuse("partition does not cover all rows")
    layout = _bordered_layout(partition, max_slots)
    # Once the same component label occurs on each non-border member
    # of every hyperedge, no within-cone Schur term can cross islands.
    for (k, entries) in enumerate(inc.sdp)
        _check_bordered_separation(entries, layout, "SDP block", k)
    end
    for (k, entries) in enumerate(inc.lp)
        _check_bordered_separation(entries, layout, "LP scalar", k)
    end
    return _BorderedPlan(layout, inc)
end
