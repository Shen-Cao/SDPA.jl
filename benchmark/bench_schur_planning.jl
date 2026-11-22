# Benchmark one Julia checkout per process. Setup cases exclude input parsing,
# Newton-state construction and (for pairs) skyline construction/allocation.
# Whole-solve rows are warmed API calls; there are no large-factor probes.
using SDPA, LinearAlgebra, SHA, Printf, Logging

const FIXTURES = joinpath(dirname(dirname(pathof(SDPA))), "test", "fixtures")
const REPEATS = parse(Int, get(ENV, "SDPA_PLANNING_REPEATS", "5"))
const NUMERIC = get(ENV, "SDPA_PLANNING_TYPE", "all")
const MODE = get(ENV, "SDPA_PLANNING_MODE", "all")
REPEATS > 0 || error("SDPA_PLANNING_REPEATS must be positive")
NUMERIC in ("all", "f64", "bf") || error("SDPA_PLANNING_TYPE must be all, f64, or bf")
MODE in ("all", "plan", "solves") || error("SDPA_PLANNING_MODE must be all, plan, or solves")

function synthetic(::Type{T}, sizes, n; grouped_solve=false) where T
    block_struct = fill(n, length(sizes))
    m = sum(sizes)
    inputs = Vector{SDPA.SDPASparseLinearSpace{T}}(undef, m)
    b = SDPA.SDPAVector{T}(m)
    C = SDPA.SDPASparseLinearSpace{T}(length(sizes), 0)
    for l in eachindex(sizes)
        C.SDP_sp_block[l] = SDPA.SDPASparseMatrix{T}(n, n, :sparse, n)
        for r in 1:n
            SDPA.setelement_sdp!(C, l, r, r, one(T))
        end
    end
    i = 0
    for (l, count) in pairs(sizes), r in 1:count
        i += 1
        a = SDPA.SDPASparseLinearSpace{T}(1, 0)
        a.SDP_sp_index[1] = l
        w = grouped_solve ? 2 : 1 + mod(13i + l, min(n, 11))
        a.SDP_sp_block[1] = SDPA.SDPASparseMatrix{T}(n, n, :sparse, w)
        if grouped_solve
            SDPA.setelement_sdp!(a, 1, 1, 1, one(T))
            SDPA.setelement_sdp!(a, 1, r, r, one(T))
            b.ele[i] = T(2)
        else
            for k in 1:w
                pos = mod1(i + k, n)
                SDPA.setelement_sdp!(a, 1, pos, pos, one(T))
            end
        end
        inputs[i] = a
    end
    p = SDPA.SDPAProblem{T}(m, length(sizes), block_struct, b, C, inputs)
    return SDPA.initialize_index!(p)
end

function repeated(f::F, count) where F
    local last
    for _ in 1:count
        last = f()
    end
    return last
end

function sample_rows(kind, label, ::Type{T}, setup::F, count, m) where {T,F}
    f = setup()
    repeated(f, 1)
    last = nothing
    for k in 1:REPEATS
        # Setup is untimed. For pair tables each sample starts with the same
        # empty capacities, matching one fresh solve, not a reuse benchmark.
        f = setup()
        stats = @timed repeated(f, count)
        last = stats.value
        @printf("trial\t%s\t%s\t%s\t%d\t%d\t%.10f\t%.1f\t%.10f\n",
                kind, label, string(T), m, k, stats.time/count, stats.bytes/count,
                stats.gctime/count)
    end
    return last
end

function formula_case(label, p::SDPA.SDPAProblem{T}, count) where T
    ns = SDPA.NewtonState{T}(p.m, p.blockStruct; allocate_dense_bmat=false)
    result = sample_rows("formula", label, T, () -> () -> SDPA.compute_formula!(ns, p),
                         count, p.m)
    println("formula_digest\t", label, "\t", T, "\t",
            bytes2hex(sha256(join(vec(result.useFormula), ','))))
end

function pairs_case(label, p::SDPA.SDPAProblem{T}) where T
    ordering = collect(p.m:-1:1)
    reaches = SDPA.build_skyline_from_problem!(p, ordering, ordering)
    profile = SDPA._SchurSkyline{T}(reaches)
    function setup()
        ns = SDPA.NewtonState{T}(p.m, p.blockStruct; allocate_dense_bmat=false)
        ns.ordering = ordering
        ns.reverse_ordering = ordering
        ns.sparse_bmat = profile
        return () -> SDPA.make_aggrigate_index_sdp!(ns, p, ordering)
    end
    ns = sample_rows("pair_table", label, T, setup, 1, p.m)
    digest = SHA.SHA2_256_CTX()
    for v in (ns.SDP_constraint1, ns.SDP_constraint2, ns.blockIndex1,
              ns.blockIndex2, ns.SDP_location_sparse_bMat, ns.SDP_block_table_len)
        SHA.update!(digest, reinterpret(UInt8, v))
    end
    println("pairs_digest\t", label, "\t", T, "\t", length(ns.SDP_constraint1), "\t",
            bytes2hex(SHA.digest!(digest)))
end

function solve_case(label, p::SDPA.SDPAProblem{T}, count; reverse=true) where T
    params = SDPA.Parameters{T}(precision=256)
    result = sample_rows("solve", label, T,
        () -> () -> SDPA.solve!(p, params; verbose=false, io=devnull,
                                reverse_primal_dual=reverse), count, p.m)
    io = IOBuffer()
    print(io, result.terminal_phase, ' ', result.exit_reason, ' ', result.iterations, ' ')
    for v in (result.final_obj_primal, result.final_obj_dual, result.final_mu,
              result.final_primal_res, result.final_dual_res)
        print(io, v, ' ')
    end
    for r in result.trajectory
        for field in fieldnames(typeof(r))
            print(io, getfield(r, field), ' ')
        end
    end
    for blocks in (result.xMat, result.zMat)
        for block in blocks.SDP_block, x in block.de_ele
            print(io, x, ' ')
        end
        for x in blocks.LP_block
            print(io, x, ' ')
        end
    end
    for y in result.yVec.ele
        print(io, y, ' ')
    end
    println("solve_digest\t", label, "\t", T, "\t", result.terminal_phase, "\t",
            result.exit_reason, "\t", result.iterations, "\t", bytes2hex(sha256(take!(io))))
    log_buffer = IOBuffer()
    with_logger(SimpleLogger(log_buffer, Logging.Info)) do
        SDPA.solve!(p, params; verbose=true, io=devnull, reverse_primal_dual=reverse)
    end
    for line in split(String(take!(log_buffer)), '\n')
        if occursin("Using dense Schur", line) || occursin("Block-level profile:", line) ||
           occursin("using dense path", line)
            println("route\t", label, "\t", T, "\t", strip(line))
        end
    end
end

function main_for(::Type{T}) where T
    small = SDPA.read_dat_s(joinpath(FIXTURES, "example1.dat-s"), T)
    mixed = SDPA.read_dat_s(joinpath(FIXTURES, "lp_first_mixed.dat-s"), T)
    if MODE in ("all", "plan")
        formula_case("example1", small, 5000)
        formula_case("lp_first_mixed", mixed, 3000)
        formula_case("one_group_4096", synthetic(T, (4096,), 71), 1)
        formula_case("two_groups_2556", synthetic(T, (2556, 2556), 71), 1)
        if T === Float64
            pairs_case("lp_first_mixed", mixed)
            pairs_case("one_group_1024", synthetic(T, (1024,), 39))
            pairs_case("two_groups_768", synthetic(T, (768, 768), 39))
        end
    end
    if MODE in ("all", "solves")
        solve_case("example1", small, T === Float64 ? 8 : 4)
        solve_case("lp_first_mixed", mixed, T === Float64 ? 4 : 2)
        p = synthetic(T, (18, 18, 18, 18, 18, 18), 18; grouped_solve=true)
        solve_case("sdp6_by18_default", p, 1; reverse=false)
    end
end

println("revision\t", get(ENV, "SDPA_PLANNING_REVISION", "not supplied"))
println("context\t", VERSION, "\t", Threads.nthreads(), "\t", Sys.CPU_NAME, "\t",
        pathof(SDPA), "\t", BLAS.get_config())
println("columns\tstage\tcase\ttype\tm\tsample\twall_s_per_call\tbytes_per_call\tgc_s_per_call")
BLAS.set_num_threads(1)
NUMERIC in ("all", "f64") && main_for(Float64)
if NUMERIC in ("all", "bf")
    setprecision(BigFloat, 256) do
        main_for(BigFloat)
    end
end
