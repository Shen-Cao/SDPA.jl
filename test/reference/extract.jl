# Extract canonical scalars from a SDPA-GMP `.result` file and write a JSON
# sidecar. Materializes the test/reference/*.json artifacts.
#
# Run manually (NOT at test time):
#
#     julia --project test/reference/extract.jl
#
# The C++ binary is not a runtime dependency of the test suite; the JSON files
# are immutable artifacts pinned by git. To regenerate references, re-run the
# C++ binary per test/reference/README.md and re-run this script.
#
# JSON is written with Base-only string formatting (no JSON.jl dep) because the
# structure is a flat object of scalars.

const SCALAR_PATTERNS = (
    r"phase\.value\s*=\s*(\S+)"                          => "phase",
    r"\bIteration\s*=\s*(\d+)\s*$"m                      => "iteration",
    r"\bmu\s*=\s*(-?\d+\.\d+(?:e[+-]?\d+)?)"              => "mu",
    r"objValPrimal\s*=\s*(-?\d+\.\d+(?:e[+-]?\d+)?)"      => "objValPrimal",
    r"objValDual\s*=\s*(-?\d+\.\d+(?:e[+-]?\d+)?)"        => "objValDual",
    r"p\.feas\.error\s*=\s*(-?\d+\.\d+(?:e[+-]?\d+)?)"    => "pFeasError",
    r"d\.feas\.error\s*=\s*(-?\d+\.\d+(?:e[+-]?\d+)?)"    => "dFeasError",
)

function extract_scalars(result_path::AbstractString)
    text = read(result_path, String)
    out = Dict{String, Any}()
    for (pat, key) in SCALAR_PATTERNS
        m = match(pat, text)
        if m === nothing
            error("pattern $(pat.pattern) not found in $(result_path)")
        end
        v = m.captures[1]
        if key == "phase"
            out[key] = v
        elseif key == "iteration"
            out[key] = parse(Int, v)
        else
            out[key] = parse(Float64, v)
        end
    end
    return out
end

function write_json(path::AbstractString, d::Dict{String, Any})
    # Emit keys in sorted order so the output is deterministic.
    # pairs(Dict) has no defined iteration order (it varies across Julia
    # sessions via hash seeding), which made regenerations diff-unstable.
    # Sorting changes ONLY the key order — never values, types, or the
    # extract_scalars parsing semantics.
    keys_sorted = sort!(collect(keys(d)))
    open(path, "w") do io
        println(io, "{")
        n = length(keys_sorted)
        for (i, k) in enumerate(keys_sorted)
            v = d[k]
            comma = i < n ? "," : ""
            if v isa AbstractString
                println(io, "  \"$(k)\": \"$(v)\"$(comma)")
            elseif v isa Int
                println(io, "  \"$(k)\": $(v)$(comma)")
            else
                println(io, "  \"$(k)\": $(repr(Float64(v)))$(comma)")
            end
        end
        println(io, "}")
    end
end

const REF_DIR = @__DIR__
const PROBLEM_BASENAMES = (
    ("example1.dat-s", "example1"),
    ("arch0.dat-s",   "arch0"),
    ("control5.dat-s","control5"),
)

for (problem, base) in PROBLEM_BASENAMES
    result_path = joinpath(REF_DIR, "$(problem).result")
    json_path = joinpath(REF_DIR, "$(base).json")
    scalars = extract_scalars(result_path)
    write_json(json_path, scalars)
    println("wrote $(json_path): phase=$(scalars["phase"]) iter=$(scalars["iteration"]) objP=$(scalars["objValPrimal"])")
end
