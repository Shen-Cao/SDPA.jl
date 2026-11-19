"""
Result writer matching the C++ SDPA `.result` schema.

Mirrors C++ `IO::printLastInfo` at `sdpa_io.cpp:996-1185`
plus `printHeader` (around `:730-760`) and `printOneIteration`
(at `:781-820`).

Uses T-aware printf formatting: BigFloat gets `%+50.40e` (40 significant
digits), Float64 gets `%+.16e` (16 digits). The iteration table uses
fixed short format for column alignment regardless of T.

## Intentional deviations from the C++ SDPA `.result` layout

The following differences from the C++ `sdpa_gmp` output are deliberate
(correct/efficient Julia layout) —
they are design choices, not drift, and are kept stable:

1. Section names `xMat`/`yVec`/`zMat` vs C++ `xVec`/`xMat`/`yMat` —
   chosen Julia-side names matching the solver's own variables.
2. Summary scalars use the T-aware `%+50.40e`-style format (uniform with
   the matrix entries) vs C++'s 16-digit scalar format.
3. Banner reads `SDPA start at` vs C++'s `SDPA-GMP start at`.
4. Brace spacing in vector printing differs from C++.
5. C++ echoes the problem file's leading comment line (e.g.
   `"Example 1: mDim = 3, ..."` from the .dat-s header) immediately
   after the start banner; the Julia writer omits the comment echo.

Only the `** Parameters **` block is byte-compatible with the C++
reference `.result` files; the rest of the layout is canonical
Julia-side and is not chased to C++ byte-parity.
"""

# T-aware scalar format for summary/data values.
# BigFloat needs ~40 digits; Float64 needs ~16.
_scalar_fmt(::Type{BigFloat}) = "%+50.40e"
_scalar_fmt(::Type{<:AbstractFloat}) = "%+.16e"

# `eps(BigFloat)` depends on the GLOBAL precision at call time, so a
# 200-bit solve whose `.result` is written afterwards (at the ambient
# 256-bit default) would print 2^-255-ish instead of 2^-199. Thread the
# solve precision through so the printed machine epsilon matches the
# precision that produced the numbers. `write_result` derives its
# default from `params.precision` (BigFloat only); a `prec` of
# `nothing` from direct helper callers falls back to the ambient
# `precision(BigFloat)` at call time. Float64/Float32 eps is type-level
# and unaffected.
function _eps_at_precision(::Type{BigFloat}, prec::Union{Nothing,Integer})
    p = prec === nothing ? Base.precision(BigFloat) : Int(prec)
    return setprecision(BigFloat, p) do
        eps(BigFloat)
    end
end

_eps_at_precision(::Type{T}, ::Union{Nothing,Integer}) where {T<:Real} = eps(T)

"""
    IterationRow

One row of the per-iteration table. Fields correspond 1:1 to the C++
row format `iter mu thetaP thetaD objP objD alphaP alphaD beta`.
"""
struct IterationRow{T<:Real}
    iter::Int
    mu::T
    thetaP::T
    thetaD::T
    objP::T
    objD::T
    alphaP::T
    alphaD::T
    beta::T
end

"""
    write_result(io, problem, params, trajectory, phase, iteration, mu,
                 relgap, gap, digits, obj_p, obj_d, p_feas, d_feas,
                 compute_time; kwargs...)

Write a `.result`-schema file to `io`. See module docstring for the
schema. Mirrors C++ `IO::printLastInfo`.

Returns `nothing`. The caller is responsible for opening/closing `io`.

Keyword arguments:
- `data_filename`, `param_filename`, `out_filename`: header strings
  (default `nothing` = omit).
- `sparse_mode=false`: print "SPARSE computation" instead of "DENSE".
- `precision=nothing`: bit precision the solve ran at (BigFloat only).
  The `relative eps` line reports `eps(BigFloat)` AT THIS precision;
  `eps(BigFloat)` alone depends on the global precision at call time,
  so a 200-bit solve whose .result is written later at the ambient
  256-bit default would otherwise print 2^-255-ish instead of 2^-199.
  The default (`nothing`) derives from `params.precision` when
  `T == BigFloat`; an explicit kwarg
  overrides it. Float64/Float32 are type-level and unaffected.
- `xMat`, `yVec`, `zMat`: final solution iterates to dump (default
  `nothing` = omit the corresponding section).
- `exit_reason`: `ExitReason` enum value printed as an
  `exit reason = <NAME>` line under `phase.value` — why the IPM loop
  stopped when the phase alone is ambiguous (maxIteration / Cholesky
  failure / cannot-move / time limit). Default `nothing` = omit the
  line (legacy output shape).

The `phase` argument is a `Symbol` (e.g. `:pdOPT`, `:noINFO`) — Phase
II's `PhaseValue` enum stringifies directly to a `Symbol`.
"""
function write_result(
    io::IO,
    problem::SDPAProblem{T},
    params::Parameters{T},
    trajectory::AbstractVector{IterationRow{T}},
    phase::Symbol,
    iteration::Int,
    mu::T,
    relgap::T,
    gap::T,
    digits::T,
    obj_p::T,
    obj_d::T,
    p_feas::T,
    d_feas::T,
    compute_time::Real;
    data_filename::Union{Nothing,AbstractString}=nothing,
    param_filename::Union{Nothing,AbstractString}=nothing,
    out_filename::Union{Nothing,AbstractString}=nothing,
    sparse_mode::Bool=false,
    precision::Union{Nothing,Integer}=nothing,
    xMat::Union{Nothing,SDPADenseLinearSpace{T}}=nothing,
    yVec::Union{Nothing,SDPAVector{T}}=nothing,
    zMat::Union{Nothing,SDPADenseLinearSpace{T}}=nothing,
    # Untyped: this file is included BEFORE solve.jl defines ExitReason.
    # Pass the SolutionLog's exit_reason field (or nothing = omit).
    exit_reason=nothing,
) where {T<:Real}
    println(io, "SDPA start at ", _now_string())
    data_filename !== nothing && println(io, "data      is ", data_filename)
    param_filename !== nothing && println(io, "parameter is ", param_filename)
    out_filename !== nothing && println(io, "out       is ", out_filename)
    println(io, sparse_mode ? "SPARSE computation" : "DENSE computation")
    println(io)

    println(io, "   mu      thetaP  thetaD  objP      objD      alphaP  alphaD  beta ")
    for row in trajectory
        print_one_iteration(io, row)
    end
    println(io)

    println(io, "phase.value = ", phase, "     ")
    # Why the loop stopped (enum prints as its identifier, e.g.
    # MAX_ITERATION). Omitted when the caller passes nothing (default),
    # preserving the legacy summary shape.
    if exit_reason !== nothing
        println(io, "exit reason = ", exit_reason)
    end
    println(io, "   Iteration = ", iteration)
    fmt = _scalar_fmt(T)
    @printf(io, "          mu = %s\n"  , Printf.format(Printf.Format(fmt), mu))
    @printf(io, "relative gap = %s\n"  , Printf.format(Printf.Format(fmt), relgap))
    @printf(io, "         gap = %s\n"  , Printf.format(Printf.Format(fmt), gap))
    @printf(io, "      digits = %s\n"  , Printf.format(Printf.Format(fmt), digits))
    @printf(io, "objValPrimal = %s\n"  , Printf.format(Printf.Format(fmt), obj_p))
    @printf(io, "objValDual   = %s\n"  , Printf.format(Printf.Format(fmt), obj_d))
    @printf(io, "p.feas.error = %s\n"  , Printf.format(Printf.Format(fmt), p_feas))
    @printf(io, "d.feas.error = %s\n"  , Printf.format(Printf.Format(fmt), d_feas))
    # Resolve the precision default from `params` for
    # BigFloat — the ambient fallback printed the write-time eps after a
    # mismatched-precision solve. An explicit kwarg still overrides.
    prec = precision === nothing && T == BigFloat ? params.precision : precision
    @printf(io, "relative eps = %s\n"  , Printf.format(Printf.Format(fmt), _eps_at_precision(T, prec)))
    println(io, "total time   = ", _format_compute_time(compute_time))

    println(io)
    println(io)
    # "Parameters are" is printed exactly once, by `_print_parameters`
    # (adjacent to the `** Parameters **` block it introduces), matching
    # the C++ reference which also prints it once.
    _print_parameters(io, params)

    if xMat !== nothing
        println(io, "xMat = ")
        _print_dense_linear_space(io, xMat, problem.blockStruct)
    end
    if yVec !== nothing
        println(io, "yVec = ")
        _print_vector(io, yVec)
    end
    if zMat !== nothing
        println(io, "zMat = ")
        _print_dense_linear_space(io, zMat, problem.blockStruct)
    end

    return nothing
end

function _now_string()
    return string(Dates.format(Dates.now(), "e u d H:M:S Y"))
end

function _format_compute_time(t::Real)
    return Printf.format(Printf.Format("%.3f"), Float64(t))
end

# ---------------------------------------------------------------------------
# SolveInfo-based adapter signatures
# ---------------------------------------------------------------------------
# These wrap the detailed canonical signatures (above) so callers that
# already hold a `SolveInfo` (with adapter fields populated by
# `SolveInfo.update!`) can render a row without re-threading every value.
#
# The detailed signatures remain the canonical internal API; the adapters
# are convenience front-ends for callers that already hold a SolveInfo.

"""
    write_result(io, problem, solution, params, phase, solve_info, compute_time, iteration; kwargs...)

SolveInfo-based adapter. Constructs a one-row
trajectory from `solve_info`'s adapter fields and delegates to the
detailed `write_result` signature.

`solution` is reserved for future use (the adapter currently sources
objective values from `solve_info`); pass `nothing` if not yet plumbed.
"""
function write_result(
    io::IO,
    problem::SDPAProblem{T},
    solution::Union{SDPASolution{T},Nothing},
    params::Parameters{T},
    phase::Symbol,
    solve_info::SolveInfo{T},
    compute_time::Real,
    iteration::Int;
    kwargs...,
) where {T<:Real}
    row = IterationRow{T}(
        iteration,
        solve_info.mu_current,
        solve_info.theta_primal,
        solve_info.theta_dual,
        solve_info.objValPrimal,
        solve_info.objValDual,
        solve_info.alpha_primal,
        solve_info.alpha_dual,
        solve_info.beta_value,
    )
    return write_result(
        io, problem, params, [row], phase, iteration,
        solve_info.mu_current, solve_info.rel_gap, solve_info.gap,
        solve_info.digits, solve_info.objValPrimal, solve_info.objValDual,
        solve_info.primal_res, solve_info.dual_res, compute_time;
        kwargs...,
    )
end

"""
    print_one_iteration(io, iter, mu, thetaP, thetaD, solve_info, alpha, beta)

SolveInfo-based adapter. Constructs an `IterationRow`
from scalars + `solve_info`'s adapter fields and delegates to the
detailed `print_one_iteration` signature.

`thetaP`/`thetaD` are the primal and dual infeasibility ratios, printed
into the thetaP and thetaD columns respectively. `alpha` must expose
`.primal` and `.dual` (e.g. a StepLength-like struct); `beta` must
expose `.value` (e.g. a Beta-like struct).
"""
function print_one_iteration(
    io::IO,
    iter::Int,
    mu::T,
    thetaP::T,
    thetaD::T,
    solve_info::SolveInfo{T},
    alpha,
    beta,
) where {T<:Real}
    row = IterationRow{T}(
        iter,
        mu,
        thetaP,
        thetaD,
        solve_info.objValPrimal,
        solve_info.objValDual,
        alpha.primal,
        alpha.dual,
        beta.value,
    )
    return print_one_iteration(io, row)
end

function print_one_iteration(io::IO, row::IterationRow{T}) where {T<:Real}
    @printf(io, "%2d %4.1e %4.1e %4.1e %+7.2e %+7.2e %4.1e %4.1e %4.2e\n",
            row.iter, row.mu, row.thetaP, row.thetaD,
            row.objP, row.objD, row.alphaP, row.alphaD, row.beta)
    return nothing
end

# Emit the C++ SDPA .result display schema (Parameter::display,
# sdpa_parts.cpp:227-255) instead of the param.sdpa INPUT layout, so the
# section is byte-compatible with the C++ reference .result files. The
# printf format strings below replicate the C++ fprintf calls exactly
# (including the trailing space after the xPrint/XPrint/YPrint/infPrint
# constants, which Parameters deliberately does not store — see the
# Parameters docstring).
function _print_parameters(io::IO, params::Parameters{T}) where {T<:Real}
    println(io, "Parameters are")
    println(io, "** Parameters **")
    @printf(io, "maxIteration =    %d\n", params.maxIteration)
    @printf(io, "epsilonStar  = %8.3e\n", params.epsilonStar)
    @printf(io, "lambdaStar   = %8.3e\n", params.lambdaStar)
    @printf(io, "omegaStar    = %8.3e\n", params.omegaStar)
    @printf(io, "lowerBound   = %8.3e\n", params.lowerBound)
    @printf(io, "upperBound   = %8.3e\n", params.upperBound)
    @printf(io, "betaStar     = %8.3e\n", params.betaStar)
    @printf(io, "betaBar      = %8.3e\n", params.betaBar)
    @printf(io, "gammaStar    = %8.3e\n", params.gammaStar)
    @printf(io, "epsilonDash  = %8.3e\n", params.epsilonDash)
    @printf(io, "precision    =    %d\n", params.precision)
    @printf(io, "xPrint       = %%+50.40Fe \n")
    @printf(io, "XPrint       = %%+50.40Fe \n")
    @printf(io, "YPrint       = %%+50.40Fe \n")
    @printf(io, "infPrint     = %%+50.40Fe \n")
    return nothing
end

function _print_vector(io::IO, v::SDPAVector{T}) where {T<:Real}
    fmt = _scalar_fmt(T)
    print(io, "{")
    for k in 1:length(v)
        if k > 1
            print(io, ", ")
        end
        print(io, Printf.format(Printf.Format(fmt), v.ele[k]))
    end
    println(io, " }")
    return nothing
end

function _print_dense_linear_space(
    io::IO, dls::SDPADenseLinearSpace{T}, blockStruct::AbstractVector{Int}
) where {T<:Real}
    fmt = _scalar_fmt(T)
    println(io, "{")
    sdp_idx = 0
    lp_idx = 0
    for k in eachindex(blockStruct)
        bs = blockStruct[k]
        if bs >= 2
            sdp_idx += 1
            mat = dls.SDP_block[sdp_idx]
            println(io, " {")
            for i in 1:bs
                print(io, "  {")
                for j in 1:bs
                    if j > 1
                        print(io, ", ")
                    end
                    print(io, Printf.format(Printf.Format(fmt), mat.de_ele[i, j]))
                end
                println(io, "}")
            end
            println(io, " }")
        else
            lp_size = abs(bs)
            for lp_k in 1:lp_size
                lp_idx += 1
                v = dls.LP_block[lp_idx]
                println(io, " { ", Printf.format(Printf.Format(fmt), v), " }")
            end
        end
    end
    println(io, "}")
    return nothing
end
