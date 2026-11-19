"""
    ParamSetting

Enum selecting one of the three C++ `setDefaultParameter` presets.
Mirrors C++ `Parameter::parameterType` at
`sdpa_parts.h:88`.

Values:
- `DEFAULT`: balanced; matches C++ `PARAMETER_DEFAULT`.
- `UNSTABLE_BUT_FAST`: aggressive β, low precision; matches C++
  `PARAMETER_UNSTABLE_BUT_FAST`.
- `STABLE_BUT_SLOW`: conservative β, high precision; matches C++
  `PARAMETER_STABLE_BUT_SLOW`.
"""
@enum ParamSetting begin
    DEFAULT
    UNSTABLE_BUT_FAST
    STABLE_BUT_SLOW
end

"""
    Parameters{T<:Real}

Solver parameters. Mirrors C++ `sdpa::Parameter` at
`sdpa_parts.h:85-114`.

The C++ class also has four `*Print` printf format strings; we omit
them in Julia (output uses native `@sprintf` / `Printf`, and the print
format is chosen based on `T` rather than user-configurable per-field).

Fields
------
- `maxIteration::Int`:  hard cap on IPM iterations
- `precision::Int`:     MPFR precision in bits (only applied inside
  `solve!` via `setprecision(BigFloat, precision) do ... end`)
- `epsilonStar::T`:     convergence threshold for the duality gap
- `lambdaStar::T`:      initial central-path parameter (X=Z=lambda*I)
- `omegaStar::T`:       feasibility threshold multiplier
- `lowerBound::T`:      lower bound for objective-unbounded detection
- `upperBound::T`:      upper bound for objective-unbounded detection
- `betaStar::T`:        lower-bound centering parameter (0 < β* < 1)
- `betaBar::T`:         upper-bound centering parameter (β* ≤ β̄ < 1)
- `gammaStar::T`:       step-length scaling (0 < γ* < 1)
- `epsilonDash::T`:     primal/dual feasibility threshold

The three preset value sets match C++ `Parameter::setDefaultParameter`
at `sdpa_parts.cpp:136-181`.

**Per-T presets:** `Parameters{BigFloat}` uses the C++-GMP
presets exactly. `Parameters{Float16}`/`Float32`/`Float64` use
upstream SDPA's double-precision tolerances (`epsilonStar = epsilonDash =
1.0e-7`, ...) because the GMP 1.0e-30 values are unreachable in binary
floating point (false infeasibility / unreachable `pdOPT`).
"""
mutable struct Parameters{T<:Real}
    maxIteration::Int
    precision::Int
    epsilonStar::T
    lambdaStar::T
    omegaStar::T
    lowerBound::T
    upperBound::T
    betaStar::T
    betaBar::T
    gammaStar::T
    epsilonDash::T

    # Inner constructor: every positional construction path (including the
    # auto-generated positional form the struct would otherwise get) runs
    # `_validate`. Without this, `Parameters{T}(f1..f11)` bypassed the
    # validators entirely and could construct silently-invalid instances
    # (e.g. maxIteration = 0). `_validate` is defined later in this file;
    # that is fine because it is called at construction time, after the
    # module is fully loaded.
    function Parameters{T}(
        maxIteration::Int, precision::Int,
        epsilonStar::T, lambdaStar::T, omegaStar::T,
        lowerBound::T, upperBound::T,
        betaStar::T, betaBar::T, gammaStar::T, epsilonDash::T,
    ) where {T<:Real}
        p = new{T}(maxIteration, precision, epsilonStar, lambdaStar,
                   omegaStar, lowerBound, upperBound, betaStar,
                   betaBar, gammaStar, epsilonDash)
        _validate(p)
        return p
    end
end

# Default values per preset. These mirror C++ sdpa_parts.cpp:138-181
# exactly. The C++ source is authoritative.
#
# Numbers as rationals where exact (β, γ); as powers of 10 where inexact
# (ε, λ, bounds). The `precision` field matches C++ SDPA-GMP default of
# 200 bits (~60 decimal digits).
const _DEFAULT_PRESET = (
    maxIteration = 200,
    precision = 200,  # matches C++ SDPA-GMP default
    epsilonStar = 1.0e-30,
    lambdaStar = 1.0e+4,
    omegaStar = 2.0,
    lowerBound = -1.0e+5,
    upperBound = 1.0e+5,
    betaStar = 0.1,
    betaBar = 0.3,
    gammaStar = 0.9,
    epsilonDash = 1.0e-30,
)
const _UNSTABLE_BUT_FAST_PRESET = (
    maxIteration = 100,
    precision = 100,
    epsilonStar = 1.0e-30,
    lambdaStar = 1.0e+2,
    omegaStar = 2.0,
    lowerBound = -1.0e+5,
    upperBound = 1.0e+5,
    betaStar = 0.01,
    betaBar = 0.02,
    gammaStar = 0.98,
    epsilonDash = 1.0e-30,
)
const _STABLE_BUT_SLOW_PRESET = (
    maxIteration = 1000,
    precision = 300,
    epsilonStar = 1.0e-30,
    lambdaStar = 1.0e+2,
    omegaStar = 2.0,
    lowerBound = -1.0e+5,
    upperBound = 1.0e+5,
    betaStar = 0.2,
    betaBar = 0.4,
    gammaStar = 0.5,
    epsilonDash = 1.0e-30,
)

# Per-T presets: Parameters{BigFloat} uses the C++ SDPA-GMP
# presets above (bit-parity with the C++ reference; the 1e-30 golden
# gate depends on this invariant). For Float16/32/64 the 1.0e-30
# tolerances are unreachable (below eps(Float64)), which makes the
# rho-based infeasibility detection fire on degenerate-but-solvable
# problems (false INFEASIBLE) and leaves pdOPT unreachable for problems
# without exactly-representable optima.
#
# The double presets mirror upstream SDPA's double-precision DEFAULT
# (epsilonStar = epsilonDash = 1.0e-7, betaBar = 0.2, lambdaStar = 1.0e2,
# maxIteration = 100; see the SDPA project's param.py). Upstream documents
# only the default preset for double, so UNSTABLE_BUT_FAST /
# STABLE_BUT_SLOW keep the GMP preset shape with the reachable 1.0e-7
# tolerances substituted. The precision field is unused for non-BigFloat
# types (solve! only wraps setprecision for BigFloat).
const _DOUBLE_DEFAULT_PRESET = (
    maxIteration = 100,
    precision = 200,
    epsilonStar = 1.0e-7,
    lambdaStar = 1.0e+2,
    omegaStar = 2.0,
    lowerBound = -1.0e+5,
    upperBound = 1.0e+5,
    betaStar = 0.1,
    betaBar = 0.2,
    gammaStar = 0.9,
    epsilonDash = 1.0e-7,
)
const _DOUBLE_UNSTABLE_BUT_FAST_PRESET = (
    maxIteration = 100,
    precision = 100,
    epsilonStar = 1.0e-7,
    lambdaStar = 1.0e+2,
    omegaStar = 2.0,
    lowerBound = -1.0e+5,
    upperBound = 1.0e+5,
    betaStar = 0.01,
    betaBar = 0.02,
    gammaStar = 0.98,
    epsilonDash = 1.0e-7,
)
const _DOUBLE_STABLE_BUT_SLOW_PRESET = (
    maxIteration = 1000,
    precision = 300,
    epsilonStar = 1.0e-7,
    lambdaStar = 1.0e+2,
    omegaStar = 2.0,
    lowerBound = -1.0e+5,
    upperBound = 1.0e+5,
    betaStar = 0.2,
    betaBar = 0.4,
    gammaStar = 0.5,
    epsilonDash = 1.0e-7,
)

function _preset_values(setting::ParamSetting)
    if setting === DEFAULT
        return _DEFAULT_PRESET
    elseif setting === UNSTABLE_BUT_FAST
        return _UNSTABLE_BUT_FAST_PRESET
    elseif setting === STABLE_BUT_SLOW
        return _STABLE_BUT_SLOW_PRESET
    end
end

# T-typed preset selection. Binary floating-point types get the
# reachable upstream-SDPA double tolerances; everything else (BigFloat)
# keeps the C++-GMP presets.
function _preset_values(setting::ParamSetting, ::Type{T}) where {T<:Real}
    if T <: Union{Float16,Float32,Float64}
        if setting === DEFAULT
            return _DOUBLE_DEFAULT_PRESET
        elseif setting === UNSTABLE_BUT_FAST
            return _DOUBLE_UNSTABLE_BUT_FAST_PRESET
        else
            return _DOUBLE_STABLE_BUT_SLOW_PRESET
        end
    end
    return _preset_values(setting)
end

"""
    Parameters{T}(; kwargs...) where {T<:Real}

Keyword-based constructor following Hypatia's `RealOrNothing` pattern:
every parameter is optional and defaults to `nothing`. The constructor
first selects a preset (controlled by the `setting` keyword, default
`DEFAULT`), then overrides any user-supplied non-`nothing` value, then
runs all validators.

Validators (mirror C++ logic):
- `epsilonStar > 0`
- `lambdaStar > 0`
- `omegaStar > 1`
- `lowerBound < upperBound`
- `0 < betaStar ≤ betaBar < 1`
- `0 < gammaStar < 1`
- `epsilonDash > 0`
- `precision > 0`
- `maxIteration > 0`

Numeric literals are converted to `T` via `T(value)` to avoid silent
Float64 fallback (a BigFloat user must get BigFloat fields).
"""
function Parameters{T}(;
    maxIteration::Union{Nothing,Int}=nothing,
    precision::Union{Nothing,Int}=nothing,
    epsilonStar::Union{Nothing,Real}=nothing,
    lambdaStar::Union{Nothing,Real}=nothing,
    omegaStar::Union{Nothing,Real}=nothing,
    lowerBound::Union{Nothing,Real}=nothing,
    upperBound::Union{Nothing,Real}=nothing,
    betaStar::Union{Nothing,Real}=nothing,
    betaBar::Union{Nothing,Real}=nothing,
    gammaStar::Union{Nothing,Real}=nothing,
    epsilonDash::Union{Nothing,Real}=nothing,
    setting::ParamSetting=DEFAULT,
) where {T<:Real}
    # Must be checked BEFORE any T() conversion: with T=Int the preset's
    # 1.0e-30 would die on an opaque InexactError instead of this descriptive
    # ArgumentError.
    T <: AbstractFloat ||
        throw(ArgumentError("Parameters{T} requires an AbstractFloat element type, got $T"))
    preset = _preset_values(setting, T)
    _to_T(x) = T(x)

    maxIteration_v = maxIteration === nothing ? preset.maxIteration : maxIteration
    precision_v    = precision    === nothing ? preset.precision    : precision
    epsilonStar_v  = epsilonStar  === nothing ? preset.epsilonStar  : epsilonStar
    lambdaStar_v   = lambdaStar   === nothing ? preset.lambdaStar   : lambdaStar
    omegaStar_v    = omegaStar    === nothing ? preset.omegaStar    : omegaStar
    lowerBound_v   = lowerBound   === nothing ? preset.lowerBound   : lowerBound
    upperBound_v   = upperBound   === nothing ? preset.upperBound   : upperBound
    betaStar_v     = betaStar     === nothing ? preset.betaStar     : betaStar
    betaBar_v      = betaBar      === nothing ? preset.betaBar      : betaBar
    gammaStar_v    = gammaStar    === nothing ? preset.gammaStar    : gammaStar
    epsilonDash_v  = epsilonDash  === nothing ? preset.epsilonDash  : epsilonDash

    # Convert every numeric literal to T to avoid silent Float64 fallback.
    params = Parameters{T}(
        Int(maxIteration_v),
        Int(precision_v),
        _to_T(epsilonStar_v),
        _to_T(lambdaStar_v),
        _to_T(omegaStar_v),
        _to_T(lowerBound_v),
        _to_T(upperBound_v),
        _to_T(betaStar_v),
        _to_T(betaBar_v),
        _to_T(gammaStar_v),
        _to_T(epsilonDash_v),
    )
    # No explicit `_validate(params)` here: the inner constructor already
    # runs it (and the T<:AbstractFloat guard above fires first for
    # non-float T).
    return params
end

# Default T-constructor: BigFloat (the primary supported precision).
Parameters(; kwargs...) = Parameters{BigFloat}(; kwargs...)

"""
    _validate(params)

Run every validator; throw ArgumentError with a clear message on first
failure. Mirrors the implicit C++ checks at the head of
`setDefaultParameter`.
"""
function _validate(p::Parameters{T}) where {T<:Real}
    # Non-float T (Int, Rational, ...) fails opaquely later (_to_T InexactError
    # / _scalar_fmt MethodError); reject it up front with a descriptive error.
    # The inner constructor validates on every construction path, so this
    # covers positional construction too.
    T <: AbstractFloat ||
        throw(ArgumentError("Parameters{T} requires an AbstractFloat element type, got $T"))
    p.maxIteration > 0 || throw(ArgumentError("maxIteration must be > 0, got $(p.maxIteration)"))
    p.precision > 0 || throw(ArgumentError("precision must be > 0, got $(p.precision)"))
    p.epsilonStar > 0 || throw(ArgumentError("epsilonStar must be > 0, got $(p.epsilonStar)"))
    p.lambdaStar > 0 || throw(ArgumentError("lambdaStar must be > 0, got $(p.lambdaStar)"))
    p.omegaStar > 1 || throw(ArgumentError("omegaStar must be > 1, got $(p.omegaStar)"))
    p.lowerBound < p.upperBound ||
        throw(ArgumentError("lowerBound must be < upperBound, got $(p.lowerBound) >= $(p.upperBound)"))
    zero(T) < p.betaStar || throw(ArgumentError("betaStar must be > 0, got $(p.betaStar)"))
    p.betaStar <= p.betaBar ||
        throw(ArgumentError("betaStar must be <= betaBar, got $(p.betaStar) > $(p.betaBar)"))
    p.betaBar < 1 || throw(ArgumentError("betaBar must be < 1, got $(p.betaBar)"))
    zero(T) < p.gammaStar || throw(ArgumentError("gammaStar must be > 0, got $(p.gammaStar)"))
    p.gammaStar < 1 || throw(ArgumentError("gammaStar must be < 1, got $(p.gammaStar)"))
    p.epsilonDash > 0 || throw(ArgumentError("epsilonDash must be > 0, got $(p.epsilonDash)"))
    return nothing
end

"""
    read_parameter_file(path, T=BigFloat)

Parse the C++ `param.sdpa` format. The file has 15 whitespace-separated
lines (one per field). The last 4 lines are printf format strings
(`%+50.40Fe` etc.) which we ignore — Julia uses native formatting.

Format (per `param.sdpa` and C++ `Parameter::readFile`
at `sdpa_parts.cpp:188-205`):

```
200                  unsigned int maxIteration
1.0E-30              epsilonStar
1.0E4                lambdaStar
2.0                  omegaStar
-1.0E5               lowerBound
1.0E5                upperBound
0.1                  betaStar
0.3                  betaBar
0.9                  gammaStar
1.0E-30              epsilonDash
200                  precision
%+50.40Fe            xPrint    (skipped)
%+50.40Fe            XPrint    (skipped)
%+50.40Fe            YPrint    (skipped)
%+50.40Fe            infPrint  (skipped)
```

The first whitespace-delimited token of each line is the value; the
rest is the descriptive comment.
"""
function read_parameter_file(
    path::AbstractString, ::Type{T}=BigFloat
) where {T<:Real}
    lines = readlines(path)
    length(lines) >= 11 ||
        error("parameter file needs >= 11 lines, got $(length(lines))")

    # A blank/whitespace-only line must not die on `split(line)[1]` with
    # a raw BoundsError: carry the file path, the 1-based line number,
    # and the expected field name into the error so the failure names
    # the exact malformed line.
    function parse_num(line::AbstractString, ::Type{U}, k::Int, field::AbstractString) where {U<:Real}
        tokens = split(line)
        isempty(tokens) && error(
            "parameter file $path: line $k (field $field): no value found — blank or malformed line"
        )
        token = tokens[1]
        return U === Int ? parse(Int, token) : parse(U, token)
    end

    maxIteration = parse_num(lines[1], Int, 1, "maxIteration")
    epsilonStar  = parse_num(lines[2], T, 2, "epsilonStar")
    lambdaStar   = parse_num(lines[3], T, 3, "lambdaStar")
    omegaStar    = parse_num(lines[4], T, 4, "omegaStar")
    lowerBound   = parse_num(lines[5], T, 5, "lowerBound")
    upperBound   = parse_num(lines[6], T, 6, "upperBound")
    betaStar     = parse_num(lines[7], T, 7, "betaStar")
    betaBar      = parse_num(lines[8], T, 8, "betaBar")
    gammaStar    = parse_num(lines[9], T, 9, "gammaStar")
    epsilonDash  = parse_num(lines[10], T, 10, "epsilonDash")
    precision    = parse_num(lines[11], Int, 11, "precision")

    params = Parameters{T}(;
        maxIteration=maxIteration,
        precision=precision,
        epsilonStar=epsilonStar,
        lambdaStar=lambdaStar,
        omegaStar=omegaStar,
        lowerBound=lowerBound,
        upperBound=upperBound,
        betaStar=betaStar,
        betaBar=betaBar,
        gammaStar=gammaStar,
        epsilonDash=epsilonDash,
    )
    return params
end
