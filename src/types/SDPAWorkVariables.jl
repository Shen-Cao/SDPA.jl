"""
    SDPAWorkVariables{T<:Real}

Pre-allocated scratch space for the IPM inner loop. Allocating in the
hot path is the single biggest source of BigFloat slowdown (each
`mpfr_t` is a heap allocation); pre-allocating once at iteration 0 and
reusing across all iterations is the standard SDPA optimization.

Mirrors C++ `sdpa::WorkVariables` at
`sdpa_dataset.h:148-173` and constructor at
`sdpa_dataset.cpp:669-720`.

Fields
------
- `DLS1, DLS2::SDPADenseLinearSpace{T}`:  dense workspace matrices
  (one per block)
"""
mutable struct SDPAWorkVariables{T<:Real}
    DLS1::SDPADenseLinearSpace{T}
    DLS2::SDPADenseLinearSpace{T}
end

function SDPAWorkVariables{T}(
    blockStruct::AbstractVector{Int}
) where {T<:Real}
    dls1 = SDPADenseLinearSpace{T}(blockStruct)
    dls2 = SDPADenseLinearSpace{T}(blockStruct)

    return SDPAWorkVariables{T}(dls1, dls2)
end
