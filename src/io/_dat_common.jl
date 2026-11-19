"""
Shared finalization helper for `.dat` and `.dat-s` readers.

Both readers construct an SDPAProblem from the same components
(mDim, nBlock, blockStruct, b_vec, C, A) and apply the same
post-processing: initialize_index! + density-gated change_to_dense!.
This helper eliminates the duplication.
"""
function _finalize_problem!(
    mDim::Int, nBlock::Int, blockStruct::Vector{Int},
    b_vec::SDPAVector{T}, C::SDPASparseLinearSpace{T},
    A::Vector{SDPASparseLinearSpace{T}}, ::Type{T}
) where {T<:Real}
    problem = SDPAProblem{T}(mDim, nBlock, blockStruct, b_vec, C, A)
    initialize_index!(problem)

    # Conditionally convert each block of C and A_i to dense form via
    # the 0.20 density gate. Mirrors C++ sdpa_main.cpp:197-199.
    change_to_dense!(C)
    for Ai in A
        change_to_dense!(Ai)
    end

    return problem
end
