"""
    mehrotra!(newton_state, direction, problem, current_pt, current_res,
              mu, beta, phase, work) -> Bool

One step of the Mehrotra predictor-corrector. Returns `false` if
Cholesky of the Schur matrix fails (caller must break the main loop).

Predictor (direction = :predictor):
  1. `make_gvec!` (calls `compute_rmat!` with predictor's R = βμI - X·Z)
  2. `make_bmat!` (assembles B, factorizes once via `compute_dyvec!`)
  3. `compute_dyvec!` (triangular solve; caches Cholesky)
  4. `compute_dzmat!`
  5. `compute_dxmat!` (symmetrized)

Corrector (direction = :corrector):
  1. `make_gvec!` (calls `compute_rmat!` with corrector's R = βμI - X·Z - Δx·Δz)
  2. `compute_dyvec!` (REUSES predictor's Cholesky factorization)
  3. `compute_dzmat!`
  4. `compute_dxmat!` (symmetrized)

The orchestrator caches the Cholesky factorization between predictor
and corrector calls. The C++ does this implicitly because
`Newton::Mehrotra` calls `Make_bMat` only when `direction == PREDICTOR`
(see `sdpa_newton.cpp:1380-1385`).

Reference: Mehrotra 1992, "On the implementation of a primal-dual
interior point method", SIAM J Optimization 2:575-601.
"""
function mehrotra!(
    newton_state::NewtonState{T},
    direction::Symbol,
    problem::SDPAProblem{T},
    current_pt::SDPASolution{T},
    current_res::SDPAResiduals{T},
    mu::AverageComplementarity{T},
    beta::DirectionParameter{T},
    phase::PhaseValue,
    work::SDPAWorkVariables{T},
) where {T<:Real}
    (direction === :predictor || direction === :corrector) ||
        error("mehrotra!: direction must be :predictor or :corrector, got $direction")

    # Step 1+2: compute gvec (and r_zinv inside it).
    make_gvec!(newton_state, direction, problem, current_pt,
               current_res, mu, beta, phase, work)

    # Step 3: assemble B and factorize it (predictor only).
    # Dispatch invariant: ALWAYS go through make_bmat! dispatcher.
    # It internally chooses threaded_compute_bmat_dense! (dense path,
    # multi-threaded) or compute_bmat_sparse_sdp! (sparse path,
    # sequential) based on newton_state.use_sparse. Calling
    # threaded_compute_bmat_dense! directly bypasses the sparse dispatch
    # and causes silent divergence at nthreads >= 2 with use_sparse=true.
    if direction === :predictor
        make_bmat!(newton_state, problem, current_pt, work)
    end

    # Step 4: solve B*dy = g. Predictor factorizes; corrector reuses.
    predictor = direction === :predictor
    ok = if newton_state.bordered === nothing
        compute_dyvec!(newton_state, predictor)
    else
        _compute_verified_bordered_dyvec!(newton_state, predictor, problem, current_pt)
    end
    if !ok
        return false  # Cholesky failed — caller breaks main loop
    end

    # Step 5: recover dz.
    compute_dzmat!(newton_state, problem, current_res, phase)

    # Step 6: recover dx (symmetrized).
    compute_dxmat!(newton_state, current_pt, work)

    return true
end
