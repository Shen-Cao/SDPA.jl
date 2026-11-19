# Architecture

## Overview

SDPA.jl implements the primal-dual Mehrotra predictor-corrector
interior-point method for semidefinite programming. The algorithm
solves:

```
minimize    ⟨C, X⟩
subject to  ⟨A_i, X⟩ = b_i,  i = 1..m
            X ≽ 0  (positive semidefinite)
```

with dual:

```
maximize    b^T y
subject to  C - Σ y_i A_i - Z = 0
            Z ≽ 0
```

## Algorithm

The solver follows the Monteiro–Zhang A5 Jordan-product symmetrization
with Mehrotra predictor-corrector, matching C++ SDPA-GMP.

### Main Loop

```
1. Compute initial point (X₀, y₀, Z₀) at λ·I
2. Compute residuals
3. While not converged and iter < maxIteration:
   a. Mehrotra predictor:
      - Compute β from phase machine
      - Compute gvec (RHS of Schur system)
      - Assemble Schur complement matrix B
      - Cholesky factorize B = L·Lᵀ
      - Solve B·Δy_pred = g
      - Recover Δx_pred, Δz_pred
      - Compute step length α_pred
   b. Mehrotra corrector:
      - Update β from α_pred
      - Compute corrected gvec
      - Reuse Cholesky → solve B·Δy_corr = g_corr
      - Recover Δx_corr, Δz_corr
      - Compute step length α_corr
   c. Update: X += α·ΔX, y += α·Δy, Z += α·ΔZ
   d. Update μ, θ, ρ, phase
4. Return solution log
```

### Schur Complement

Each iteration solves the Newton system:

```
[ B  ] [Δy]   [g]
[    ] [  ] = [ ]
[ H  ] [Δz]   [h]
```

where `B` is the `m×m` Schur complement matrix. This is the
computational bottleneck. Two assembly paths exist:

- **Dense** (`compute_bmat_dense!`): Fill full `m×m` matrix.
  Threaded across constraint pairs. Used for small/dense problems.

- **Sparse** (`compute_bmat_sparse_sdp!` / `threaded_compute_bmat_sparse_sdp!`):
  Assemble only the profile-form entries directly in `sp_ele`. The
  profile is built from the **block-level skyline** (see
  [sparse-path.md](sparse-path.md)): constraints are connected
  iff they touch the same SDP block, the connectivity C++ uses via
  `Chordal::makeGraph` (`sdpa_chordal.cpp:141-211`). `make_aggrigate_index_sdp!` errors on
  any block-level pair outside the profile (C++ "program bug" parity).
  Threaded via `@tasks` for both SDP and LP contributions
  (bit-identical to serial). Used when the block-level profile
  density is ≤ 25%; problems with complete block connectivity (NSPS,
  theta5, example1) take the dense path, same as C++.

See [Sparse Path](sparse-path.md) for details.

### F1/F2/F3 Formulas

The Schur complement entry `B[i,j]` depends on the sparsity of the
constraint matrices `A_i` and `A_j`. Three formula variants:

- **F1**: `A_i` is dense (or above sparsity threshold).
  `B[i,j] = ⟨X · A_i · Z⁻¹, A_j⟩_F`
  Precompute `G = X · A_i · Z⁻¹`, then `B[i,j] = ⟨G, A_j⟩`.

- **F2**: `A_i` is sparse, `A_j` is sparse or dense.
  `B[i,j] = Σ_{(α,β)∈nnz(A_j)} A_j[α,β] · (X[:,α] · F[:,β])`
  where `F = A_i · Z⁻¹`.

- **F3**: Both `A_i` and `A_j` are sparse.
  4-index contraction exploiting both sparsity patterns.
  `B[i,j] = Σ_{(α,β)∈nnz(A_j)} Σ_{(γ,δ)∈nnz(A_i)} A_j[α,β] · A_i[γ,δ] · Z⁻¹[δ,β] · X[α,γ]`

Formula selection (`compute_formula!`) happens once per problem: it
fills `useFormula::Matrix{Symbol}` per (constraint, block), indexed
`[i, l]` with l the problem-block index, mirroring C++'s
`Newton::useFormula[i * SDP_nBlock + l]`. F3 is
never selected for dense blocks (asserted in `compute_formula!`, C++
`sdpa_newton.cpp:768-774`).

For BigFloat, the F1/F2/F3 inner products use in-place `mpfr_fma!`
(via `mp_fma!` wrappers) for zero-allocation accumulation. See
[Threading](threading.md) for performance details.

### In-Place MPFR Optimization

BigFloat arithmetic allocates ~88 bytes per operation (one `BigFloat`
struct per `+`, `*`, etc.), and a large solve performs hundreds of
millions of them — before this optimization, GC pressure from that
allocation stream dominated wall time.

The optimization (`src/mpfr_inplace.jl`) wraps MutableArithmetics.jl's
`operate_to!` API for zero-allocation arithmetic:

| Wrapper | Underlying ccall | Operation |
|---------|------------------|-----------|
| `mp_fma!(r,x,y,w)` | `mpfr_fma` | `r = x*y + w` (single rounding) |
| `mp_mul!(r,a,b)` | `mpfr_mul` | `r = a * b` |
| `mp_add!(r,a,b)` | `mpfr_add` | `r = a + b` |
| `mp_sub!(r,a,b)` | `mpfr_sub` | `r = a - b` |
| `mp_div!(r,a,b)` | `mpfr_div` | `r = a / b` |
| `mp_sqrt!(r,x)` | `mpfr_sqrt` | `r = sqrt(x)` |
| `mp_fms!(r,x,y,z)` | `mpfr_fms` | `r = x*y - z` |

**Key aliasing lesson (Julia 1.12)**: `BigFloat(buf)` returns the
SAME object (identity convert), not a copy; for `BigFloat`
containers, `copy`, `copyto!`, `fill!`, and `zeros` are reference
operations (see "BigFloat Aliasing" below). The type layer never
relies on them for values.

### MPLAPACK Port

Matrix operations ported from MPLAPACK (the arbitrary-precision
BLAS/LAPACK used by C++ SDPA-GMP):

| Module | Functions | Source |
|--------|-----------|--------|
| `Rgemm.jl` | `Rgemm!` (NN/TN/NT/TT) | `Rgemm.cpp` + `Rgemm_NN_omp.cpp` |
| `Rblas.jl` | `Raxpy!`, `Rdot`, `Rgemv!`, `Rscal!`, `Rcopy!` | `Raxpy_omp.cpp`, `Rdot_serial.cpp`, etc. |
| `Rlapack.jl` | `Rpotf2!`, `Rpotrf!`, `Rtrsm!`, `Rsyrk!`, `Rpotrf_with_adjust!` | `Rpotf2.cpp`, `Rpotrf.cpp`, `Rtrsm.cpp` |
| `dispatch.jl` | `sdpa_mul!` | — (type-based dispatcher) |

`Rgemm!` uses OhMyThreads `@tasks` for outer-loop parallelism (matching
the C++ `#pragma omp parallel for`). The `@tasks` branch (m*n*k > 4096)
uses per-task accumulator locals; the serial-branch accumulators are
named `*_serial` so the task closures never capture a shared BigFloat.
`Rpotrf!` (blocked Cholesky) calls `Rpotf2!` for
diagonal blocks and `Rgemm!`/`Rtrsm!` for off-diagonal updates.

`Rpotrf_with_adjust!` is plain `Rpotrf!` returning
the LAPACK-style info directly: 0 on success, non-zero on failure. The
name is kept for historical reasons — there is no adjust behavior: the
C++ diagonal-adjust algorithm (`choleskyFactorWithAdjust`,
`sdpa_linear.cpp:631-680`) is `#if 0` dead code and is deliberately not
ported. See [Sparse Path: Cholesky Failure](sparse-path.md) for the
failure propagation.

### Phase Machine

11-phase state machine tracking primal/dual feasibility:

| Phase | Meaning |
|-------|---------|
| `noINFO` | Initial / unknown |
| `pFEAS` | Primal feasible |
| `dFEAS` | Dual feasible |
| `pdFEAS` | Both feasible |
| `pdINF` | Primal-dual infeasible |
| `pFEAS_dINF` | Primal feasible, dual infeasible |
| `pINF_dFEAS` | Primal infeasible, dual feasible |
| `pdOPT` | Primal-dual optimal |
| `pUNBD` | Primal unbounded |
| `dUNBD` | Dual unbounded |
| `not_called` | Solver not started |

Termination: `pdOPT` when `μ < εStar`, residuals < `εDash`, and
`ρ < 1 + 1e-6`.

## BigFloat Aliasing and Zero-Allocation Kernels

### Reference Operations vs Value Semantics

Julia's `copyto!(dst::Matrix{BigFloat}, src::Matrix{BigFloat})` copies
**references** (not values): after `copyto!`, `dst[i] === src[i]` for all
elements. `copy`, `fill!` (`fill!(A, x)` aliases every slot to `x`), and
`zeros(BigFloat, ...)` (every slot is one shared zero) are reference
operations in the same class. This element-level aliasing silently
corrupts in-place MPFR kernels (`mp_add!`, `mp_fma!`, `mp_set!` mutate
the shared object).

The type layer is therefore built on value-semantic primitives
defined in `src/types/SDPADenseMatrix.jl`:

- `_independent_zeros(T, n)` / `_independent_zeros(T, m, n)`: fresh,
  independent BigFloat objects (never shared).
- `_fill_independent!(x, val)`: value-fills with independent elements.
- `_copyto_independent!(dst, src)`: value-copy via `mp_set!` (falls back
  to `copyto!` for isbits types).
- `_copy_independent(v)`: value-copy for `copy`.

Every public BigFloat container now holds independent elements and
implements value semantics through these primitives: `SDPAVector`,
the `LinearSpace` LP blocks, the `SDPASparseMatrix` dense branch and
`SDPADenseMatrix setidentity!`, `SDPASolution` LP fills, the min-eigen
Lanczos vectors, `BigFloatPool` elements, and the caller-exposed copies
(Optimizer LP blocks + SolutionLog), with `_copyto_independent!` Vector
overloads.

Historically, the kernels' `C[i,j] = deepcopy(acc[i])` pattern had
accidentally masked this aliasing; replacing `deepcopy` with in-place
`mp_set!` without independent elements caused silent corruption and
solver divergence — which is why the value-semantic layer above exists
and `deepcopy` is gone from the hot loops.

### Zero-Allocation Kernel Pattern

All MPLAPACK kernels (Rgemm, Rpotf2!, Rtrsm!) now use the zero-alloc
direct-write pattern:

```julia
# Rgemm NN (112 bytes for 10×10 BF256):
for j in 1:n
    if iszero(beta)
        for i in 1:m; mp_zero!(C[i,j]); end
    for l in 1:k
        mp_mul!(temp, alpha, B[l,j])
        for i in 1:m; mp_fma!(C[i,j], temp, A[i,l], C[i,j]); end
```

This writes directly to C via `mp_fma!`, eliminating the accumulator
array + deepcopy pattern. Requires C elements to be independent
(guaranteed by `_fill_independent!`, `_independent_zeros`, and the
value-semantic type layer).

## REVERSE_PRIMAL_DUAL Convention

The C++ SDPA-GMP is compiled with `#define REVERSE_PRIMAL_DUAL 1`
(`sdpa_tool.h:84`). This flag:

1. Swaps primal/dual labels in the display
2. Negates both objective values
3. Causes the solver to follow a central path that matches published
   optima

SDPA.jl replicates this by:
1. Negating C before solve
2. Transforming the SolutionLog for reversed display
3. Swapping xMat ↔ zMat

This convention is the `solve!` default (`reverse_primal_dual=true`) for
C++ parity: the solver negates C internally, displayed values follow the
C++ convention, and the file format stores the constant with the flipped
sign (`read_dat_s` reads +value where C++ reads -value). **Limitation**:
with the negated problem bounded, unboundedness/infeasibility of the
standard-form problem may NOT be detected (reported as `pdOPT`);
`reverse_primal_dual=false` gives standard-form `min` semantics with
direct detection. The MOI path (`src/moi/Optimizer.jl`) always passes
`reverse_primal_dual=false` — MOI/JuMP models are standard-form and rely
on the direct status mapping.

## Source Layout

```
src/
├── SDPA.jl                    Module entry point, __init__ (MPFR TLS check)
├── Parameters.jl              Parameters{T} + ParamSetting enum
├── solve.jl                   solve! + _solve_inner! + conditional BLAS policy
├── mpfr_inplace.jl            Zero-allocation MPFR wrappers (mp_fma!, mp_mul!, ...)
├── mplapack/                  MPLAPACK port (arbitrary-precision BLAS/LAPACK)
│   ├── Rgemm.jl               Rgemm! (NN/TN/NT/TT) with @tasks parallelism
│   ├── Rblas.jl               Raxpy!/Rdot/Rgemv!/Rscal!/Rcopy!
│   ├── Rlapack.jl             Rpotf2!/Rpotrf!/Rtrsm!/Rsyrk!/Rpotrf_with_adjust!
│   └── dispatch.jl            sdpa_mul! type-based dispatcher
├── types/                     Core data types
│   ├── SDPAProblem.jl         Problem definition (A, b, C, blockStruct)
│   ├── SDPASolution.jl        Current iterate (xMat, yVec, zMat)
│   ├── SDPAResiduals.jl       Primal/dual residuals
│   ├── SDPAWorkVariables.jl   Scratch space
│   ├── SDPAVector.jl          Vector type (independent elements)
│   ├── SDPADenseMatrix.jl     _independent_zeros/_fill_independent!/_copyto_independent!
│   ├── SDPASparseMatrix.jl    Sparse + dense matrix type
│   └── ...
├── io/                        Input/output
│   ├── read_dat.jl            Dense format parser
│   ├── read_dat_s.jl          Sparse format parser
│   └── write_result.jl        Result writer + iteration printer
├── algorithm/                 Core algorithm
│   ├── SchurComplement.jl     F1/F2/F3 (per-(i,l) useFormula) + make_bmat! + threaded_compute_bmat_sparse_sdp! / _compute_bmat_sparse_lp!
│   ├── SchurSolve.jl          Cholesky (Rpotrf_with_adjust! = plain Rpotrf! + failure) + solve
│   ├── JordanAlgebra.jl       Jordan products, trace, inv_chol (in-place MPFR)
│   ├── Mehrotra.jl            Predictor-corrector
│   ├── StepLength.jl          Step length via min eigenvalue
│   ├── IterateUpdate.jl       Update X, y, Z
│   ├── PhaseMachine.jl        Phase transitions
│   ├── SolveInfo.jl           Per-iteration tracking + infeasibility
│   ├── DirectionParameter.jl  β (centering parameter)
│   └── NewtonState.jl         Newton direction state
├── linear_algebra/            BigFloat linear algebra helpers
│   └── min_eigen.jl           Lanczos min-eigenvalue (restart at k > sqrt(n+9))
├── sparse/                    Sparse path
│   ├── AggregateSparsity.jl   Block-level skyline + AMD/METIS ordering + density gate
│   └── SparseCholesky.jl      Blocked profile-form Cholesky (@tasks) + triangular solve (in-place MPFR, zero-alloc)
├── threading/                 Multithreading
│   ├── ThreadedOps.jl         @tasks parallelization
│   └── BigFloatPool.jl        Per-task BigFloat workspace pool, wired via OhMyThreads @local
└── moi/                       MathOptInterface/JuMP integration
    ├── Optimizer.jl           MOI optimizer + result accessors
    └── ingest.jl              Problem ingestion from MOI model
```
