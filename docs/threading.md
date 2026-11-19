# Threading

SDPA.jl uses [OhMyThreads.jl](https://github.com/JuliaFolds/OhMyThreads.jl)
for multithreading. The BLAS thread policy is set **conditionally** inside
`solve!` based on numeric type: BigFloat → 1 BLAS thread (BigFloat
arithmetic doesn't use BLAS; Julia threads do the parallel work),
Float64 → `Threads.nthreads()` (LAPACK parallelism for dense Float64).

## Where Threading Is Applied

| Operation | Function | Parallelism |
|-----------|----------|-------------|
| Dense bmat assembly | `threaded_compute_bmat_dense!` | Outer constraint loop (k₁) |
| Sparse bmat assembly (SDP) | `threaded_compute_bmat_sparse_sdp!` | Per SDP block (`@tasks`) |
| Sparse bmat assembly (LP) | `threaded_compute_bmat_sparse_lp!` | Per LP scalar (`@tasks`) |
| Sparse Cholesky | `sparse_cholesky!` | Block-columns (`@tasks` within) |
| Rgemm! (BigFloat matmul) | `_Rgemm_NN!` / `_Rgemm_TN!` / ... | Outer column loop (j) |
| Inverse Cholesky | `threaded_get_inv_chol!` | Per SDP block |
| Inverse Cholesky + Inverse | `threaded_get_inv_chol_and_inv!` | Per SDP block |
| Min eigenvalue | `threaded_min_eigen_dls` | Per SDP block |
| Inner product (residual) | `multi_inner_product_to_a!` | Per constraint |

All use `OhMyThreads.@tasks` (not `Threads.@threads`):

```julia
@tasks for k in 1:nblocks
    # Each iteration writes to a unique output slot — no reduction needed.
    vals[k] = compute_block(k)
end
```

Sparse assembly and Cholesky threading are **bit-identical** to their
serial paths: per-entry accumulation order is preserved exactly (the LP
assembly uses per-task value buffers + sequential reduction because
multiple LP scalars can contribute to one entry; the blocked Cholesky
serializes across block-columns and parallelizes only the independent
within-block updates). Verified at 1 vs 4 threads on BigFloat and
Float64.

### Per-Task Scratch via OhMyThreads `@local` (BigFloatPool)

The dense bmat assembly and the min-eigenvalue step keep their scratch
workspace in per-task pools wired through OhMyThreads `@local` task-local
values. The pattern:

```julia
@tasks for k in 1:nconstr_l
    # One typed TLV: the initializer closure runs ONCE per task on first
    # access and is cached in that task's local storage. TLVs cannot
    # reference each other, so a single TLV carries the whole tuple.
    @local ws::Tuple{BigFloatPool{T}, Matrix{T}, Matrix{T},
                     BigFloat, BigFloat, BigFloat, BigFloat} = let
        pool = BigFloatPool{T}(; precision=precision(BigFloat),
                               max_mats=2, max_vecs=0, mat_size=(bs, bs))
        (pool,
         acquire_mat!(pool, bs, bs), acquire_mat!(pool, bs, bs),
         zero(BigFloat), zero(BigFloat), zero(BigFloat), zero(BigFloat))
    end
    # ... work1 = ws[2], work2 = ws[3], scalar accumulators = ws[4:7],
    # reused across ALL iterations of this task's chunk ...
end
```

`threaded_compute_bmat_dense!` uses the tuple form above (pool + work1/
work2 + four scalar accumulators, acquired once per task and reused
across every k1 in the task's chunk); `threaded_min_eigen_dls` uses the
simpler `@local pool::BigFloatPool{T} = BigFloatPool{T}(...)` form and
acquires/releases its per-block matrices through the pool (equal-sized
blocks within a chunk hit the free list). Block sizes vary there, so the
pool is created with `mat_size=(0, 0)` and acquires on demand.

Reuse across iterations is bit-identical because every read of a scratch
buffer follows a full overwrite within the same iteration:
`_sparse_dense_mul!` zeroes work1 in-place, `sdpa_mul!`/`mul!` fully
overwrite work2 (beta = 0), and `cal_f1!`/`cal_f2!`/`cal_f3!` zero their
scalar accumulators on entry (`mp_zero!`). There is no cross-task
sharing: each task touches only its own TLV, so no locking is needed.

The pool's elements are independent BigFloats: the pre-allocated
matrices/vectors come from `_independent_zeros`, and reused ones are
re-zeroed with `_fill_independent!`. Plain `zeros(BigFloat, ...)`
would alias every slot to one shared object and corrupt the in-place
MPFR writes. See
[Architecture: BigFloat Aliasing](architecture.md) for the aliasing
primitives.

## In-Place MPFR and MPLAPACK

The MPFR optimization added:

- **`src/mpfr_inplace.jl`**: Zero-allocation BigFloat wrappers (`mp_fma!`,
  `mp_mul!`, `mp_add!`, etc.) using direct `mpfr_*` ccalls via
  MutableArithmetics.jl's `operate_to!` API. These eliminate the ~88 bytes
  per BigFloat operation that dominated GC pressure in the pre-optimization
  baseline.

- **`src/mplapack/`**: Ported BLAS/LAPACK operations with `@tasks`
  parallelism:
  - `Rgemm.jl`: Matrix multiply (NN/TN/NT/TT), 11× faster than
    `LinearAlgebra.mul!` at n=100 BigFloat(256). The `@tasks` branch
    (m*n*k > 4096) uses per-task accumulator locals.
  - `Rblas.jl`: Level-1 BLAS (Raxpy!, Rdot, Rgemv!, Rscal!, Rcopy!).
  - `Rlapack.jl`: Rpotf2! (unblocked Cholesky), Rpotrf! (blocked Cholesky),
    Rtrsm! (triangular solve), Rsyrk! (symmetric rank-k update),
    Rpotrf_with_adjust! (plain `Rpotrf!` + LAPACK-style info return).
  - `dispatch.jl`: `sdpa_mul!` dispatcher — Rgemm! for BigFloat,
    LinearAlgebra.mul! for Float64.

## Bit-Identity Guarantee

Results are **bit-identical across thread counts**, and the claim is
enforced by tests above the parallel threshold:

1. Each task writes to exactly one output slot (no shared writes).
2. No floating-point reductions across threads.
3. BigFloat arithmetic is deterministic given fixed operation order.
4. Threaded variants replicate the serial accumulation order term-by-term
   (e.g. `threaded_get_inv_chol!` and `threaded_get_inv_chol_and_inv!`
   reproduce the serial `mp_fma!` chains bit-for-bit; `s += a*b` would
   round twice and differ by ulps for multi-term sums).
5. Task-closure locals are genuinely per-task: same-named serial-branch
   accumulators are renamed `*_serial` so `@tasks` bodies can never
   capture (and race on) a shared BigFloat.

Bit-identity is TESTED above the parallel threshold (m*n*k > 4096,
where the Rgemm! kernels take the `@tasks` branch):

- `test/mpfr/test_Rgemm.jl`, "Parallel @tasks bit-identity across
  thread counts": Rgemm! results `===` at 1 vs 4 threads on 64³
  matrices (the parallel branch).
- `test/threading/test_cross_thread_bit_identity.jl`, "Cross-thread-count
  bit-identity: BigFloat above the Rgemm parallel threshold": solves arch0 (a
  161×161 SDP block, 161³ ≫ 4096, so every BigFloat Rgemm! crosses the
  threshold each iteration) in subprocesses at `JULIA_NUM_THREADS=1` and
  `=4` and asserts the objectives are `===`. Float64 would not work
  here: it goes through LAPACK/BLAS, whose internal reduction order
  changes with BLAS thread count.

## Performance Characteristics

The zero-allocation MPFR + MPLAPACK rework gave 2.6-30× on the small
benchmarks of its era (example1 dense 30×, arch0 2.6×, Rgemm! 11×); see
the landed-wins table in [performance.md](performance.md).

### Small problems (m < ~200)

Threading is **counterproductive** for Float64: OhMyThreads
task-scheduling overhead (~ms per task) exceeds the per-iteration
compute time at this problem size (measured on arch0, m=174: 8 threads
ran 2.3× slower than 1 thread).

### Large problems (m > ~1000)

For large problems, the **block-level skyline density check**
automatically selects the optimal path:
- Block-level skyline density ≤ 25%: sparse path (profile-form Cholesky)
- Block-level skyline density > 25%: dense path (LAPACK for Float64,
  Rpotrf! for BigFloat)

The density is computed from `build_skyline_from_problem!` (constraints
connected iff they touch the same SDP block, the C++
`Chordal::makeGraph` connectivity), not from the position-level
aggregate graph. The **ordering choice matters**: ordering the
position-level aggregate graph interleaves dense SDP cliques and yields
a near-full profile; ordering the block-level graph keeps cliques
contiguous. Measured per-iteration numbers for both orderings and the
`SDPA_ORDERING=auto` routing rule: [sparse-path.md](sparse-path.md) and
`benchmark/nsps/README.md`.

Thread-count scaling on the NSPS forced-sparse factor phase showed
useful scaling through 16-32 threads with efficiency falling off above
that (memory-bandwidth-bound; the reference machine's measured
8/16/32/64-thread table is in `benchmark/nsps/README.md` — on smaller
machines cap `JULIA_NUM_THREADS` accordingly).

## BLAS Configuration

The `_solve_inner!` function sets BLAS threads conditionally:

```julia
# In src/solve.jl _solve_inner!:
if T <: BigFloat
    BLAS.set_num_threads(1)      # BigFloat doesn't use BLAS
else
    BLAS.set_num_threads(Threads.nthreads())  # Float64 uses LAPACK
end
```

This prevents BLAS from competing with Julia threads for BigFloat
solves, while allowing LAPACK parallelism for Float64.

The `__init__` hook performs a runtime `mpfr_buildopt_tls_p()` check
to verify MPFR was built thread-safe.

## Recommended Configuration

```bash
# For development/testing (matches CI):
JULIA_NUM_THREADS=4 julia --project=. -e 'using Pkg; Pkg.test("SDPA")'

# For large sparse problems:
JULIA_NUM_THREADS=8 julia --project=. your_script.jl

# For small/dense problems (threading overhead not worth it):
JULIA_NUM_THREADS=1 julia --project=. your_script.jl
```
