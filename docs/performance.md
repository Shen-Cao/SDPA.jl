# Performance

The evergreen picture: where SDPA.jl stands relative to C++ SDPA-GMP,
which cost factors are verified, which hypotheses were tested and
rejected, and which gaps remain. Planned work lives in
[roadmap.md](roadmap.md).

The reference workload is NSPS (m=10259, 4 SDP blocks of 71×71, 1 LP
block of 14536) at 200-256-bit precision against C++ SDPA-GMP. All
wall-clock numbers on this page come from a single reference machine
(see `benchmark/nsps/README.md`); cross-machine comparisons have not
been made — treat absolute times as machine-relative.

## Verified cost factors

### 1. GMP mpf vs MPFR BigFloat arithmetic (~1.1-1.5× per op)

C++ SDPA-GMP uses GMP's `mpf_class` (no correct rounding — it truncates
to the target precision after each operation). Julia uses MPFR's
`BigFloat` (guaranteed round-to-nearest). Published benchmarks show MPFR
is 1.5-3× slower than GMP at the same precision
(https://www.mpfr.org/mpfr-3.1.0/timings.html); a naive top-level Julia
loop even measured ~200 ns/op for a 200-bit MPFR multiply. Careful
microbenchmarks on the reference machine put the real ratio at
**1.1× for simple ops and 1.5× for the fused op** (the earlier "~3×"
claim was a benchmark artifact):

| Op (200-bit) | GMP mpf | MPFR | Ratio |
|---|---:|---:|---:|
| mul | 27.4 ns | 30.3 ns | 1.11× |
| add | 12.8 ns | 13.9 ns | 1.09× |
| div | 67.8 ns | 69.8 ns | 1.03× |
| FMA-equivalent (mul+add vs `mp_fma!`) | 30.7 ns | 47.2 ns | **1.54×** |

Only the FMA ratio matters (FMA dominates the kernels). For
interior-point SDP the correct rounding is unnecessary (the algorithm is
self-correcting) — the overhead is pure waste, but bounded at ~1.5×.
Scale: the NSPS dense Cholesky is O(m³/3) ≈ 3.6×10¹¹ FMA operations; at
200 ns/op (MPFR) that alone is ~72,000 s single-threaded. Note GMP has no
fused op at all — kernels on an mpf-backed type would revert to separate
mul+add (2 roundings).

### 2. Full dense vs profile storage (~2×)

C++ stores the Schur matrix in profile form (upper triangle only):
52.6M entries for NSPS. A full dense m×m matrix is 105.2M entries. The
lower triangle is never used (the matrix is symmetric), but a dense
representation allocates and zeroes it, and the assembly fills both
triangles. The Cholesky itself only processes the upper triangle, so the
extra factor of ~1.5× work is in assembly, not factorization. For
NSPS-class problems (100% block-level density) **both solvers take the
dense path by design** — the same choice C++ makes at its density gate
(see [sparse-path.md](sparse-path.md)).

### 3. Per-element size (~2×)

| | Julia BigFloat(200) | GMP mpf_class(200) |
|---|---|---|
| Struct | 32 B (precision, sign, exponent, limb pointer) | 24 B (size, exponent, limb pointer) |
| Heap limbs | 32 B (4 × 64-bit) | 32 B (4 × 64-bit) |
| Runtime overhead | ~48 B (GC type tag, mark bits, padding) | 0 (manual memory management) |
| **Total** | **112 B** | **~56 B** |

Impact on NSPS: C++ sparse profile 52.6M × 56 B ≈ 2.9 GB vs a Julia
dense matrix 105.2M × 112 B ≈ 11.8 GB (a 4× ratio = 2× full-vs-upper ×
2× per-element). This cannot be reduced without a different
arbitrary-precision type (same as factor 1) — architectural.

## Tested and rejected

### SPOOLES supernodal Cholesky advantage — NOT a factor

Both C++ and Julia use the identical scalar left-looking profile-form
Cholesky algorithm: C++ `Lal::getCholesky(SparseMatrix&, int*)` at
`sdpa_linear.cpp:395-437`; Julia `sparse_cholesky!` at
`src/sparse/SparseCholesky.jl`. SPOOLES is used for **ordering**
(nested dissection, minimum degree) and **symbolic factorization**
(computing the profile structure), not for the numeric factorization.
For NSPS specifically, the block-level graph is complete, so ANY
ordering produces the full upper-triangle profile.

### OMP parallelism in the C++ Cholesky — NOT a factor

All `#pragma omp` directives in `sdpa_linear.cpp` and `sdpa_newton.cpp`
are commented out:

```cpp
//#pragma omp parallel for    ← line 236, COMMENTED
//#pragma omp critical        ← line 252, COMMENTED
//#pragma omp parallel        ← line 942, COMMENTED
```

The reference binary links `libgomp` only because the SPOOLES library
uses OMP internally for its ordering algorithms. The C++ 742 s/iter NSPS
reference is **single-threaded**; Julia's `@tasks` threading (bit-ident
by construction, see [threading.md](threading.md)) is a genuine
advantage on multi-core.

### Cache blocking for BigFloat GEMM — rejected by decision

GotoBLAS-style cache-blocked (tiled) `Rgemm!` kernels were implemented
for all four transpose cases and benchmarked: blocked was slower in
every case (worst +4.5%, best −0.7%, within noise). BigFloat `mp_fma!`
is compute-bound (ccall → MPFR, hundreds of cycles per element), so
memory traffic is irrelevant — fundamentally different from Float64
GEMM. The finding generalizes: cache-oriented techniques (blocking,
supernodal factorization) are unlikely to help the *numeric*
factorization at BigFloat; structural parallelism (ordering, assembly)
still pays.

## Gap decomposition (per-core, vs single-thread C++)

| Factor | Impact | Evidence |
|--------|-------:|----------|
| GMP vs MPFR arithmetic | **~1.5×** | measured 1.1–1.5× (see above; the earlier ~3× estimate was superseded) |
| Full vs upper-triangle storage | **~2×** | 105M vs 52.6M entries |
| Per-element size (GC overhead) | **~2×** | 112 vs 56 bytes/element |
| Cholesky algorithm | **1×** | identical scalar profile-form |
| OMP parallelism | **1×** | both single-threaded for Cholesky |
| **Combined (per-core)** | **~12×** | 742 s × 12 ≈ 8900 s ≈ observed single-core Julia |

The residual ~20% beyond the decomposition comes from Julia function-call
overhead (ccall dispatch per mpfr_* op), GC scanning of the ~105M BigFloat
objects, and residual bounds checks.

## Historical bug classes (all fixed)

Recorded here because they explain the shape of the current code; the
detailed per-fix history is gone. The classes:

- **BigFloat container aliasing**: `zeros`, `fill!`, and `copyto!` on
  BigFloat containers are *reference* operations in Julia — every
  in-place MPFR mutation through one alias corrupted all aliases. The
  type layer now uses independent elements (`_independent_zeros`,
  `_fill_independent!`) and value copies (`_copyto_independent!`,
  `_copy_independent`) everywhere; see
  [architecture.md](architecture.md) "BigFloat Aliasing".
- **Sparse-profile truncation**: a position-level profile silently drops
  structurally nonzero Schur entries and diverges — block-level profile
  only, with a fatal error on uncovered pairs
  (see [sparse-path.md](sparse-path.md)).
- **Block-index desync**: problem-block indices vs SDP-block positions
  must map through `blockStruct`; counting heuristics desync on empty
  blocks.
- **Shared-accumulator races in `@tasks`**: task closures capturing
  same-named serial-branch locals race on one BigFloat; serial-branch
  accumulators are now distinctly named and bit-identity is tested above
  the parallel threshold.
- **Allocation-driven GC pressure**: `deepcopy` in hot loops (~8 TB of
  garbage per NSPS-class factorization), dense per-constraint LP storage
  (149M objects for 32K nonzeros), and worst-case `sizehint!` (8.4 GB
  pre-allocation) are all eliminated.

## Landed wins (allocation side)

| Item | Result |
|------|--------|
| bmat assembly allocation | **97×** serial (15.5 MB → 160 KB), **99×** threaded (16.5 MB → 166 KB) per iteration |
| Threaded dense scratch | **50×** (10.08 MB → ~203 KB) |
| Triangular path (`get_inv_chol_and_inv!`) | **1.54× wall, 3.8× alloc** (19.49 ms → 12.68 ms; 32.8 MB → 8.6 MB) |
| `min_eigen_lanczos` | wall flat (honestly reported — matvecs are a small fraction of its runtime), alloc **1.6× less** |
| Sparse Cholesky threading | 261.6 s → 38.5 s @8T (6.8×), bit-identical |
| Compact LP storage | 16.5 GB → 461 MB problem data |

## Benchmark status (NSPS)

Both the dense and the forced-sparse paths converge on the NSPS
benchmark problems with **C++-parity iteration counts (24/54) and
optima**; the forced-sparse path runs ~1.6-1.9× faster per iteration
than the dense path and single-threaded C++ (384-417 s/iter vs 625-771
s/iter dense vs ~750 s/iter C++, 256-bit, 8 threads), and peak RSS is
10.3 GB (forced-sparse) vs ~13 GB (dense) after the lazy dense-bmat
allocation. The full dated campaign records live in
`benchmark/nsps/README.md`.

## Performance gates are machine-relative

On the reference machine the dense and sparse paths at Float64 are
within ~±5% of each other and the *direction* flips between runs;
the historical "sparse is 2× faster" claim does not reproduce on
OpenBLAS-based machines, and a fixed speedup gate on small problems is
structurally unreachable. Consequently the test-suite performance gates
are opt-in tripwires (`SDPA_PERF_GATES=1`, one-sided "sparse must not
be materially slower", thresholds configurable), not speedup promises;
absolute comparisons belong in benchmarks, not assertions.

## Remaining gaps vs C++

- **GMP vs MPFR ~1.5× per operation** (measured 1.1-1.5×): requires a
  GMP-backed Julia type — evaluated and deliberately not taken (see
  [roadmap.md](roadmap.md)); the same factor is available from threading
  and ordering.
- **Per-element size ~2×** (GC overhead): architectural, same type
  change required.
- **Full vs profile storage ~2×**: remains for NSPS-class problems,
  because the (correct) block-level density gate sends them to the dense
  path — exactly as C++ does. For genuinely sparse problems the profile
  path avoids both factors.
