# Sparse Schur Complement Path

## Overview

The sparse path exploits **aggregate sparsity** in the Schur complement
matrix B (m×m). For problems where constraint matrices {A_i} are sparse,
most entries B[i,j] are structurally zero, and the sparse path avoids
O(m²) storage and O(m³) Cholesky cost.

## When It Activates

Decision in `src/sparse/AggregateSparsity.jl` → `analyze_sparsity`:
with at least 100 constraints, a single packed LP block uses an exact
diagonal profile if every constraint has a nonempty support and no LP
scalar is used in more than one constraint. The dense fallback still
handles empty rows and shared scalars. Otherwise:

```
use_dense = m < 100 || nBlock < 5 || aggregate_density > 0.25
```

| Condition | Dense | Sparse |
|-----------|:-----:|:------:|
| m < 100 | ✓ | |
| nBlock < 5 | ✓ | |
| aggregate density > 25% | ✓ | |
| otherwise | | ✓ |

Override: `solve!(problem, params; force_sparse=true)`.

After `analyze_sparsity`, solve.jl recomputes the density from the
**block-level** skyline (`build_skyline_from_problem!`); if that density
exceeds 25%, the sparse path is rejected even when the first pass said
sparse (see "Skyline Density Check" below).

## Sparsity Analysis: Block-Level vs Position-Level

### Block-Level (current; matches C++ SDPA-GMP)

`build_skyline_from_problem!(problem, ordering, reverse_ordering)`
connects two constraints **iff they both touch the same SDP block**
(regardless of position within the block). This is the connectivity C++
SDPA-GMP uses via `Chordal::makeGraph` (`sdpa_chordal.cpp:141-211`),
which merges the ENTIRE block constraint list per constraint.

Why block-level is the correct connectivity: the F1 formula
B[i,j] = tr(A_i · Z⁻¹ · X · A_j) couples ALL positions within a block,
because X and Z⁻¹ are dense positive-definite throughout the IPM, so
X·A_i·Z⁻¹ is dense within the block. B[i,j] is therefore *structurally*
nonzero whenever i and j share any block, even at different positions.
A position-level profile truncates the Schur matrix — the documented
cause of the NSPS sparse divergence (see "Block-level connectivity"
below).

The block-level skyline is used for BOTH the sparse/dense density gate
and the actual sparse profile:

```julia
block_skyline = build_skyline_from_problem!(problem, ordering, reverse_ordering)
sparse_bmat = _SchurSkyline{T}(block_skyline)
```

`make_aggrigate_index_sdp!` then ERRORS on any block-level pair outside
the profile (an `error` through `_schur_assembly_slot`, mirroring the
C++ "program bug" at `sdpa_newton.cpp:481`). It never silently drops
entries. The explicit-coordinate `profile_from_skyline!` form and its
`diagonalIndex` output remain available on the compatibility path.

### Position-Level (approximation, retained for ordering)

`build_aggregate_graph(problem)` builds an adjacency graph where two
constraints i,j are **connected iff they share the same (block, row, col)
position** in any constraint matrix:

```julia
position_map = Dict{Tuple{Int,Int,Int}, Set{Int}}()
# For each nonzero (blk, r, c) in A_i, mark constraint i at (blk, r, c)
# Two constraints sharing (blk, r, c) → edge in adjacency graph
```

This UNDER-approximates the Schur complement structure (B[i,j] can be
nonzero at positions that never co-occur), so it is NOT used for the
sparse profile. solve.jl keeps it for two purposes only:

1. The fill-reducing ordering (`compute_ordering` on the aggregate
   graph, AMD or METIS).
2. The first-pass aggregate density inside `analyze_sparsity`.

Block-level is the C++-matching structure; position-level is only the
ordering/density approximation.

For NSPS (m=10259, 4 blocks of 71×71, each constraint ~5 nonzeros/block):
- Aggregate (position-level) density: **0.057%** (60K nnz out of 10259²)
- Block-level density: **100%** (all constraints touch all blocks, so
  the profile is the full upper triangle) → dense path, same as C++
  (`sdpa_chordal.cpp:349` gate)

## Ordering Algorithms

The ordering determines the **skyline shape**, which controls:
1. How many profile entries exist (affects memory + Cholesky cost)
2. How tightly the block-level skyline packs (a fill-reducing ordering
   compresses the profile, since every block pair spans the permuted
   positions of its two constraints)
3. Numerical robustness of the Cholesky factorization

The ordering is computed from the position-level aggregate graph
(`analyze_sparsity` → `build_aggregate_graph` → `compute_ordering`);
the block-level profile is then built on top of that ordering.

### Metis (default, matches C++ Spooles)

- **Algorithm**: Nested dissection
- **C++ equivalent**: Spooles ND (nested dissection) / MS (multisection)
- **Properties**: Balances fill-in minimization with numerical stability
- **Julia package**: Metis.jl

### AMD (alternative)

- **Algorithm**: Approximate Minimum Degree
- **Properties**: Aggressively minimizes fill-in → tightest skyline
- **Julia package**: AMD.jl

Measured ordering comparisons (block- vs position-level, at the closed
envelope) are recorded in `benchmark/nsps/README.md`, together with the
`SDPA_ORDERING=auto` routing rationale below.

## Profile-Form Storage

The sparse Schur matrix is stored in **profile form** (also called
"skyline" or "envelope" format):

```
diagonalIndex[j] = start position of column j in sp_ele
sp_ele[diagonalIndex[j] + (i - j)] = B_perm[i, j]   for skyline[j] ≥ i ≥ j
```

Each column j stores entries from the diagonal (row j) up to the skyline
(furthest connected row). Entries outside the skyline are implicitly zero.

Inside `solve!`, the closed profile owns only its numeric values and
the `m + 1` offsets. There are no per-entry row or column index arrays.
This removes two native-word integers per stored value compared with
the explicit representation; a BigFloat value still owns its precision-
dependent backing memory. Other costs, such as the SDP location table and
threaded contribution buffers, are unchanged.

Before allocating values, the solver checks that each column's reach is
within `j:m`, that reaches are nondecreasing (fill-in closure), and that
the one-past-the-end offset fits in an `Int`. Routing counts hypothetical
slots in a widened integer and compares the density thresholds exactly;
it does not allocate offsets or values for an envelope it will reject.
Numeric assembly still
rejects any required SDP or LP pair outside its column interval. The
explicit-coordinate `SDPASparseMatrix`, `profile_from_skyline!`, and the
factor/solve entry points taking explicit offsets remain available for
reference and compatibility.

### Profile-Form Indexing Formula

The checked interval address used by the solver matches
C++ `sdpa_newton.cpp:458-485`:

```
For permuted indices (ii, jj) where ii = min(p_i, p_j), jj = max(p_i, p_j):
    position in sp_ele = diagonalIndex[ii] + (jj - ii)
```

## Sparse Cholesky

`src/sparse/SparseCholesky.jl` implements **blocked left-looking
profile-form Cholesky**, mirroring the blocked structure of the dense
`Rpotrf!`:

```
for each block-column j0 (nb=64):
    factor diagonal+within-block columns (serial chain across block-columns)
    for each later block-column within the source block's skyline reach:
        apply outer-product updates from block-column j0
```

- **Threaded**: serial chain only across block-columns; within a block-column,
  the updates to later columns are independent and run as `@tasks`
  (`@set split=:roundrobin` for load balance on the triangular work). The
  per-entry accumulation order is unchanged, so results are **bit-identical**
  to the scalar algorithm at any thread count.
- **Zero-alloc**: BigFloat updates use `mp_sub_mul!` directly on the
  independent `sp_ele` elements — no `deepcopy`, no per-task scratch.
- **No pivoting**: Fixed column order from fill-reducing ordering
- **Failure mode**: Returns `false` on first non-positive pivot

### Cholesky Failure and Failure Propagation

The dense path uses `Rpotrf_with_adjust!` (the name is historical; there
is no adjustment). It calls plain `Rpotrf!`
and returns the LAPACK-style info directly: 0 on success, non-zero on
failure — exactly mirroring the **active** C++ path (`Lal::getCholesky`,
`sdpa_linear.cpp:352-361`: plain `Rpotrf("Lower", ...)` + return `FAILURE`
on `info != 0`).

The C++ diagonal-adjust algorithm (`choleskyFactorWithAdjust`,
`sdpa_linear.cpp:631-680`) is `#if 0` dead code in the reference
implementation and is deliberately **not** ported.

On failure, `cholesky_schur!` returns `false` and the main loop breaks —
the same failure propagation as the C++ `FAILURE` path ("cannot cholesky
decomposition", "Could you try with smaller gammaStar?").

**The sparse path behaves identically**: `sparse_cholesky!` returns false
immediately on the first non-positive pivot, breaking the main loop. There
is no adjustment fallback on either path.

## Assembly: Location Table

`make_aggrigate_index_sdp!` builds a location table mapping each
(i, j, block_l) triple within the profile to its position in `sp_ele`.

For each SDP block l, it iterates ALL constraint pairs (k1, k2) within
that block and checks:
1. Upper-triangle skip (inz < jnz or tie-break)
2. **Skyline coverage**: `error()` via `_schur_assembly_slot` unless
   `jj <= ii + col_height - 1` (every block-level pair MUST be covered
   by the profile). A pair outside it means the profile was built from a
   position-level skyline (a truncated Schur matrix); C++ treats this as
   a fatal "program bug" (`sdpa_newton.cpp:481`), and so do we. The
   error can never fire once solve.jl builds the profile from the
   block-level skyline.

Pairs passing both checks are stored in 5 parallel arrays:
- `SDP_constraint1[k]`, `SDP_constraint2[k]`: constraint indices i, j
- `blockIndex1[k]`, `blockIndex2[k]`: within-block sparse indices
- `SDP_location_sparse_bMat[k]`: position in sp_ele

**Block-index invariant**: the location table is PROBLEM-block indexed
(`SDP_block_table_len` is sized to `nBlock`; LP blocks keep 0), while the
SDP-block arrays (`xMat.SDP_block`, `invzMat.SDP_block`,
`work.DLS1/DLS2.SDP_block`) are SDP-position indexed. Assembly maps
between the two with `_sdp_blocks_up_to(l, blockStruct)` (the one shared
block-position helper) and asserts both sides (`blockStruct[l] >= 2`,
position in range). Mapping by counting non-empty SDP blocks instead of
by `blockStruct` desyncs on SDP blocks with zero constraints and silently
reads the WRONG X/Z⁻¹ blocks. The threaded variant applies the same
mapping.

**Threaded assembly for LP-mixed problems**: since the block-index
mapping makes the threaded SDP path correct on mixed problems,
`make_bmat!` uses `threaded_compute_bmat_sparse_sdp!` +
`threaded_compute_bmat_sparse_lp!` whenever `Threads.nthreads() > 1`,
for both SDP-only and LP-mixed problems (serial variants at
`nthreads == 1`). The LP assembly threads over LP scalars via `@tasks`,
iterating the compact per-constraint nonzero lists instead of the old
`14536 × m²` zero-skip scan. Multiple LP scalars can contribute to the
same `sp_ele[loc]`, so the threaded LP path uses per-task value buffers
plus a sequential reduction — bit-identical accumulation order.

**Empty-block list handling**: `initialize_index!` (SDPAProblem.jl)
skips zero-nonzero (empty) SDP/LP blocks when building
`SDP_constraint`/`SDP_blockIndex`/`LP_*`, mirroring C++
`SparseLinearSpace` (which keeps only blocks with nonzeros at
construction, `sdpa_struct.cpp:885-912`). The threaded sparse assembly
additionally filters empty location-table blocks (`active_blocks`, since
`@tasks` bodies cannot `continue`).

**Per-(constraint, block) formula**: the F1/F2/F3 selection is
`newton_state.useFormula::Matrix{Symbol}`, indexed `[i, l]` with l the
problem-block index, mirroring C++'s
`useFormula[i * SDP_nBlock + l]`. `compute_formula!` runs once per
problem; F3 is asserted never on dense blocks (C++
`sdpa_newton.cpp:768-774`).

**Cost**: O(nC_l²) per block for the iteration (the upper-triangle rule
plus the O(1) block-pair assert).

**Memory note**: Do NOT use `sizehint!` with worst-case nC_l²/2 entries.
For NSPS this would pre-allocate 8.4 GB. Let Vector grow dynamically; the
entry count is the number of block-level pairs, which for complete block
graphs is the full upper triangle.

## Assembly: Zero-Allocation Pattern

The sparse path assembly uses in-place MPFR operations (same pattern
as the dense path optimization):

| Operation | Old (allocating) | New (zero-alloc) |
|-----------|-----------------|-----------------|
| sp_ele zeroing (serial) | `fill!(sp_ele, zero(T))` | `mp_zero!(sp_ele[idx])` loop |
| sp_ele zeroing (threaded) | `fill!(sp_ele, zero(T))` | per-element `mp_zero!` loop |
| threaded value buffer | `zeros(BigFloat, n)` (aliased) | `_independent_zeros` + `mp_set!` value-copy |
| _sparse_dense_mul writes | `work[r,k] = deepcopy(buf)` | `mp_set!(work[r,k], buf)` |
| sp_ele accumulation | `sp_ele[loc] += value` | `mp_add!(sp_ele[loc], sp_ele[loc], value)` |

**Important**: `fill!` on `Vector{BigFloat}` creates SHARED elements (all
point to same object). Must use `mp_zero!` loop to preserve independence.
After zeroing, `mp_add!` modifies each element in-place — requires
independent elements (guaranteed by `_independent_zeros`; see
[Architecture: BigFloat Aliasing](architecture.md)).

The threaded sparse assembly (`threaded_compute_bmat_sparse_sdp!`) follows
the same discipline: its `sp_ele` zeroing is a per-element `mp_zero!`
loop (a `fill!` would alias every slot to ONE shared BigFloat, and the
sequential reduction mutates `sp_ele[loc]` IN PLACE via `mp_add!`, so
shared elements would accumulate across locations, giving wrong values
not bit-identical to the serial path). Its per-entry value buffer is
`_independent_zeros(T, total_entries)` (plain `zeros(BigFloat, n)` fills
every slot with the same object), and each slot receives the computed
value via `mp_set!` (a value-copy into an independent element that
replaces `deepcopy`, no GC churn). The dense paths (`compute_bmat_dense!`
and `threaded_compute_bmat_dense!`) likewise zero in-place with `mp_zero!`
and accumulate with in-place `mp_add!`.

## Dense vs Sparse Path Comparison

| Aspect | Dense | Sparse |
|--------|-------|--------|
| Storage | m² entries | profile_nnz entries |
| Cholesky | O(m³/3), blocked + parallel (Rpotrf!) | O(profile_fill), blocked + parallel |
| Assembly | All (i,j) pairs, threaded | Only skyline pairs, threaded |
| Cholesky failure | Rpotrf! info != 0 → main-loop break | First non-positive pivot → main-loop break |
| Parallelism | @tasks in Rgemm! | @tasks in blocked Cholesky + assembly |

Both paths converge on the NSPS benchmark problems with C++-parity
iteration counts; without forcing, NSPS takes the dense path (the
fill-in-closed envelope exceeds the 25% gate). For absolute numbers and
dated verdicts: see `benchmark/nsps/README.md`.

## Ordering and the SDPA_ORDERING env var

Because the fill-in-closed envelope is ~full for both orderings on NSPS,
`SDPA_ORDERING=auto` cannot route on the block-level density; it routes
on the POSITION-level closed-profile density (the fill-in-closed
envelope computed with the position ordering, hoisted before any
block-graph work): **block iff position-level closed-profile density >
0.4**. This is a repo-measured assumption with C++ `extend_threshold =
0.4` as a loose precedent — NOT verified parity. The mid-density
crossover is unvalidated (the two observed regimes — NSPS ~100% vs
sparse problems ~0.03% — are fully separated); the constant is
adjustable only with recorded measurement evidence.

`SDPA_ORDERING` env var:

- unset → current default behavior exactly (position-METIS unless
  `SDPA_BLOCK_ORDERING=1`); verified bit-identical to the default
  routing (full trajectory rows match).
- `SDPA_ORDERING=block` → force the block-level ordering (identical to
  `SDPA_BLOCK_ORDERING=1`).
- `SDPA_ORDERING=auto` → route per the pinned density rule above; the
  block graph + METIS ordering are built lazily, only when routed to
  block (the 26M-edge block-graph cost is never paid on sparse
  problems). Verified on NSPS: both benchmark problems converge with the
  same 24/54 iteration counts as `SDPA_BLOCK_ORDERING=1`, to
  bit-identical optima.
- Precedence: `SDPA_ORDERING` wins if both are set;
  `SDPA_BLOCK_ORDERING=1` + `SDPA_ORDERING=auto` → auto;
  `SDPA_BLOCK_ORDERING=1` alone behaves as `SDPA_ORDERING=block`. The
  `force_sparse=true` override (position-METIS when the aggregate graph
  is dense) applies with the same precedence to both env names, and the
  dense/sparse gate is unchanged. Any other `SDPA_ORDERING` value is an
  error (`SDPA_ORDERING must be 'block' or 'auto'`).

## Memory Architecture

### Dense bmat (lazy)

`NewtonState` used to always create `SDPADenseMatrix{T}(m, m)` for
`bmat`, regardless of sparse/dense path selection — for NSPS at BF(200):
10259² × ~112 bytes ≈ **11.8 GB** unused on the sparse path, which
dominated sparse-path RSS. The dense bmat is now lazy: `solve!`
constructs the state with `allocate_dense_bmat=false` and the dense-only
consumers (`compute_bmat_dense!`, `threaded_compute_bmat_dense!`,
`cholesky_schur!`) materialize it on demand via `ensure_dense_bmat!`, so
the forced-sparse path never allocates it (measured peak RSS drop
22.1 → 10.3 GB at NSPS 256-bit; see `benchmark/nsps/README.md`).

### Compact problem-data storage

`SDPASparseLinearSpace` stores LP problem-data compactly: per
constraint, only the nonzero LP entries (`LP_nnz`, `LP_sp_index`,
`LP_sp_val`), mirroring C++ `SparseLinearSpace`. A dense per-constraint
`LP_sp_block::Vector{T}` of length `LP_nBlock` would materialize 149M
BigFloat objects (16.5 GB) for NSPS's 32,536 actual LP nonzeros.

## Debugging Guide

### "Solve produces no output"

Check `flush(io)` after `@printf` in `solve.jl`. Julia buffers stdout
when redirected to file. Without `flush(io)`, iteration output is stuck
in the buffer.

### "Skyline density 100% → redirects to dense"

Correct behavior: the density gate uses the block-level skyline, and
problems where all constraints touch all blocks (NSPS, theta5, example1)
genuinely need the dense path; C++ SDPA-GMP takes it too
(`sdpa_chordal.cpp:349`). If you expected the sparse path, check whether
the problem really has complete block connectivity.

### "Cholesky fails after a few iterations"

The Schur matrix B can become genuinely indefinite as the iterate moves
(the profile is no longer truncated, so a failure is a real non-PD
pivot, not an approximation artifact). Try:
1. Switch ordering: `method=:metis` (nested dissection, more robust)
2. Reduce `gammaStar` (C++ parity: both dense and sparse paths return
   failure immediately on a non-positive pivot; there is no adjustment
   fallback)

A `block-level pair (ii=..., jj=...) outside block-level profile`
error instead means the profile was NOT built from the block-level
skyline (programming error; C++ "program bug" parity at
`sdpa_newton.cpp:481`).

### "Memory grows to 40+ GB"

All known 40 GB+ causes are fixed. If memory still grows abnormally:
1. Check that LP storage is compact (per-constraint nonzero lists, not a
   dense LP block per constraint).
2. Check `push!` loops building large index vectors without `sizehint!`
   (the block-graph and location-table builders pre-size exactly).
3. Check for `deepcopy` in BigFloat hot loops (any new one is a bug).

### "Process runs for hours with no output"

Two causes:
1. stdout buffering (fix: `flush(io)` after @printf)
2. Dense path selected for large m (check skyline density < 25%)

## Skyline Density Check

The sparse/dense decision uses the **block-level skyline density**:
solve.jl computes `block_skyline` via `build_skyline_from_problem!` and
redirects to the dense path when its profile density exceeds 25%,
matching C++ `Chordal::ordering_bMat` (`sdpa_chordal.cpp:348-365`). This
is broader than the position-level aggregate density from
`analyze_sparsity` (the first pass), because F1/F2/F3 couple all
constraints within a block. Problems with complete block connectivity
(NSPS, theta5, example1) therefore take the dense path, identical to C++
SDPA-GMP.

The `force_sparse=true` option bypasses the initial `analyze_sparsity`
decision but NOT the block-level density check. To truly force sparse
(the `SDPA_FORCE_SPARSE_PROFILE=1` bench env var), the density check is
bypassed — with the threaded blocked Cholesky, a dense profile is no
longer slower than the dense path's blocked Cholesky, so forcing sparse
on NSPS-class problems is viable for benchmarking (measured numbers:
`benchmark/nsps/README.md`). A truncated position-level profile still
diverges — the block-level profile is mandatory.

## References

- C++ source: `sdpa_newton.cpp` (location table), `sdpa_chordal.cpp`
  (aggregate graph + ordering)
- C++ sparsity: `Chordal::makeGraph`, `Chordal::ordering_bMat` (Spooles
  ND/MS/MMD)
- Julia implementation: `src/sparse/AggregateSparsity.jl`,
  `src/sparse/SparseCholesky.jl`
- Measured campaigns: `benchmark/nsps/README.md`
