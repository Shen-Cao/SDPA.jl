# Measuring one-time Schur planning

This is a **Julia-to-Julia** comparison, separate from the C++ parity harness.
It times the exact formula selection and sparse SDP location-table build in
isolation, then reports warmed full-solver calls on small fixtures. It does
not cap a large solve, run NSPS, or time any large factorization.

From this checkout, run the *same script* in a separate Julia process for
each tested revision. First instantiate each project; on an uncontended
Linux host use the same CPU mask for both processes (choose one allowed CPU).
Pass the checkout commit as a label and confirm `context` names the intended
SDPA source. Leave sparse-routing and ordering overrides unset.

```sh
script="$PWD/benchmark/bench_schur_planning.jl"
repo=/path/to/checkout-under-test
julia --project="$repo" -e 'using Pkg; Pkg.instantiate()'
unset SDPA_ORDERING SDPA_BLOCK_ORDERING SDPA_FORCE_SPARSE_PROFILE SDPA_SCHUR_BACKEND
SDPA_PLANNING_REVISION="$(git -C "$repo" rev-parse HEAD)" \
OPENBLAS_NUM_THREADS=1 taskset -c 0 julia --startup-file=no --gcthreads=1 \
  --threads=1,0 --project="$repo" "$script" > planning.tsv
```

- `SDPA_PLANNING_REPEATS`: independent measured samples in each process
  (default 5), after one warm-up call per case.
- `SDPA_PLANNING_MODE`: `all` (default), `plan` (formula and table), or
  `solves`.
- `SDPA_PLANNING_TYPE`: `all` (default), `f64`, or `bf` (256-bit BigFloat).
  The table-only rows use Float64 storage: the five record arrays are
  integer metadata, and no numeric assembly or factorization is timed.

The `trial` rows report wall seconds, total allocated bytes (not peak RSS)
and GC seconds **per call**. The smallest calls are timed in batches.
Input generation and reads, Newton state construction, and the sparse
numeric profile allocation are outside the planning timers. Each table
sample starts with a fresh empty table. The `solve` rows measure a normal,
in-process `solve!` call on the already-read problem, including it doing
its own planning, factorization and iterations. They exclude startup,
method warm-up and parsing. Check the `formula_digest` and `pairs_digest`
rows match; `solve_digest` covers exit, objectives, residuals, every
trajectory entry and the returned iterates. The separately collected
`route` lines check that solve comparisons stayed on the same backend.

The large formula-only synthetic cases have one group of 4,096 or two
groups of 2,556 incidences and variable integer weights. The latter borrows
a dimension from the NSPS fixture metadata (2,556 Gram entries per block,
PSD size 71), not its matrix values or the full workload. These cases
allocate no dense Schur complement. The table-only cases build smaller
bounded one/two-group profiles; the default sparse full-solve check is a
six-group, 108-constraint SDP. The two other full solves are the
`example1` and `lp_first_mixed` fixtures.

Compare distributions or process medians, not minima or one
microsecond-scale sample. Alternate checkout order across independent
processes before claiming a performance effect. For the sort, report both
time and temporary storage on unsorted weights; already ordered keys need
no sort buffer. The exact pair count removes the duplicate quadratic
**counting** scan, not the existing quadratic emission or its table memory.
A setup win is not a claim of full-solve speedup, smaller Schur storage, or
a reason to pick a new backend.
