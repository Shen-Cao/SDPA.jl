# SDPA.jl C++ Golden Fixtures

Generated 2026-08-02 02:51:59 against C++ SDPA-GMP
(a local scratch build, byte-identical to upstream nakatamaho/sdpa-gmp),
Julia 1.12.6.

## Invocation

```bash
cp test/fixtures/param.sdpa <scratch>/   # binary reads ./param.sdpa from CWD
cd <scratch> && sdpa_gmp <fixture>.dat-s <fixture>.result -pt 0
```

`-pt 0` = default parameter file `param.sdpa` (CWD): precision 200 bits,
epsilonStar = 1e-30, epsilonDash = 1e-30, maxIteration = 200 — identical to Julia
`Parameters{T}(precision=200)` defaults.

## Precision semantics

- All records are `-pt 0` (200-bit MPFR in C++; `setprecision(BigFloat, 200)` in Julia).
- The C++ `.result` summary prints `objValPrimal`/`objValDual` at 16 significant
  digits only. For full precision, `xVec`/`xMat`/`yMat` print 40-digit values; the
  golden records use `objP = c'x` (from `xVec`) and `objD = c'x - tr(X*Y)` (from
  `xMat`/`yMat`), computed at BigFloat(200).
- `julia_bigfloat.rtol_1e30_verdict_primal` compares Julia BigFloat(200)
  `final_obj_primal` vs the full-precision C++ value at rtol 1e-30 (dual likewise).
- `julia_float64.rtol_1e10_verdict_primal` compares Julia Float64 vs the same C++
  value at rtol 1e-10. A Float64 record marked `informational_only = true` did NOT
  pass (objective rtol and/or phase match). Tolerances
  are NOT weakened.

## Julia invocation note (threads)

All Julia solves here ran with `JULIA_NUM_THREADS=1` (the C++ ground truth is
single-threaded anyway). The solve API used is exactly
`SDPA.solve!(problem, SDPA.Parameters{T}(precision=200); verbose=false)`.

## Fixtures

The retained C++ golden record (only `block_sparse.json` is loaded by the
test suite, `test/algorithm/test_block_sparse_f1.jl`; the archived
`arch0.json`/`control5.json`/`example1.json`/`minimal_sdp_lp.json` records
were reference data for the correctness contract and are not shipped):

| fixture | m | nBlock / blockStruct | C++ phase | C++ iters | C++ objP (full) |
|---|---|---|---|---|---|
| block_sparse.dat-s | 10 | 3 / [10,10,10] | pdOPT | 38 | 165.0000000000000000000000000001553506303907999999999999999999999999999999999 |

## Block-sparse fixture

`test/fixtures/block_sparse.dat-s` is a synthetic fixture: the
position-level sparsity graph differs from the block-level graph. m = 10, nBlock = 3,
blockStruct = [10, 10, 10]. Constraint k (1..10) touches exactly one scattered diagonal
position per block: (k,k) in block 1, ((k mod 10)+1, ...) in block 2, and
((3k-2 mod 10)+1, ...) in block 3. Every constraint touches every block, so the
block-level sparsity graph is complete; no two constraints share a position, so the
position-level graph is disconnected (Schur complement structurally diagonal).
Constructed with X* = 0, Y* = I, x* = (1..10): optimal value c'x* = 3*55 = 165,
verified by C++ (pdOPT, 38 iters) and Julia BigFloat(200) (pdOPT, 38 iters).
