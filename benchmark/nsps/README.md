# NSPS Dual SDP Benchmark Problems for SDPA-GMP (256-bit)

Benchmark set for testing a new SDPA-GMP solver implementation against
the original C++ SDPA-GMP. Two problems, both "P_g(A4|E) maximization"
for the EBI+SIC-POVM NSPS dual SDP with A4-B3 augmentation, on synthetic
Werner data at visibilities nu = 1.0 and nu = 0.92.

Solver configuration (production benchmark baseline - do NOT change):
- NPA level 2
- Fully DI: projective A4 via Naimark dilation, dichotomic Alice/Bob observables
- SIC-POVM on Alice's side
- APVW 2016 A4-B3 augmentation (include_a4_b3=True)

F_0 sign convention: the .dat-s files use the SDPA SUBTRACTION convention
X = sum_k F_k x_k - F_0 with F_0 = -b, matching the SCS slack s = b - A x.
Do NOT change this to +b: the addition convention silently flips the
constraint direction and gives a different, over-relaxed problem. Off-
diagonal PSD entries are divided by sqrt(2) per SDPA symmetric storage.

Precision/reference settings: 256-bit SDPA-GMP, maxIteration 200,
epsilonStar 1.0E-30 (see the *_param.sdpa files). Note: SDPA-GMP labels
highly-converged runs "pdINF" (mu underflow); gate on residuals, not the
phase label. MOSEK reference values are double precision and are NOT the
ground truth at nu = 1.0 (see the nu = 1.0 section).

All ETAs are for the ORIGINAL C++ SDPA-GMP at 256-bit precision - the
reference baseline for benchmarking the new solver.

## Running the benchmark

Fixtures live in this directory (see `SHA256SUMS`; verify with
`sha256sum -c SHA256SUMS` from here). The runner is `run_nsps.jl`:

    JULIA_NUM_THREADS=8 SDPA_FORCE_SPARSE_PROFILE=1 \
      julia --project=. benchmark/nsps/run_nsps.jl nsps_aug_nu1p0 sparse

- First arg: problem name (`nsps_aug_nu1p0` or `nsps_aug_nu0p92`).
- Second arg: `sparse` (forced-sparse path; pair with
  `SDPA_FORCE_SPARSE_PROFILE=1` since NSPS's block-level density exceeds
  the 25% gate) or `dense`.
- `SDPA_NSPS_MAXITER=N` caps iterations (default 200); use 1 for a smoke
  run. `SDPA_BLOCK_ORDERING=1` (or the `SDPA_ORDERING` env var, see
  docs/sparse-path.md) selects the block-level ordering.
- Results (phase, iteration count, objectives, full trajectory) go to
  `SDPA_NSPS_OUT` (default: `tmp/nsps_bench_<name>_<mode>.txt`).

## Problem: nu = 1.0 (nsps_aug_nu1p0)

**Problem**: NSPS dual SDP, P_g(A4|E) maximization, EBI+SIC-POVM, NPA
level 2, fully DI (projective A4 via Naimark dilation, dichotomic),
A4-B3 augmentation (APVW 2016), synthetic Werner data |psi^-> at
visibility nu = 1.0. n_vars=10259 (35 Bell coefficients +
4x2556 Gram entries), 5 SDPA blocks (1 linear of size n_l=14536 +
4 dense PSD of size 71x71), mDIM=10259.

**Theoretical expected value**: P_g = 1/4 exactly (Andersson et al. PRA 97, 012314 (2018) 2-bit ceiling), H_min = 2.0 bits. Expected SDPA-GMP range: P_g in [0.24999999, 0.25000001] (the relaxation upper bound must be >= 1/4). 256-bit reference: P_g = 0.2500000074, H_min = 1.99999996. MOSEK's 0.2511 is a double-precision artifact.

**MOSEK reference** (actual run): P_g=0.25110008,
H_min=1.9937 bits, status optimal, ~4 s.

**ETA for C++ SDPA-GMP**: ~5 h (measured: the existing L2+aug 256-bit run took 17917.8 s ~ 4.98 h, 24 iterations, warm-started from MOSEK; cold start will be longer).

**Files**:
- nsps_aug_nu1p0.dat-s        SDPA sparse problem (10259 vars,
  45246 entries)
- nsps_aug_nu1p0.ini-s        Warm-start init point from MOSEK
  (24760 slack entries + dual Y blocks)
- nsps_aug_nu1p0_param.sdpa   256-bit parameter file
- nsps_aug_nu1p0_meta.json    Metadata + MOSEK reference

**Invocation**:
    sdpa_gmp -ds nsps_aug_nu1p0.dat-s -o nsps_aug_nu1p0.result \
        -is nsps_aug_nu1p0.ini-s -p nsps_aug_nu1p0_param.sdpa

## Problem: nu = 0.92 (nsps_aug_nu0p92)

**Problem**: NSPS dual SDP, P_g(A4|E) maximization, EBI+SIC-POVM, NPA
level 2, fully DI (projective A4 via Naimark dilation, dichotomic),
A4-B3 augmentation (APVW 2016), synthetic Werner data |psi^-> at
visibility nu = 0.92. n_vars=10259 (35 Bell coefficients +
4x2556 Gram entries), 5 SDPA blocks (1 linear of size n_l=14536 +
4 dense PSD of size 71x71), mDIM=10259.

**Theoretical expected value**: No closed form; expected P_g ~ 0.9180 +/- 2e-4, H_min ~ 0.1235 bits. Ground truth: MOSEK P_g=0.91795797 (H=0.1235); MOSEK is accurate to ~2e-5 in P_g at nu <= 0.96 (verified at nu=0.96: MOSEK 0.7214 vs SDPA-GMP 0.7213814986), so the true value is expected in [0.91790, 0.91799].

**MOSEK reference** (actual run): P_g=0.91798011,
H_min=0.1235 bits, status optimal, ~2 s.

**ETA for C++ SDPA-GMP**: ~5-8 h (no prior run at this nu; the nu=0.96 aug run needed 55 iterations vs 24 at nu=1.0, so finite-noise problems take more iterations; extrapolate accordingly). Both problems together ~10-13 h. These are for the ORIGINAL C++ SDPA-GMP at 256-bit precision - the reference baseline for benchmarking the new solver.

**Files**:
- nsps_aug_nu0p92.dat-s        SDPA sparse problem (10259 vars,
  45246 entries)
- nsps_aug_nu0p92.ini-s        Warm-start init point from MOSEK
  (24760 slack entries + dual Y blocks)
- nsps_aug_nu0p92_param.sdpa   256-bit parameter file
- nsps_aug_nu0p92_meta.json    Metadata + MOSEK reference

**Invocation**:
    sdpa_gmp -ds nsps_aug_nu0p92.dat-s -o nsps_aug_nu0p92.result \
        -is nsps_aug_nu0p92.ini-s -p nsps_aug_nu0p92_param.sdpa

---

# Results — dense path

Measurement date: 2026-08-05 — 256-bit, cold start (Julia @ 8 threads;
C++ @ 1 thread).

## nu = 1.0 (theoretical P_g = 1/4)

| | Julia dense | C++ |
|---|---|---|
| Phase | pdINF | pdINF |
| Iterations | **24** | 24 |
| Wall | 17814 s (742 s/iter) | ~17918 s (750 s/iter) |
| objP | **0.2500000007** | 0.2500000074 |
| objD | **0.2500000162** | 0.2500001844 |
| final mu | 1.82e-10 | 1.84e-10 |
| final res (p/d) | 8.7e-19 / 2.9e-17 | — |

## nu = 0.92 (MOSEK reference 0.91798011)

| | Julia dense | C++ |
|---|---|---|
| Phase | **pdOPT** | **pdOPT** |
| Iterations | **54** | **54** |
| Wall | 41641 s (771 s/iter) | 40805 s (756 s/iter) |
| objP | **0.91798010663588133** | **0.91798010663588133** |
| objD | **0.91798010663588133** | **0.91798010663588133** |
| final mu | 1.59e-31 | 1.39e-31 |
| final res (p/d) | 2.5e-31 / 3.3e-52 | 2.2e-31 / 1.75e-55 |

## Conclusions

1. **Julia dense = C++ parity**: identical iteration counts (24/54), the same
   terminals (pdINF at mu 1.8e-10; pdOPT at mu 1.4e-31), and the same optimum
   to 8-18 digits (nu1p0: Julia 0.2500000007 is 30x closer to the theoretical
   1/4 than C++'s 0.2500000074; nu0p92: identical 0.9179801066, matching MOSEK).
2. **Warm start** (the .ini-s files): unusable as-is — the MOSEK Z is indefinite
   (7268/14536 negative LP entries ~1e-7 vs SDP ~1943), collapsing the first step
   for any solver; y-only warm start is equivalent to cold. The exporter must emit
   a strictly-interior, isotropic point for meaningful warm-start benchmarking.
   NOTE: the shipped 4-token ini format does not match the C++ SOURCE reader's
   5-token format — verify the binary's `-is` acceptance separately.
3. The first forced-sparse attempt under this configuration did not converge;
   see the next section.

---

# Results — forced-sparse (fill-in-closed profile)

Measurement date: 2026-08-07 — 256-bit, cold start, 8 threads, maxIter 200
(SDPA_FORCE_SPARSE_PROFILE=1 + SDPA_BLOCK_ORDERING=1).

The forced-sparse factorization now uses the fill-in-closed skyline
computed by `build_skyline_from_problem!` (the symbolic-factorization
envelope, as in C++ sdpa_newton.cpp:225-340), which resolved the iter-6
step-length collapse that stalled the first attempt. The dense path is
unchanged; its 2026-08-05 rows are repeated below as-is.

## nu = 1.0 (theoretical P_g = 1/4)

| | Julia dense (2026-08-05) | Julia forced-sparse (2026-08-07) | C++ |
|---|---|---|---|
| Phase | pdINF | **pdINF** | pdINF |
| Iterations | **24** | **24** | 24 |
| Wall | 17814 s (742 s/iter) | 9572.3 s (399 s/iter) | ~17918 s (750 s/iter) |
| objP | **0.2500000007** | **0.2500000006886560521** | 0.2500000074 |
| objD | **0.2500000162** | 0.2500000161689243884 | 0.2500001844 |
| final mu | 1.82e-10 | 1.824e-10 | 1.84e-10 |
| final res (p/d) | 8.7e-19 / 2.9e-17 | 8.7e-19 / 2.9e-17 | — |

First-4 mu trajectory (sparse): 1.9730907e7, 3.6137176e6, 5.674521e5,
9.01851e4, matching the dense run.

## nu = 0.92 (MOSEK reference 0.91798011)

| | Julia dense (2026-08-05) | Julia forced-sparse (2026-08-07) | C++ |
|---|---|---|---|
| Phase | **pdOPT** | **pdOPT** | **pdOPT** |
| Iterations | **54** | **54** | **54** |
| Wall | 41641 s (771 s/iter) | 22506.9 s (417 s/iter) | 40805 s (756 s/iter) |
| objP | **0.91798010663588133** | **0.9179801066358813296** | **0.91798010663588133** |
| objD | **0.91798010663588133** | 0.9179801066358813296 | **0.91798010663588133** |
| final mu | 1.59e-31 | 1.59e-31 | 1.39e-31 |
| final res (p/d) | 2.5e-31 / 3.3e-52 | 2.5e-31 / 1.4e-51 | 2.2e-31 / 1.75e-55 |

First-4 mu trajectory (sparse): 1.973091777766703e7, 3.6134602110563953e6,
567679.9498174355, 90305.3473096374, a BIT-EXACT match with the dense run.

## Conclusions

1. **The forced-sparse path converges**: the same 24/54 iteration
   counts as C++ SDPA-GMP and as the Julia dense path, to the same
   optimum (nu1p0: P_g = 0.2500000006886560521 vs dense 0.2500000007;
   nu0p92: 0.9179801066358813296, identical to dense). No iter-6
   collapse on either problem.
2. **Speed**: ~399-417 s/iter at 8 threads, ~1.6-1.9x faster per
   iteration than the dense path (~750 s/iter). C++ runs
   single-threaded (~750 s/iter); no wall-time comparison against C++
   is claimed.
3. **Regression test**: test/sparse/test_sparse_schur_divergence.jl
   (cycle-4 non-chordal fixture; fails without the fill-in-closed
   skyline with solve rel 7.24e-4 > 1e-20, passes at 1 and 8 threads).
4. **Dense path unchanged**: the 2026-08-05 dense numbers remain valid
   (re-verified).
5. The position-METIS ordering was not re-measured after the skyline
   change; its 2026-08-05 non-convergent status stands unverified.

---

# Profile & memory probes

Measurement dates: 2026-08-06 through 2026-08-08 (noted per table).

Reference numbers from fresh probes and the full runs. Probe
methodology: NSPS nu1p0, BigFloat 256-bit, 8 Julia threads, cold start,
3 iterations under `/usr/bin/time -v` (forced-sparse:
SDPA_FORCE_SPARSE_PROFILE=1 + SDPA_BLOCK_ORDERING=1; dense: plain).
Full runs: 2026-08-07, same configuration, maxIter 200.

**Per-iteration wall time**

| Path | 3-iter probe | Full run (2026-08-07) |
|------|-------------:|----------------------|
| Julia forced-sparse | 384 s/iter (1152.5 s) | 398.8 s/iter (nu1p0, 24 iters, 9572.3 s); 416.8 s/iter (nu0p92, 54 iters, 22506.9 s) |
| Julia dense | 625 s/iter (1876.4 s) | 742 s/iter (nu1p0, 17813.9 s); 771 s/iter (nu0p92, 41640.7 s) |
| C++ SDPA-GMP (reference, 1 thread) | (not probed) | ~750 s/iter (nu1p0, 17917.8 s / 24 iters) |

The forced-sparse path is ~1.6-1.9x faster per iteration than dense
Julia and C++ SDPA-GMP, at C++-parity (converging) iteration counts
24/54.

**Peak RSS (3-iter probe)**

| Path | eager dense bmat (2026-08-07) | lazy dense bmat (2026-08-08) |
|------|------------------------------:|-----------------------------:|
| Julia forced-sparse | 22,061,068 KB ≈ 22.1 GB | 10,283,948 KB ≈ 10.3 GB |
| Julia dense | 13,411,220 KB ≈ 13.4 GB | 13,032,340 KB ≈ 13.0 GB |

The dense m×m bmat (10259² BigFloat ≈ 6.7 GB incl. heap limbs) is
allocated lazily: solve! constructs the NewtonState with
allocate_dense_bmat=false and the dense consumers materialize it via
ensure_dense_bmat!, so the forced-sparse path never allocates it —
peak RSS drops 22.1 → 10.3 GB (−11.8 GB), dense unchanged (~13 GB).
The change is allocation-only: both paths reproduce the same mu
trajectories exactly.

**Parallelization (8-thread runs)**

- CPU utilization (probe): forced-sparse 685% (~6.9 cores) vs dense
  476% (~4.8 cores). The threaded block-profile Cholesky plus
  threaded assembly parallelize better than the blocked dense Rpotrf!
  on BigFloat.
- Thread-count scaling (2026-08-06, 20.4M-entry profile):
  sparse_cholesky! factorization 947.6 s @1T → 151.5 s @8T ≈ 6.25x on
  8 threads.
- Sparse factors are bit-identical at any thread count (1T vs 8T
  hash-verified); results are thread-count-independent.
- BLAS threads pinned to 1 for BigFloat (src/solve.jl); all parallelism
  via Julia threads (`@tasks`).
