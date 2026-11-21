# Matched-accuracy SDPA-GMP baseline

This opt-in harness compares this SDPA.jl checkout to **unchanged** upstream
[nakatamaho/sdpa-gmp at `ca110db5`](https://github.com/nakatamaho/sdpa-gmp/tree/ca110db5ea1cc46e811b70dfea9cbb25db74448d).
It uses a common SDPA sparse problem, common explicit controls and cold
`lambdaStar*I` start, separate processes on the same CPUs, and an independent
audit of the returned witnesses. There is no C++ dependency at Julia runtime.
The harness has no default dependency downloads or full-suite run.

## Reproduce a bounded first comparison

Requires Linux with `pidfd_open` (kernel 5.3+), Python 3.11+, a C compiler, Julia
1.12+, and the native SDPA-GMP executable with dynamically linked GMP/OpenMP.
For the build below, also install GNU make, GNU patch, tar, GCC/G++, and GMP
development headers/libraries. Prepare Julia before timing:

```sh
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate(); using SDPA'
python3 -B -m unittest discover -s benchmark/parity -v

git clone https://github.com/nakatamaho/sdpa-gmp.git tmp/sdpa-gmp
git -C tmp/sdpa-gmp checkout --detach ca110db5ea1cc46e811b70dfea9cbb25db74448d
python3 -B benchmark/parity/build_cpp.py tmp/sdpa-gmp tmp/gmp-build --jobs 2

python3 -B benchmark/parity/run.py \
  --cpp tmp/gmp-build/sdpa_gmp --cpp-build tmp/gmp-build/build.json \
  --input test/fixtures/example1.dat-s --out tmp/parity-example1 \
  --bits 256 --target 1e-30 --threads 1 --repeat 3 --timeout 120
```

The fallback build follows the pinned `Makefile.am` source list for SDPA-GMP
and MPLAPACK; it builds vendored SPOOLES with exactly the upstream dependency
patches. It uses explicit GCC/OpenMP flags and links the host GMP by default.
Use `--gmp-prefix` with a build of the vendored GMP 6.2.1 or another recorded
installation to choose the runtime library. The compiler, exact commands,
patches and library/binary hashes are emitted in `build.json`. Alternatively,
build with upstream Autotools (`autoreconf --force --install`,
`./configure --enable-openmp=yes`, `make -j2`). A supplied `--cpp` without
a build record is intentionally labeled **unattested**; it can be useful
locally but does not establish a comparison against pinned source. Use only
a trusted local executable. The runner rejects a build record whose SHA or
binary/GMP/OpenMP digests disagree with the run.

`--project` lets you select a separate instantiated project/manifest, but the
Julia worker rejects one that resolves SDPA.jl from a different checkout.
Julia cache/depot setup is outside the timer. Use a new output path per run.
The report copies the project's conventional Julia project and manifest
files, including version-specific manifests, into `julia-environment/`
and records the actually active project per Julia run. Review these files
before publication: local path dependencies may need the same checkout or
path remapping on another host.
For scaling, repeat the same commands at each thread count on an idle host,
with explicit logical CPU IDs where possible (e.g. `--threads 2 --cpus 2,4`
if those IDs are from distinct physical cores). Without `--cpus`, the runner
takes the first N allowed logical CPUs; this is not a topology selection.
Start within one socket, then separately measure socket/SMT effects. The
runner assigns the same CPU mask to both children, sets explicit OpenMP
CPU places on the C++ side, and leaves Julia's own workers free within
that mask. It resets precision/routing-sensitive environment knobs and
limits BLAS and Julia GC workers. It does not claim each solver uses all
requested workers.

The generated `param.sdpa` comes from the first 11 numeric fields of
`--params` (default: `test/fixtures/param.sdpa`). It is passed explicitly via
C++ `-p`. Both solvers get the C++ binary64 representations of the scalar
**controls**; raw problem coefficients are parsed at each solver's working
precision. `--bits` is the requested GMP precision. The runner queries the
actual rounded default in the resolved GMP library (e.g. 200 → 256 on
64-bit limbs) and gives Julia that exact number of MPFR bits unless
`--julia-bits` is specified. This does not equate rounding or algorithms:
GMP `mpf` precision is a minimum, not an MPFR precision/rounding contract.

## What passes

* **Matched accuracy (default):** both solvers must return native `pdOPT`,
  with a complete finite witness, and independently pass the specified
  `--target`. Julia must also report a normal terminal exit. An initial
  iterate that already qualifies can be reported at zero iterations. A small
  objective change, process exit zero, or `pdINF` is not success. This
  narrow first harness measures completed runs verified to meet the target,
  not *first iterate* to meet it. Unsupported status/certificates,
  incomplete output, timeouts and malformed results stay visible as failures.
* **Fixed-iteration diagnostic:** replace the cap with e.g.
  `--fixed-iterations 2`. Require exactly two completed iterations and a
  finite, shaped, symmetric and cone-valid iterate. High residuals or gaps
  are expected and reported, not called optimal. The solvers may choose
  different paths and perform different work per iteration; do not claim
  a kernel speedup or time-to-solution win from these rows. For kernel
  attribution, a separate common-matrix replay is needed.

Both outputs are mapped to the **displayed SDPA convention**: minimize
`b' x` with slack `X = Σ F_i x_i - F_0`; maximize `<F_0,Y>` with
`<F_i,Y> = b_i` and `X,Y` in the PSD/LP cones. C++ natively prints
`xVec=-internal y`, `xMat=internal Z`, and `yMat=internal X`. The Julia
adapter emits that same sign and order. The C++ scalar summary is too
short for verification, so the runner changes only output *formats* in
the parameter file and parses the **timed run's** `xVec/xMat/yMat`.
Precision-dependent formatting leaves generous guard digits; no
untimed re-solve supplies an alternate witness.

The verifier independently reads the original finite decimal tokens.
The two affine residuals are maximum-entry defects scaled by
`1 + max(|F_0|)` and `1 + max(|b|)`, respectively. The absolute objective
gap and `|<X,Y>|` are scaled by `1 + |b'x| + |<F_0,Y>|`; an off-diagonal
coefficient acts on both matrix entries. Comparisons are performed on the
decimal values before rounding displayed ratios. For each SDP or LP block,
check against `target * (1 + max(|F_0|))` for the primal witness and
`target * (1 + max(|b|))` for the dual witness. These are **input-only**
scales: a large unrelated witness entry cannot hide a negative eigenvalue,
negative LP value, or skew. SDP symmetry uses the maximum absolute
entrywise difference from the transpose. The independent Decimal
Cholesky candidate uses a half-tolerance diagonal shift, and passes only
if its exactly recomputed reconstruction error (largest absolute row sum) is
within the remaining half. Diagonal dominance is also a sufficient
certificate. LP blocks check each signed entry.
The candidate guard grows with the magnitude of a block relative to the
shift; exact-decimal and Cholesky checks are capped at 10,000 working
digits. Exhausting that cap is reported as inconclusive, never as an
accepted witness.
These are conservative numerical witness checks, not exact optimality or
infeasibility proofs. The offline verifier is cubic in each nontrivial
PSD block and is intended for bounded benchmark cases, not huge dense
certification runs.

Input support is unambiguous sparse SDP/LP `.dat-s`: separate dimension
lines with optional `= label` (ASCII words, no signs or digits),
column-1 leading comments and mDIM lines at most 255 bytes including the
line ending, nonzero signed blocks, objective vector, and upper-triangle
entries. Duplicates,
truncation, nonfinite values and unsupported shape variants are rejected
rather than silently interpreted differently by the two readers.

## Timing and report

* Common wall time is measured by the same small native monitor from child
  launch through exit. It includes executable startup, Julia loading/JIT,
  problem read, solving, and witness serialization. It excludes the offline
  validator (its cost is recorded separately). Native solver timing fields
  have different boundaries and are diagnostic only. Peak RSS is the
  waited child's Linux maximum, not allocated bytes, a process-tree memory
  sum, or the Python driver's memory. Timeouts retain elapsed deadline
  evidence but not a fabricated completed-run RSS/time.
* `manifest.json` pins input/binary/parameter/harness/environment provenance;
  `results.jsonl` keeps every run in seeded interleaved order; the per-run
  directories retain witness, timing record, and native logs. `summary.json`
  aggregates successful campaigns. A failed pair suppresses the aggregate
  speed ratio; individual rows remain available. Exit codes: 0 all runs
  passed their mode, 1 one or more run failures, 2 setup/input error.
* Smoke runs on `example1` establish integration, not general performance.
  Add `block_sparse`, `arch0`, `control5`, and representative sparse and mixed
  LP/SDP problems separately. Use repeated measurements, matched core and
  memory budgets and multiple machines before claiming parity. The historical
  [NSPS report](../nsps/README.md) uses different scopes and thread counts;
  do not merge those rows with this harness's paired results.
