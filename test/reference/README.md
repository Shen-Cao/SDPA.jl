# Reference outputs

This directory holds the captured C++ SDPA-GMP outputs used as the
correctness baseline for the Julia port.

## Source binary

- **Name**: `sdpa_gmp` (SDPA-GMP with OMP stub; the OMP variant was
  verified bit-identical to canonical sdpa-gmp).
- **Build location**: the SDPA-GMP binary (untracked, built locally).
- **Build date**: 2026-07-25 (per the `start at` line in each `.result`).
- **Git HEAD of the source tree**: `ca110db`.
- **Parameters**: the SDPA-GMP `param.sdpa` (the canonical 200-bit
  preset; identical to `test/fixtures/param.sdpa`).

## Files

| File | Problem | Iterations | Phase | objVal |
| --- | --- | ---: | --- | --- |
| `example1.dat-s.result` | example1 (m=3, nBlock=1)   | 39 | pdOPT | -4.1900000000000000e+01 |
| `arch0.dat-s.result`    | SDPLIB arch0 (m=174)       | 53 | pdOPT |  5.6651727321592959e-01 |
| `control5.dat-s.result` | SDPLIB control5 (m=351)    | 51 | pdOPT |  1.6883599050955793e+01 |

Each `.result` file has a sidecar `.json` (canonical scalars extracted by
`extract.jl`).

## How to regenerate

The C++ binary is NOT a runtime dependency of the Julia test suite. To
regenerate these references (e.g. after a precision change), run:

```bash
cd <sdpa-gmp-build-dir>
./sdpa_gmp -ds ../../test/fixtures/example1.dat-s \
           -o  ../../test/reference/example1.dat-s.result \
           -p  <sdpa-gmp-source>/param.sdpa
./sdpa_gmp -ds ../../test/fixtures/arch0.dat-s \
           -o  ../../test/reference/arch0.dat-s.result \
           -p  <sdpa-gmp-source>/param.sdpa
./sdpa_gmp -ds ../../test/fixtures/control5.dat-s \
           -o  ../../test/reference/control5.dat-s.result \
           -p  <sdpa-gmp-source>/param.sdpa
julia --project test/reference/extract.jl
```

The `param.sdpa` default is 200-bit precision (≈60 decimal digits). The
Julia solver's default precision is also 200 bits (matching C++ SDPA-GMP),
so direct comparisons against these references are valid to `rtol=1e-30`.
To produce higher-precision references, manually edit `param.sdpa`'s
`precision` line and re-run; the regenerated files replace the committed
ones in a new commit.

## Capture metadata

- **Captured at**: 2026-07-25 (SDPA.jl working tree).
- **Wall times** (200-bit reference run): example1 ≈ 0.013s;
  arch0 ≈ 67.4s; control5 ≈ 74.8s.
