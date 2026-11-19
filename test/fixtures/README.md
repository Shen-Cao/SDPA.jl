Source of test fixtures
=======================

This directory holds small SDP problem instances used by the test suite.
They are committed so tests run without depending on the
SDPA-GMP source tree.

Files
-----

`example1.dat-s`, `example1.dat`
  Canonical SDPA smoke-test problem (m=3, nBlock=1, blockStruct=[2]).
  Byte-identical copy of the SDPA-GMP `example1.dat-s` and
  the SDPA-GMP `example1.dat`. Published optimum:
  `objValPrimal = objValDual = -4.19e+01`.

`arch0.dat-s`
  SDPLIB arch0 problem (m=174, n=335). Decompressed from
  SDPLIB `arch0.dat-s.xz`. Published optimum:
  `objValPrimal = 5.6651727321592959e-01`.

`control5.dat-s`
  SDPLIB control5 problem (m=351, n=75). Decompressed from
  SDPLIB `control5.dat-s.xz`. Published optimum:
  `objValPrimal = 1.6883599050955793e+01`.

`theta5.dat-s`
  SDPLIB theta5 problem. Decompressed from
  SDPLIB `theta5.dat-s.xz`. Used in the
  sparse-vs-dense performance comparison.

`param.sdpa`
  C++ SDPA-GMP parameter file (200-bit precision default). Byte-identical
  copy of the SDPA-GMP `param.sdpa`. Used by `test/reference/extract.jl`
  and by manual reference-regeneration workflows.

`SHA256SUMS`
  SHA-256 checksums of each fixture. Verify after any copy with
  `sha256sum -c test/fixtures/SHA256SUMS`.

NSPS benchmark fixtures
-----------------------

The NSPS benchmark set lives under `benchmark/nsps/` together with the
nu = 0.92 problem, warm-start points, parameter files, metadata, and
checksums — see `benchmark/nsps/README.md`.

C++ golden fixtures
-------------------

`golden/` holds the C++ SDPA-GMP 200-bit reference records used by the
test suite (`test/algorithm/test_block_sparse_f1.jl` reads
`golden/block_sparse.json`; the other records are reference data for
the correctness contract). See `golden/README.md` for the generation
methodology. Verify after any copy with
`sha256sum -c SHA256SUMS` run from inside `golden/`.

SDPLIB citation
---------------

Borchers, B., SDPLIB 1.2, A Library of Semidefinite Programming Test
Problems. Optimization Methods and Software. 11(1):683-690, 1999.
