# test/fixtures_checksums.jl
#
# Fixture checksums were documented but never enforced
# — the three SHA256SUMS files were verified only by hand (`sha256sum -c`).
# Several fixtures back only self-referential sparse-vs-dense comparisons,
# so a corrupted-but-parseable fixture silently defangs its test (e.g.
# cycle4_nonchordal becoming chordal trivially passes the fill-closure
# property).
#
# This testset (a) verifies the SHA-256 digest of every file listed in
# test/fixtures/SHA256SUMS, test/fixtures/golden/SHA256SUMS, and
# benchmark/nsps/SHA256SUMS, and (b) enforces completeness: every data
# file of the kinds a SUMS file covers must be listed. A listed file that
# is missing on disk FAILS (naming it via the per-file testset), never
# silently skips.
#
# Paths are @__DIR__-anchored so the checks work under Pkg.test's temp
# sandbox (the full source tree, benchmark/ included, is copied there).
# benchmark/nsps itself may be absent from packaged installs; that case
# skips with a loud @info, mirroring repo_hygiene.jl's non-git skip.

using SHA
using Test

# `<sha256hex>  <filename>` (one or two spaces; filename relative to the
# SUMS file's directory). Blank lines and `#` comment lines are skipped —
# test/fixtures/SHA256SUMS carries an inline comment block.
function _parse_sha256sums(sums_path)
    entries = Tuple{String,String}[]  # (digest, filename)
    for line in eachline(sums_path)
        line = strip(line)
        (isempty(line) || startswith(line, "#")) && continue
        parts = split(line)  # splits on any whitespace run
        length(parts) == 2 ||
            error("malformed SHA256SUMS line in $sums_path: \"$line\"")
        push!(entries, (String(parts[1]), String(parts[2])))
    end
    return entries
end

_digest_matches(path, digest) =
    lowercase(bytes2hex(SHA.sha256(read(path)))) == lowercase(digest)

function _enforce_sums(sums_path, kinds)
    @test isfile(sums_path)
    entries = _parse_sha256sums(sums_path)
    dir = dirname(sums_path)

    # (a) every listed file exists and its digest matches (case-insensitive)
    for (digest, name) in entries
        @testset "$name (listed in $(basename(sums_path)))" begin
            path = joinpath(dir, name)
            @test isfile(path)
            if isfile(path)
                @test occursin(r"^[0-9a-fA-F]{64}$", digest)
                @test _digest_matches(path, digest)
            end
        end
    end

    # (b) completeness: every data file of the covered kinds is listed
    covered = Set(
        name for name in readdir(dir)
        if isfile(joinpath(dir, name)) && any(endswith(name, k) for k in kinds)
    )
    listed = Set(last.(entries))
    @test sort!(collect(covered)) == sort!(collect(listed))
end

@testset "fixture SHA256SUMS enforced" begin
    # File kinds derived from what each SUMS file actually lists.
    _enforce_sums(
        joinpath(@__DIR__, "fixtures", "SHA256SUMS"),
        (".dat-s", ".dat", ".sdpa"),
    )
    _enforce_sums(
        joinpath(@__DIR__, "fixtures", "golden", "SHA256SUMS"),
        (".json",),
    )

    # benchmark/nsps lives outside test/; @__DIR__/.. is the repo root both
    # in-repo and inside the Pkg.test sandbox (the full tree is copied).
    nsps_sums = normpath(
        joinpath(@__DIR__, "..", "benchmark", "nsps", "SHA256SUMS"),
    )
    @testset "benchmark/nsps/SHA256SUMS" begin
        if isfile(nsps_sums)
            _enforce_sums(nsps_sums, (".dat-s", ".ini-s", ".json", ".sdpa"))
        else
            @info "benchmark/nsps/SHA256SUMS absent (packaged install?) — skipping nsps checksum guard"
            @test_skip "nsps checksum guard (no benchmark/nsps tree)"
        end
    end
end
