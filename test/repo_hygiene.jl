# test/repo_hygiene.jl
#
# Repo-hygiene guard: no tracked files may live under tmp/.
#
# tmp/ is gitignored scratch space; analysis artifacts that must be
# preserved live under docs/benchmarks/ and benchmark/nsps/ instead.
# This guard fails the suite if anything gets committed under tmp/ again.
#
# The guard is anchored with `git -C <pkgdir>` for cwd-robustness and
# is skipped for packaged/non-git installs (pkgdir(SDPA) has neither a
# .git directory nor a .git worktree pointer file).
@testset "repo hygiene: no tracked files under tmp/" begin
    pkgdir_sdpa = pkgdir(SDPA)
    # A git worktree keeps a .git FILE pointing at the main repo's admin
    # tree; a plain checkout has a .git directory. Both are checkouts.
    is_checkout = isdir(joinpath(pkgdir_sdpa, ".git")) ||
                  isfile(joinpath(pkgdir_sdpa, ".git"))
    if is_checkout
        tracked = read(`git -C $(pkgdir_sdpa) ls-files tmp/`, String)
        @test isempty(tracked)
    else
        @info "not a git checkout — skipping tracked-tmp guard"
        @test_skip "tracked-tmp guard (non-git install)"
    end
end
