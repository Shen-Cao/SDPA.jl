"""Build the clean pinned upstream solver using GCC/OpenMP and system GMP.

The ordinary upstream Autotools build may instead be supplied to run.py.
This fallback follows its bundled MPLAPACK source list and bundled SPOOLES
patch list without changing any tracked solver code.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import shlex
import subprocess
import sys

from run import UPSTREAM, binary_info, output, sha, write_json
from witness import Invalid


def build(args):
    src = args.source.resolve(strict=True)
    dst = args.out.resolve()
    commit = output(["git", "rev-parse", "HEAD"], cwd=src)
    if commit != UPSTREAM or output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=src):
        raise Invalid(f"expected clean tracked sources at upstream {UPSTREAM}")
    dst.mkdir(parents=True, exist_ok=False)
    spooles = dst / "spooles"
    spooles.mkdir()
    objects = dst / "objects"
    objects.mkdir()
    spooles_rules = src / "external/spooles/Makefile.am"
    spooles_archive = src / "external/spooles/download/spooles.2.2.tgz"
    patches = re.findall(r"patch -p0 < ../../(patches/\S+)", spooles_rules.read_text())
    if not patches:
        raise Invalid("could not find upstream SPOOLES patches")
    patch_paths = [spooles_rules.parent / p for p in patches]
    rules = (src / "Makefile.am").read_text().replace("\\\n", " ")
    sources = re.findall(r"^sdpa_gmp_SOURCES = (.+)$", rules, re.M)
    if len(sources) != 2:
        raise Invalid("could not find upstream solver source list")
    sources = [src / name for name in re.findall(r"\S+\.cpp", sources[1])]
    target = dst / "sdpa_gmp"
    cflags = "-O2 -funroll-all-loops -std=gnu89 -D_REENTRANT -fcommon " \
             "-Wno-error=implicit-function-declaration -Wno-error=incompatible-pointer-types"
    cxxflags = ["-O2", "-march=native", "-D_REENTRANT", "-DUseMETIS=0", "-fopenmp",
                f"-I{src}", f"-I{src / 'mplapack'}", f"-I{spooles}"]
    ldflags = ["-lpthread", "-lgmpxx", "-lgmp", "-lm"]
    if args.gmp_prefix:
        gmp = args.gmp_prefix.resolve(strict=True)
        gmp_lib = gmp / "lib"
        cxxflags += [f"-I{gmp / 'include'}"]
        ldflags = [f"-L{gmp_lib}", f"-Wl,-rpath,{gmp_lib}", *ldflags]

    commands = []
    # Separate native log for each compile so concurrent GCC output does not interleave.
    def logged(command, cwd, logfile):
        with logfile.open("w", encoding="utf-8") as handle:
            print(shlex.join(command), file=handle, flush=True)
            subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True)

    bootstrap = [
        (["tar", "xzf", str(spooles_archive)], spooles, dst / "extract.log"),
        *[(["patch", "-p0", "-i", str(p)], spooles, dst / f"patch-{i:02d}.log")
          for i, p in enumerate(patch_paths)],
        (["make", "-j1", "-f", "makefile", "global", "CC=gcc", f"CFLAGS={cflags}"],
         spooles, dst / "spooles.log"),
    ]
    for command, cwd, log in bootstrap:
        commands.append(dict(argv=command, cwd=str(cwd), log=str(log)))
        logged(command, cwd, log)
    paths = [objects / (p.stem + ".o") for p in sources]
    if not sources or len(set(paths)) != len(paths):
        raise Invalid("empty source list or duplicate object filenames")
    compilation = [
        (["g++", *cxxflags, "-c", str(src_path), "-o", str(object_path)], src,
         object_path.with_suffix(".log")) for src_path, object_path in zip(sources, paths)
    ]
    commands += [dict(argv=c, cwd=str(d), log=str(l)) for c, d, l in compilation]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for _ in pool.map(lambda item: logged(*item), compilation):
            pass
    link = ["g++", "-fopenmp", "-o", str(target), *map(str, paths),
            str(spooles / "spooles.a"), *ldflags]
    commands.append(dict(argv=link, cwd=str(src), log=str(dst / "link.log")))
    logged(link, src, dst / "link.log")
    if output(["git", "rev-parse", "HEAD"], cwd=src) != commit or output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=src
    ):
        raise Invalid("tracked source changed while building the solver")
    # The runner will verify these digests against the binary and runtime GMP again.
    info = binary_info(target, None, 256, None)
    write_json(dst / "build.json", {
        "upstream_commit": commit,
        "source_path": str(src),
        "recipe": "clean upstream C++/MPLAPACK, bundled SPOOLES with upstream patches, GCC OpenMP",
        "compiler": output(["g++", "--version"]),
        "cc": output(["gcc", "--version"]),
        "spooles_archive_sha256": sha(spooles_archive),
        "spooles_patch_sha256": {p.name: sha(p) for p in patch_paths},
        "binary_sha256": info["sha256"], "gmp_sha256": info["gmp"]["sha256"],
        "gmp": info["gmp"], "openmp": info["openmp"], "dynamic_links": info["dynamic_links"],
        "gmp_prefix": str(args.gmp_prefix) if args.gmp_prefix else None,
        "commands": commands,
    })
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="clean upstream git checkout")
    parser.add_argument("out", type=Path, help="fresh build directory")
    parser.add_argument("--jobs", type=int, default=2, help="parallel C++ compilation only")
    parser.add_argument("--gmp-prefix", type=Path, help="optional GMP installation prefix (lib/ and include/)")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 16:
        parser.error("--jobs must be between 1 and 16")
    try:
        build(args)
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"SDPA-GMP build: {error}. Build logs are under {args.out}.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
