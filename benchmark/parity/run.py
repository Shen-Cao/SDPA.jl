"""Paired fresh-process SDPA-GMP / SDPA.jl benchmark (Linux, Python stdlib)."""
import argparse
import ctypes
from decimal import DecimalException
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import select
import shutil
import signal
import statistics
import subprocess
import sys
import time

from witness import Invalid, decimal, read_problem, read_witness, validate


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
UPSTREAM = "ca110db5ea1cc46e811b70dfea9cbb25db74448d"
CONTROLS = ("maxIteration", "epsilonStar", "lambdaStar", "omegaStar", "lowerBound",
            "upperBound", "betaStar", "betaBar", "gammaStar", "epsilonDash", "precision")
PROGRESS_PHASES = {"noINFO", "pFEAS", "dFEAS", "pdFEAS", "pdOPT"}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def output(command, **kwargs):
    return subprocess.run(command, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=True, **kwargs).stdout.strip()


def path_info(path):
    return {"path": str(path), "sha256": sha(path)}


def write_json(path, item):
    path.write_text(json.dumps(item, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def binary_info(cpp, build, requested, env):
    links = output(["ldd", str(cpp)], env=env, timeout=10)
    if "not found" in links:
        raise Invalid("a C++ library failed to resolve; inspect ldd")
    dependencies = {}
    for line in links.splitlines():
        match = re.match(r"\s*(lib\S+)\s+=>\s+(/\S+)\s+\(", line)
        if match:
            dependencies[match[1]] = Path(match[2]).resolve()
    gmp = next((p for lib, p in dependencies.items() if lib.startswith("libgmp.so.")), None)
    omp = next((p for lib, p in dependencies.items()
                if lib.startswith(("libgomp.so.", "libomp.so.", "libiomp5.so"))), None)
    if gmp is None or omp is None:
        raise Invalid("need a dynamically linked GMP and OpenMP runtime; inspect ldd")
    library = ctypes.CDLL(str(gmp))
    get_prec = getattr(library, "__gmpf_get_default_prec")
    get_prec.argtypes, get_prec.restype = [], ctypes.c_ulong
    set_prec = getattr(library, "__gmpf_set_default_prec")
    set_prec.argtypes, set_prec.restype = [ctypes.c_ulong], None
    original = get_prec()
    try:
        set_prec(requested)
        effective = get_prec()
    finally:
        set_prec(original)
    info = path_info(cpp) | {
        "dynamic_links": links,
        "gmp": path_info(gmp) | {
            "version": ctypes.c_char_p.in_dll(library, "__gmp_version").value.decode("ascii"),
            "limb_bits": ctypes.c_int.in_dll(library, "__gmp_bits_per_limb").value,
            "requested_bits": requested,
            "effective_default_bits": effective,
        },
        "openmp": path_info(omp),
        "source": "unattested caller-supplied binary",
    }
    if build is not None:
        record = json.loads(build.read_text(encoding="utf-8"))
        if record.get("upstream_commit") != UPSTREAM or record.get("binary_sha256") != info["sha256"]:
            raise Invalid("C++ build record does not match the pinned source and supplied binary")
        if record.get("gmp_sha256") != info["gmp"]["sha256"]:
            raise Invalid("C++ build record does not match the GMP library resolved for this run")
        if record.get("openmp", {}).get("sha256") != info["openmp"]["sha256"]:
            raise Invalid("C++ build record does not match the OpenMP library resolved for this run")
        info["source"] = "pinned source; supplied build record and binary/library hashes agree"
        info["build_record"] = record
    return info


def parameter_text(template, requested_bits, iteration_override, digits):
    rows = [line.split()[0] for line in template.read_text(encoding="ascii").splitlines()
            if line.strip()]
    if len(rows) < len(CONTROLS):
        raise Invalid("parameter template needs all 11 native numeric controls")
    rows = rows[:len(CONTROLS)]
    if not rows[0].isdigit() or not rows[10].isdigit():
        raise Invalid("parameter maxIteration and precision must be integers")
    rows[0] = str(iteration_override if iteration_override is not None else int(rows[0]))
    rows[10] = str(requested_bits)
    if not 1 <= int(rows[0]) <= 2**31 - 1:
        raise Invalid("maxIteration outside the positive C++ int range")
    controls = []
    for token in rows[1:10]:
        original = decimal(token)
        value = float(token)
        if not math.isfinite(value) or (value == 0 and original != 0):
            raise Invalid(f"parameter underflows or overflows the upstream double: {token}")
        controls.append(value)
    eps, initial, omega, lower, upper, beta, beta_bar, gamma, eps_dash = controls
    if not (eps > 0 and initial > 0 and omega > 1 and lower < upper
            and 0 < beta <= beta_bar < 1 and 0 < gamma < 1 and eps_dash > 0):
        raise Invalid("native numeric controls are outside SDPA.jl's supported ranges")
    rendered = "".join(f"{value}  {key}\n" for value, key in zip(rows, CONTROLS))
    rendered += "".join(f"%+.{digits}Fe  {field}\n" for field in ("xPrint", "XPrint", "YPrint", "infPrint"))
    return dict(zip(CONTROLS, rows)), rendered


def child_environment(threads):
    # Do not inherit experimental Julia routes or OpenMP/BLAS affinity,
    # scheduling, nested pools, or thread counts from the invoking shell.
    prefixes = ("SDPA_", "OMP_", "GOMP_", "KMP_", "MKL_", "OPENBLAS_", "VECLIB_")
    ignored = {"JULIA_NUM_THREADS", "JULIA_NUM_GC_THREADS", "JULIA_EXCLUSIVE", "JULIA_DEBUG",
               "LD_PRELOAD", "LD_AUDIT"}
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(prefixes) and key not in ignored}
    overrides = {
        "LC_ALL": "C", "LANG": "C", "OMP_NUM_THREADS": str(threads), "OMP_DYNAMIC": "FALSE",
        "OMP_MAX_ACTIVE_LEVELS": "1", "OMP_PROC_BIND": "TRUE", "OMP_PLACES": "threads",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "BLAS_NUM_THREADS": "1",
        "JULIA_LOAD_PATH": "@:@stdlib",
    }
    return env | overrides, overrides


def run_child(measure, command, directory, env, cpus, timeout):
    """Wait for a small native monitor to reap the target and report usage."""
    affinity = os.sched_getaffinity(0)
    report = directory / "usage.json"
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        os.sched_setaffinity(0, cpus)
        try:
            start = time.monotonic_ns()
            proc = subprocess.Popen([str(measure), str(report), *command],
                                    cwd=directory, env=env, stdin=subprocess.DEVNULL,
                                    stdout=stdout, stderr=stderr, start_new_session=True)
        finally:
            os.sched_setaffinity(0, affinity)
        timed_out = False
        pidfd = None
        try:
            deadline = start + int(timeout * 1e9)
            pidfd = os.pidfd_open(proc.pid)
            poller = select.poll()
            poller.register(pidfd, select.POLLIN)
            # Poll detects exit without consuming the wait4 resource usage,
            # and avoids an alarm racing with a successful reaping.
            while not poller.poll(max(0, min(60_000, math.ceil((deadline - time.monotonic_ns()) / 1e6)))):
                if time.monotonic_ns() >= deadline:
                    timed_out = True
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    break
            proc.wait()
            elapsed = (time.monotonic_ns() - start) / 1e9
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise
        finally:
            if pidfd is not None:
                os.close(pidfd)
    result = {"external_elapsed_seconds": elapsed, "timed_out": timed_out}
    if timed_out:
        # The supervisor was killed before it could reap and report the
        # true child's RSS. Do not substitute the Python or supervisor RSS.
        return result | {"wall_seconds": None, "peak_rss_bytes": None, "exit_code": None}
    if proc.returncode != 0:
        raise OSError(f"measurement supervisor exited with {proc.returncode}")
    measurements = json.loads(report.read_text(encoding="utf-8"))
    if measurements["finished_monotonic_ns"] > deadline:
        # pidfd readiness alone only means "already finished"; a delayed
        # Python observer may wake after a late completion. Compare the
        # native monitor's actual elapsed interval to the same deadline.
        return result | {"timed_out": True, "timeout_detail": "late completion",
                         "wall_seconds": None, "peak_rss_bytes": None, "exit_code": None}
    return result | measurements


def julia_route(stderr):
    # 'Using sparse Schur complement path' is tentative: a later density
    # veto can still choose dense. Do not treat that first message as final.
    text = stderr.read_text(encoding="utf-8")
    if "Using dense Schur complement path" in text or "Block-level density >25%" in text:
        return "dense"
    if "Block-level profile:" in text or "FORCED sparse profile" in text:
        return "sparse"
    return "unobserved"


def assess(row, path, problem, args, actual_bits):
    if row["timed_out"]:
        row["status"] = "timeout"
        return
    if row["exit_code"] != 0:
        row["status"] = "process_error"
        return
    solver = row["solver"]
    validation_start = time.monotonic_ns()
    try:
        witness = read_witness(path, problem, solver)
        row["phase"] = witness.phase
        row["iterations"] = witness.iterations
        row["reported"] = witness.meta
        if solver == "julia":
            witness.meta["reported_route"] = julia_route(path.parent / "stderr.log")
            if int(witness.meta["threads"]) != args.threads:
                raise Invalid("Julia process did not report the configured thread count")
        expected = args.bits if solver == "cpp" else actual_bits
        if int(witness.meta["precision"]) != expected:
            raise Invalid("worker precision disagrees with the configured precision")
        row["witness_sha256"] = sha(path)
        row["validation"] = validate(problem, witness, args.target, args.guard_digits)
        exit_reason = witness.meta.get("exit reason")
        if args.fixed_iterations is not None:
            if witness.iterations != args.fixed_iterations or witness.phase not in PROGRESS_PHASES:
                row["status"] = "incomplete_iterations"
            elif solver == "julia" and exit_reason != (
                "OPTIMAL_TERMINAL" if witness.phase == "pdOPT" else "MAX_ITERATION"
            ):
                row["status"] = "solver_status"
            elif not all(row["validation"][k] for k in
                         ("primal_symmetric", "dual_symmetric", "primal_cone", "dual_cone")):
                row["status"] = "invalid_iterate"
            else:
                row["status"] = "diagnostic_complete"
        elif (
            witness.phase != "pdOPT" or not 0 < witness.iterations <= args.iteration_cap
            or (solver == "julia" and exit_reason != "OPTIMAL_TERMINAL")
        ):
            row["status"] = "solver_status"
        elif not row["validation"]["accuracy_passed"]:
            row["status"] = "accuracy_failed"
        else:
            row["status"] = "verified"
    except (DecimalException, OSError, UnicodeError, ValueError) as error:
        row.update(status="invalid_result", detail=str(error))
    finally:
        row["validation_wall_seconds"] = (time.monotonic_ns() - validation_start) / 1e9


def positive(text):
    if not text.isdecimal() or int(text) <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return int(text)


def tolerance_arg(text):
    try:
        value = decimal(text)
    except (DecimalException, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError("expected a finite decimal in (0, 1)")
    return value


def cpu_set(value, threads):
    allowed = os.sched_getaffinity(0)
    if value is None:
        cpus = sorted(allowed)[:threads]
    else:
        try:
            cpus = [int(token) for token in value.split(",")]
        except ValueError as error:
            raise Invalid("--cpus needs comma-separated logical CPU IDs") from error
    if len(cpus) != threads or len(set(cpus)) != threads or not set(cpus) <= allowed:
        raise Invalid("choose one distinct allowed logical CPU per requested thread")
    return cpus


def options(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpp", type=Path, required=True, help="trusted local OpenMP SDPA-GMP binary")
    p.add_argument("--cpp-build", type=Path, help="optional build manifest from build_cpp.py")
    p.add_argument("--julia", default="julia", help="Julia executable")
    p.add_argument("--cc", default="cc", help="compiler for the small Linux measurement monitor")
    p.add_argument("--project", type=Path, default=REPO, help="instantiated project loading this checkout")
    p.add_argument("--input", type=Path, required=True, help="one common .dat-s file")
    p.add_argument("--params", type=Path, default=REPO / "test/fixtures/param.sdpa")
    p.add_argument("--out", type=Path, required=True, help="fresh campaign directory")
    p.add_argument("--threads", type=positive, default=1)
    p.add_argument("--cpus", help="exact comma-separated logical CPUs; default first allowed IDs")
    p.add_argument("--bits", type=positive, default=256, help="GMP-requested bits")
    p.add_argument("--julia-bits", type=positive, help="override Julia bits (default actual GMP bits)")
    p.add_argument("--target", type=tolerance_arg, default="1e-30", help="independent output threshold")
    p.add_argument("--repeat", type=positive, default=3)
    p.add_argument("--seed", type=int, default=20261121, help="paired execution order seed")
    p.add_argument("--timeout", type=float, default=300.0, help="external wall-clock seconds per child")
    work = p.add_mutually_exclusive_group()
    work.add_argument("--max-iter", type=positive, help="override template cap for matched accuracy")
    work.add_argument("--fixed-iterations", type=positive, help="diagnostic only; not a speed-to-solution claim")
    args = p.parse_args(argv)
    if not 64 <= args.bits <= 4096 or not 64 <= (args.julia_bits or args.bits) <= 4096:
        p.error("use 64–4096 arithmetic bits (enough to represent binary64 controls)")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 604800:
        p.error("--timeout must be positive and at most seven days per child")
    return args


def optional_text(path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def campaign(args):
    if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
        raise Invalid("this runner requires Linux with pidfd, wait4, and CPU affinity")
    if args.input.suffix != ".dat-s":
        raise Invalid("expected an input .dat-s file")
    cpp = args.cpp.resolve(strict=True)
    julia_path = shutil.which(args.julia)
    if julia_path is None:
        raise Invalid(f"Julia executable not found: {args.julia}")
    julia = Path(julia_path).resolve()
    cpus = cpu_set(args.cpus, args.threads)
    env, overrides = child_environment(args.threads)
    cpp_info = binary_info(cpp, args.cpp_build, args.bits, env)
    cpp_bits = cpp_info["gmp"]["effective_default_bits"]
    actual_bits = args.julia_bits or cpp_bits
    digits = len(str(2 ** max(cpp_bits, actual_bits))) + 16
    args.guard_digits = max(80, digits + 32, -args.target.adjusted() + 40)
    if args.guard_digits > 10_000:
        raise Invalid("accuracy target exceeds the verifier's guard limit")
    controls, params = parameter_text(args.params, args.bits, args.fixed_iterations or args.max_iter, digits)
    args.iteration_cap = int(controls["maxIteration"])
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    common_input, common_params = out / "input.dat-s", out / "param.sdpa"
    shutil.copyfile(args.input, common_input)
    common_params.write_text(params, encoding="ascii")
    problem = read_problem(common_input)
    measure = out / "measure_child"
    cc = shutil.which(args.cc)
    if cc is None:
        raise Invalid(f"C compiler not found: {args.cc}")
    measure_sources = HERE / "measure_child.c"
    measure_command = [cc, "-O2", "-std=c11", "-Wall", "-Wextra",
                       str(measure_sources), "-o", str(measure)]
    with (out / "measure_build.log").open("wb") as log:
        subprocess.run(measure_command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=30)

    project = args.project.resolve(strict=True)
    manifests = sorted(project.glob("Manifest*.toml"))
    mode = "fixed_iterations" if args.fixed_iterations is not None else "matched_accuracy"
    cpuinfo = optional_text(Path("/proc/cpuinfo"))
    hardware = None if cpuinfo is None else re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
    manifest = {
        "schema_version": 1, "mode": mode, "accuracy_target": str(args.target),
        "verify_guard_digits": args.guard_digits, "start": "cold default lambdaStar*I",
        "timing": "fresh child wall until exit, including startup/JIT/read/solve/witness serialization; validation excluded",
        "rss": "native monitor's wait4 ru_maxrss in bytes, not aggregate process-tree or Python-launcher RSS",
        "input": path_info(common_input) | {"original_path": str(args.input.resolve()),
                                          "m": len(problem.b), "blocks": problem.blocks},
        "parameters": path_info(common_params) | {
            "numeric_fields": controls, "witness_decimal_places": digits,
            "floating_controls_binary64_hex": {name: float(controls[name]).hex() for name in CONTROLS[1:10]},
        },
        "cpp": cpp_info,
        "julia": path_info(julia) | {
            "working_bits": actual_bits, "project": str(project),
            "project_files": [path_info(p) for p in [project / "Project.toml", *manifests] if p.is_file()],
            "checkout_commit": output(["git", "rev-parse", "HEAD"], cwd=REPO, timeout=10),
            "tracked_worktree_dirty": bool(output(["git", "status", "--porcelain", "--untracked-files=no"],
                                                    cwd=REPO, timeout=10)),
            "julia_depot_path": env.get("JULIA_DEPOT_PATH"),
        },
        "harness": [path_info(p) for p in (HERE / "run.py", HERE / "witness.py", HERE / "solve_julia.jl", measure_sources)],
        "measurement_monitor": path_info(measure) | {"build_argv": measure_command,
                                                       "compiler": output([cc, "--version"], timeout=10)},
        "host": {"platform": platform.platform(), "python": platform.python_version(),
                 "cpu": hardware[1] if hardware else None,
                 "cpu_max": optional_text(Path("/sys/fs/cgroup/cpu.max")),
                 "memory_max": optional_text(Path("/sys/fs/cgroup/memory.max")),
                 "affinity_cpus": cpus},
        "env_overrides": overrides, "threads": args.threads, "repeats": args.repeat,
        "seed": args.seed, "timeout_seconds": args.timeout,
    }
    write_json(out / "manifest.json", manifest)

    rng = random.Random(args.seed)
    records = {"cpp": [], "julia": []}
    with (out / "results.jsonl").open("w", encoding="utf-8") as stream:
        for repeat_id in range(1, args.repeat + 1):
            solvers = list(records)
            rng.shuffle(solvers)
            for solver in solvers:
                directory = out / f"{repeat_id:03d}-{solver}"
                directory.mkdir()
                witness = directory / "witness.out"
                if solver == "cpp":
                    command = [str(cpp), "-ds", str(common_input), "-p", str(common_params), "-o", str(witness)]
                else:
                    command = [str(julia), "--startup-file=no", "--history-file=no", f"--threads={args.threads},0",
                               "--gcthreads=1", f"--project={project}", str(HERE / "solve_julia.jl"),
                               str(REPO), str(common_input), str(common_params), str(witness),
                               str(actual_bits), str(digits), str(args.threads)]
                row = {"mode": mode, "solver": solver, "repeat": repeat_id, "argv": command}
                try:
                    row.update(run_child(measure, command, directory, env, cpus, args.timeout))
                    assess(row, witness, problem, args, actual_bits)
                except (OSError, ValueError) as error:
                    row.update(status="launch_or_measurement_error", detail=str(error))
                records[solver].append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                duration = row.get("wall_seconds") or row.get("external_elapsed_seconds")
                label = f"{duration:9.4f} s" if duration is not None else "(no timing)"
                print(f"{repeat_id:03d} {solver:5s}  {row['status']:22s} {label}", flush=True)

    success_kind = "verified" if mode == "matched_accuracy" else "diagnostic_complete"
    passed = all(r["status"] == success_kind for rows in records.values() for r in rows)
    summary = {"mode": mode, "all_runs_passed": passed,
               "status": {name: [r["status"] for r in rows] for name, rows in records.items()}}
    if passed:
        summary["median_wall_seconds"] = {name: statistics.median(r["wall_seconds"] for r in rows)
                                          for name, rows in records.items()}
        summary["max_peak_rss_bytes"] = {name: max(r["peak_rss_bytes"] for r in rows)
                                         for name, rows in records.items()}
        summary["paired_median_cpp_over_julia_wall"] = statistics.median(
            c["wall_seconds"] / j["wall_seconds"] for c, j in zip(records["cpp"], records["julia"])
        )
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if passed else 1


def main(argv=None):
    try:
        return campaign(options(argv))
    except (DecimalException, OSError, UnicodeError, ValueError,
            subprocess.SubprocessError, AttributeError) as error:
        print(f"parity: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
