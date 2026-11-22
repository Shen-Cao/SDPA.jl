import copy
from decimal import Decimal as D, localcontext
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run import REPO, assess, child_environment, cpu_set, julia_route, parameter_text, run_child
from witness import Inconclusive, Invalid, MAX_DECIMAL_DIGITS, cone_gate, read_problem, read_witness, validate


# Standard-form min <C,Y>, <A,Y>=6 becomes displayed SDPA min b*x,
# X=x*A-(-C). For x=-2, X=[[1,-1],[-1,1]] and Y=ones(2,2).
SDP = """1 = mDIM
1 = nBLOCK
{2} = blocks
6
0 1 1 1 -5
0 1 1 2 -1
0 1 2 2 -5
1 1 1 1 2
1 1 1 2 1
1 1 2 2 2
"""
DISPLAY = """xVec =
{-2}
xMat =
{ {{1,-1},{-1,1}} }
yMat =
{ {{1,1},{1,1}} }
"""
# LP-first then PSD then a size-one LP block. The two LP blocks must not
# be merged or permuted in the displayed witness.
MIXED = """1
3
{-2, 2, 1}
1
0 1 1 1 1
0 2 1 1 1
0 2 1 2 1
0 2 2 2 1
1 1 1 1 1
1 1 2 2 2
1 2 1 1 2
1 2 1 2 1
1 2 2 2 2
1 3 1 1 1
"""
MIXED_DISPLAY = """xVec =
{1}
xMat =
{ {0,2} {{1,0},{0,1}} {1} }
yMat =
{ {1,0} {{0,0},{0,0}} {0} }
"""
ZERO_COLD = """xVec =
{0}
xMat =
{ {{10000,0},{0,10000}} }
yMat =
{ {{10000,0},{0,10000}} }
"""


def julia_text(body=DISPLAY, phase="pdOPT", reason="OPTIMAL_TERMINAL", iterations=2):
    return f"""phase.value = {phase}
Iteration = {iterations}
exit reason = {reason}
precision = 256
threads = 1
julia = 1.13.0
active project = /some/Project.toml
mpfr = 4.2.2
blas = LBTConfig([ILP64] libopenblas64_.so)
read_seconds = 0.1
solve_seconds = 0.2
solve_core_seconds = 0.15
""" + body


class ParityOracle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def files(self, data=SDP, result=julia_text()):
        p = self.directory / "problem.dat-s"
        p.write_text(data, encoding="ascii")
        w = self.directory / "witness.out"
        w.write_text(result, encoding="ascii")
        return read_problem(p), w

    def test_exact_native_cpp_gold(self):
        p = read_problem(REPO / "test/fixtures/example1.dat-s")
        w = read_witness(REPO / "test/reference/example1.dat-s.result", p, "cpp")
        self.assertEqual((w.phase, w.iterations), ("pdOPT", 39))
        result = validate(p, w, D("1e-30"), 112)
        self.assertTrue(result["accuracy_passed"])
        self.assertLess(D(result["primal_objective"]), D("-41.899999999999"))

    def test_indefinite_dual_does_not_hide_behind_perfect_arithmetic(self):
        p, path = self.files()
        w = read_witness(path, p, "julia")
        self.assertTrue(validate(p, w, D("1e-30"), 80)["accuracy_passed"])
        # Feasibility, both objectives, gap, and raw complementarity remain
        # exact. Testing only objectives/residuals would report a false pass.
        bad = copy.deepcopy(w)
        bad.Y[0][0][0], bad.Y[0][1][1] = D(2), D(0)
        result = validate(p, bad, D("1e-30"), 80)
        for key in ("relative_gap", "primal_residual", "dual_residual", "complementarity"):
            self.assertEqual(D(result[key]), D(0))
        self.assertFalse(result["dual_cone"])
        self.assertFalse(result["accuracy_passed"])

    def test_shifted_cone_margin_and_skew(self):
        with localcontext() as ctx:
            ctx.prec = 100
            s, c, _ = cone_gate([[D("1"), D(2)], [D(2), D(4)]], D("1e-35"), 80, D(1))
            self.assertTrue(s and c)  # non-diagonally-dominant rank-one PSD
            s, c, _ = cone_gate([[D(1), D(3)], [D(2), D(9)]], D("1e-35"), 80, D(1))
            self.assertFalse(s)
            self.assertTrue(c)       # symmetry isn't silently repaired to a pass
            s, c, _ = cone_gate([[D(1), D("1.0000000001")],
                                [D("1.0000000001"), D(1)]], D("1e-12"), 80, D(1))
            self.assertTrue(s)
            self.assertFalse(c)

    def test_large_unrelated_witness_cannot_relax_cone_budget(self):
        # F1=diag(1,0), x=0 and X=0. Y can satisfy the affine equation
        # Y11=-1 only by leaving the cone. A huge Y22 cannot mask that.
        for sizes, zero, false_Y in [
            ("2", "{{0,0},{0,0}}", "{{-1,0},{0,1e40}}"),
            ("-2", "{0,0}", "{-1,1e40}"),
        ]:
            with self.subTest(sizes=sizes):
                data = f"1\n1\n{sizes}\n-1\n1 1 1 1 1\n"
                display = f"xVec =\n{{0}}\nxMat =\n{{{zero}}}\nyMat =\n{{{false_Y}}}\n"
                p, path = self.files(data=data, result=julia_text(display))
                result = validate(p, read_witness(path, p, "julia"), D("1e-30"), 100)
                for key in ("relative_gap", "primal_residual", "dual_residual", "complementarity"):
                    self.assertEqual(D(result[key]), D(0))
                self.assertFalse(result["dual_cone"])
                self.assertFalse(result["accuracy_passed"])

    def test_large_unrelated_witness_cannot_relax_symmetry_budget(self):
        data = "1\n1\n3\n1\n1 1 1 1 1\n"
        display = """xVec =
{0}
xMat =
{ {{0,0,0},{0,0,0},{0,0,0}} }
yMat =
{ {{1,1,0},{-1,0,0},{0,0,1e40}} }
"""
        p, path = self.files(data=data, result=julia_text(display))
        result = validate(p, read_witness(path, p, "julia"), D("1e-30"), 100)
        self.assertTrue(result["dual_cone"])
        self.assertFalse(result["dual_symmetric"])
        self.assertFalse(result["accuracy_passed"])

    def test_large_psd_witness_does_not_lose_small_input_scale_shift(self):
        # Y is genuinely rank one. The input-scale shift is much smaller
        # than the values in Y, so the candidate needs more than the
        # baseline guard for 256-bit solver output.
        data = "1\n1\n2\n0\n1 1 1 1 9\n1 1 1 2 -3\n1 1 2 2 1\n"
        display = """xVec =
{0}
xMat =
{ {{0,0},{0,0}} }
yMat =
{ {{2e100,6e100},{6e100,18e100}} }
"""
        p, path = self.files(data=data, result=julia_text(display))
        w = read_witness(path, p, "julia")
        self.assertTrue(validate(p, w, D("1e-30"), 126)["accuracy_passed"])
        with self.assertRaises(Inconclusive):
            validate(p, w, D("1e-30"), MAX_DECIMAL_DIGITS + 1)

    def test_input_digits_beyond_binary64_change_the_verdict(self):
        p, path = self.files(data=MIXED.replace("\n1\n0", "\n1.000000000000000000000000000000000000000000005\n0"),
                             result=julia_text(MIXED_DISPLAY))
        self.assertEqual(float(p.b[0]), 1.0)
        row = validate(p, read_witness(path, p, "julia"), D("1e-50"), 94)
        self.assertFalse(row["accuracy_passed"])
        self.assertGreater(D(row["dual_residual"]), D("1e-46"))

    def test_block_layout_and_missing_numbers(self):
        p, path = self.files(data=MIXED, result=julia_text(MIXED_DISPLAY))
        w = read_witness(path, p, "julia")
        self.assertTrue(validate(p, w, D("1e-30"), 80)["accuracy_passed"])
        for changed in (
            MIXED_DISPLAY.replace("{0,2}", "{0}"),
            MIXED_DISPLAY.replace("{0,2}", "{0,NaN}"),
            MIXED_DISPLAY.replace("{0,2}", "{0,2} {0}"),
            MIXED_DISPLAY.replace("{0,2}", "{0,2,}"),
            MIXED_DISPLAY.replace("{0,2}", "{0,2}") + "unparsed extra section",
        ):
            with self.subTest(changed=changed):
                _, result = self.files(data=MIXED, result=julia_text(changed))
                with self.assertRaises(Invalid):
                    read_witness(result, p, "julia")

    def test_shared_input_rejects_incompatible_or_ambiguous_data(self):
        for data in (SDP + "1 1 2 1 1\n", SDP + "1 1 1 1 2\n", SDP + "1 1 1\n",
                     "\n" + SDP, SDP.replace("\n6\n", "\n.6\n"),
                     MIXED + "1 1 1 2 3\n", SDP.replace("1 1 2 2 2", "1 1 2 2 NaN"),
                     SDP.replace("nBLOCK", "nBLOCKv2"), SDP.replace("= blocks", "= x-1"),
                     '"' + 'c' * 254 + "\n" + SDP, " " * 255 + SDP, SDP.replace("\n6\n", "\r6\n"),
                     SDP.replace("\n6\n", "\n6\0\n")):
            with self.subTest(data=data):
                with self.assertRaises(Invalid):
                    self.files(data=data)
        data = SDP.replace("\n", "\r\n")
        p, _ = self.files(data=data)
        self.assertEqual((p.blocks, p.b), ((2,), (D(6),)))

    def test_shell_success_not_enough_and_fixed_iterations_not_accuracy(self):
        args = SimpleNamespace(threads=1, bits=256, target=D("1e-30"), guard_digits=80,
                               iteration_cap=3, fixed_iterations=None)
        (self.directory / "stderr.log").write_text("Using dense Schur complement path\n")
        cases = [
            ("verified", julia_text(iterations=0), None),
            ("solver_status", julia_text(reason="CHOLESKY_FAILURE"), None),
            ("solver_status", julia_text(phase="pdINF"), None),
            ("solver_status", julia_text(phase="pFEAS_dINF"), None),
            ("invalid_result", julia_text().replace("{-2}", "{NaN}"), None),
            ("accuracy_failed", julia_text(DISPLAY.replace("{-2}", "{-1}")), None),
            ("invalid_result", julia_text().replace("threads = 1", "threads = 9"), None),
            ("incomplete_iterations", julia_text(ZERO_COLD, "noINFO", "MAX_ITERATION", 1), 2),
            ("diagnostic_complete", julia_text(ZERO_COLD, "noINFO", "MAX_ITERATION", 2), 2),
        ]
        for expected, result, cap in cases:
            with self.subTest(expected=expected):
                p, path = self.files(result=result)
                row = {"solver": "julia", "exit_code": 0, "timed_out": False}
                args.fixed_iterations = cap
                assess(row, path, p, args, 256)
                self.assertEqual(row["status"], expected)
                if cap and expected == "diagnostic_complete":
                    self.assertFalse(row["validation"]["accuracy_passed"])
        row = {"solver": "cpp", "exit_code": 0, "timed_out": False}
        assess(row, self.directory / "missing.out", p, args, 256)
        self.assertEqual(row["status"], "invalid_result")

    def test_parameter_lines_are_explicit_and_include_four_formats(self):
        controls, param = parameter_text(REPO / "test/fixtures/param.sdpa", 200, 3, 95)
        self.assertEqual((controls["maxIteration"], controls["precision"]), ("3", "200"))
        self.assertEqual(controls["epsilonStar"], "1.0E-30")
        lines = param.splitlines()
        self.assertEqual(len(lines), 15)
        self.assertEqual(lines[-4], "%+.95Fe  xPrint")
        self.assertEqual(lines[-1], "%+.95Fe  infPrint")

    def test_cpp_explicit_cpu_places_do_not_bind_julias_initial_thread(self):
        with patch.dict(os.environ, {"OMP_PLACES": "sockets", "GOMP_CPU_AFFINITY": "1"}):
            cpp, _ = child_environment([2, 6], "cpp")
            julia, _ = child_environment([2, 6], "julia")
            self.assertEqual((cpp["OMP_NUM_THREADS"], cpp["OMP_PLACES"]), ("2", "{2},{6}"))
            self.assertEqual(julia["OMP_PROC_BIND"], "FALSE")
            self.assertNotIn("OMP_PLACES", julia)
            self.assertNotIn("GOMP_CPU_AFFINITY", cpp)


class JuliaRoute(unittest.TestCase):
    """Route attribution must survive the messages solve! actually emits."""

    def route_for(self, text):
        with tempfile.TemporaryDirectory() as t:
            log = Path(t) / "stderr.log"
            log.write_text(text, encoding="utf-8")
            return julia_route(log)

    def test_forced_sparse_wins_over_the_density_veto(self):
        # SDPA_FORCE_SPARSE_PROFILE=1 logs the density-veto line first and
        # the FORCED override second; both appear in one stderr stream.
        # The override is definitive, so the run must be reported sparse.
        self.assertEqual(self.route_for(
            "[ Info: Block-level skyline: 5050 entries (density=100.0%)\n"
            "[ Info: Block-level density >25% — using dense path "
            "(F1/F2/F3 couple all constraints within blocks)\n"
            "[ Info: FORCED sparse profile (SDPA_FORCE_SPARSE_PROFILE=1): "
            "5050 entries (density=100%)\n"), "sparse")

    def test_dense_path_and_late_density_veto_report_dense(self):
        self.assertEqual(
            self.route_for("[ Info: Using dense Schur complement path\n"),
            "dense")
        # The opening sparse message is tentative; a later veto is final.
        self.assertEqual(self.route_for(
            "[ Info: Using sparse Schur complement path (m=100)\n"
            "[ Info: Block-level density >25% — using dense path "
            "(F1/F2/F3 couple all constraints within blocks)\n"), "dense")

    def test_plain_sparse_profile_and_no_markers(self):
        self.assertEqual(self.route_for(
            "[ Info: Block-level profile: 100 entries (density=12.34%)\n"),
            "sparse")
        self.assertEqual(self.route_for(""), "unobserved")
        self.assertEqual(self.route_for("[ Info: unrelated output\n"),
                         "unobserved")


@unittest.skipUnless(sys.platform == "linux" and hasattr(os, "pidfd_open") and shutil.which("cc"),
                     "needs Linux pidfd and a C compiler")
class MeasurementBoundary(unittest.TestCase):
    def test_child_status_timeout_and_rss_not_inherited_from_python(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            helper = tmp / "measure"
            subprocess.run(["cc", "-o", str(helper), str(Path(__file__).with_name("measure_child.c"))],
                           check=True)
            cpus = cpu_set(None, 1)
            # Keep pages alive and resident; a plain Python wait4() on its
            # child sees this parent heap before exec.
            allocated = bytearray(48 * 1024**2)
            for i in range(0, len(allocated), 4096):
                allocated[i] = 42
            ok_dir = tmp / "ok"
            ok_dir.mkdir()
            row = run_child(helper, ["/bin/sh", "-c", "exit 7"], ok_dir, dict(os.environ), cpus, 10)
            self.assertEqual(row["exit_code"], 7)
            self.assertLess(row["peak_rss_bytes"], len(allocated) // 2)
            self.assertGreater(row["wall_seconds"], 0)
            kill_dir = tmp / "killed"
            kill_dir.mkdir()
            row = run_child(helper, ["/bin/sh", "-c", "sleep 30; echo unexpected"], kill_dir,
                            dict(os.environ), cpus, 0.05)
            self.assertTrue(row["timed_out"])
            self.assertLess(row["external_elapsed_seconds"], 5)
            self.assertIsNone(row["peak_rss_bytes"])
            self.assertEqual((kill_dir / "stdout.log").read_bytes(), b"")

    def test_deadline_uses_actual_completion_not_delayed_observation(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            helper = tmp / "measure"
            subprocess.run(["cc", "-o", str(helper), str(Path(__file__).with_name("measure_child.c"))],
                           check=True)
            pidfd_open = os.pidfd_open

            def delayed_open(pid):
                fd = pidfd_open(pid)
                time.sleep(0.25)
                return fd

            cpus = cpu_set(None, 1)
            env = dict(os.environ)
            with patch("run.os.pidfd_open", delayed_open):
                d = tmp / "on_time"
                d.mkdir()
                on_time = run_child(helper, ["/bin/true"], d, env, cpus, 0.18)
                self.assertGreater(on_time["external_elapsed_seconds"], 0.18)
                self.assertFalse(on_time["timed_out"])
                d = tmp / "late"
                d.mkdir()
                late = run_child(helper, ["/bin/sleep", "0.14"], d, env, cpus, 0.05)
                self.assertTrue(late["timed_out"])
                self.assertEqual(late["timeout_detail"], "late completion")
                self.assertIsNone(late["wall_seconds"])


if __name__ == "__main__":
    unittest.main()
