"""Independent SDPA sparse-data / displayed-witness reader and decimal oracle.

This module deliberately does not import SDPA or either solver's arithmetic.
See README.md for the convention, normalization, and scope.
"""
from dataclasses import dataclass
from decimal import Decimal as D, localcontext
from itertools import chain
from pathlib import Path
import re


class Invalid(ValueError):
    """An unsupported input or incomplete/malformed solver result."""


NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
INTEGER = r"[+-]?[0-9]+"
ZERO = D(0)
ONE = D(1)
TWO = D(2)


def decimal(token):
    if not re.fullmatch(NUMBER, token):
        raise Invalid(f"not a finite decimal: {token[:80]!r}")
    value = D(token)
    return ZERO if value.is_zero() else value


def integers(tokens):
    if not all(re.fullmatch(INTEGER, t) for t in tokens):
        raise Invalid("expected integer dimensions/indices")
    return tuple(map(int, tokens))


@dataclass
class Problem:
    blocks: tuple
    b: tuple
    entries: tuple   # one dictionary per F_k; (block, i, j), zero-based, upper triangular


def read_problem(path):
    lines = iter(enumerate(Path(path).read_text(encoding="ascii").splitlines(), 1))
    first = next(lines, None)
    # The C++ reader recognizes comments only at column 1, before the header.
    while first and first[1].startswith(('"', '*')):
        first = next(lines, None)
    header = [first, next(lines, None), next(lines, None)]
    if any(line is None or not line[1].strip() for line in header):
        raise Invalid("expected three separate nonempty dimension lines")
    fields = [line.split("=", 1)[0].split() for _, line in header[:2]]
    fields.append(re.sub(r"[{},]", " ", header[2][1].split("=", 1)[0]).split())
    if len(fields[0]) != 1 or len(fields[1]) != 1:
        raise Invalid("expected separate mDIM and nBLOCK lines")
    m, n = integers(fields[0] + fields[1])
    blocks = integers(fields[2])
    if m <= 0 or n <= 0 or len(blocks) != n or 0 in blocks:
        raise Invalid("expected positive dimensions and nonzero signed blocks")
    # The pinned C++ fscanf scans for a digit or sign before each numeric
    # field (bare '.5' could be interpreted as '5', unlike Julia).
    def input_value(token):
        if token.startswith("."):
            raise Invalid("use a leading zero in .dat-s decimals (write 0.5)")
        return decimal(token)

    b = []
    entries = None
    for row, line in lines:
        tokens = re.sub(r"[{},]", " ", line).split()
        missing = m - len(b)
        if missing:
            b.extend(input_value(v) for v in tokens[:missing])
            tokens = tokens[missing:]
        if len(tokens) % 5:
            raise Invalid(f"line {row}: partial sparse 5-tuple")
        if not tokens:
            continue
        if entries is None:
            entries = tuple({} for _ in range(m + 1))
        for start in range(0, len(tokens), 5):
            k, block, i, j = integers(tokens[start:start + 4])
            if not (0 <= k <= m and 1 <= block <= n and 1 <= i <= j <= abs(blocks[block - 1])):
                raise Invalid(f"line {row}: sparse index out of range (expected upper triangle)")
            if blocks[block - 1] < 2 and i != j:
                raise Invalid(f"line {row}: off-diagonal entry in an LP block")
            key = (block - 1, i - 1, j - 1)
            if key in entries[k]:
                raise Invalid(f"line {row}: duplicate entry (solver handling can differ)")
            entries[k][key] = input_value(tokens[start + 4])
    if len(b) != m:
        raise Invalid("truncated objective vector")
    if entries is None:
        raise Invalid("expected at least one sparse entry")
    return Problem(blocks, tuple(b), entries)


class Braces:
    """Read exactly the native nested shape; never discard unmatched text."""

    token_re = re.compile(r"\s*([{},]|" + NUMBER + r")")

    def __init__(self, text):
        self.text = text
        self.at = 0

    def take(self):
        match = self.token_re.match(self.text, self.at)
        if match is None:
            raise Invalid(f"missing or invalid witness token near {self.text[self.at:self.at + 40]!r}")
        self.at = match.end()
        return match[1]

    def expect(self, token):
        if self.take() != token:
            raise Invalid(f"expected {token!r} in displayed witness")

    def vector(self, n):
        self.expect("{")
        result = []
        for k in range(n):
            if k:
                self.expect(",")
            result.append(decimal(self.take()))
        self.expect("}")
        return result

    def blocks(self, sizes):
        self.expect("{")
        result = []
        for bs in sizes:
            if bs < 2:
                result.append(self.vector(abs(bs)))
                continue
            self.expect("{")
            rows = []
            for i in range(bs):
                if i:
                    self.expect(",")
                rows.append(self.vector(bs))
            self.expect("}")
            result.append(rows)
        self.expect("}")
        return result

    def end(self):
        if self.text[self.at:].strip():
            raise Invalid("extra data after displayed witness")


def marker(text, pattern, label):
    matches = list(re.finditer(pattern, text, re.M))
    if len(matches) != 1:
        raise Invalid(f"expected one {label} in result; got {len(matches)}")
    return matches[0]


@dataclass
class Witness:
    phase: str
    iterations: int
    meta: dict
    x: list
    X: list
    Y: list


def read_witness(path, problem, solver):
    text = Path(path).read_text(encoding="utf-8")
    xv, xm, ym = (marker(text, rf"^[ \t]*{key}[ \t]*=[ \t]*$", key)
                  for key in ("xVec", "xMat", "yMat"))
    if not xv.start() < xm.start() < ym.start():
        raise Invalid("out-of-order displayed witness")
    head = text[:xv.start()]
    phase = marker(head, r"^phase\.value[ \t]*=[ \t]*([A-Za-z]+)[ \t]*$", "phase")[1]
    steps = int(marker(head, r"^[ \t]*Iteration[ \t]*=[ \t]*([0-9]+)[ \t]*$", "iteration")[1])
    metadata = {}
    fields = {"precision": r"[0-9]+"}
    if solver == "cpp":
        trailer = marker(text, r"^[ \t]*main loop time[ \t]*=[ \t]*(" + NUMBER + r")[ \t]*$",
                         "C++ main-loop footer")
        if trailer.start() <= ym.end():
            raise Invalid("out-of-order C++ footer")
        witness_end = trailer.start()
        metadata["reported_route"] = marker(head, r"^(DENSE|SPARSE) computation[ \t]*$", "C++ route")[1].lower()
        metadata["native_cpp_main_loop_seconds"] = str(decimal(trailer[1]))
        for key, outkey in [("total time", "native_cpp_read_and_solve_seconds"),
                            ("file   read time", "native_cpp_file_read_seconds")]:
            match = marker(text[witness_end:], rf"^[ \t]*{key}[ \t]*=[ \t]*({NUMBER})[ \t]*$", key)
            metadata[outkey] = str(decimal(match[1]))
    elif solver == "julia":
        witness_end = len(text)
        fields |= {"exit reason": r"[A-Z_]+", "threads": r"[0-9]+", "julia": r"[^\r\n]+",
                   "mpfr": r"[^\r\n]+", "blas": r"[^\r\n]+", "read_seconds": NUMBER,
                   "solve_seconds": NUMBER, "solve_core_seconds": NUMBER}
    else:
        raise Invalid(f"unknown solver {solver!r}")
    for key, pattern in fields.items():
        metadata[key] = marker(head, rf"^{key}[ \t]*=[ \t]*({pattern})[ \t]*$", key)[1].strip()

    parsers = [Braces(text[xv.end():xm.start()]), Braces(text[xm.end():ym.start()]),
               Braces(text[ym.end():witness_end])]
    x = parsers[0].vector(len(problem.b))
    X = parsers[1].blocks(problem.blocks)
    Y = parsers[2].blocks(problem.blocks)
    for parser in parsers:
        parser.end()
    return Witness(phase, steps, metadata, x, X, Y)


def values(blocks, sizes):
    for bs, block in zip(sizes, blocks):
        if bs < 2:
            yield from block
        else:
            yield from chain.from_iterable(block)


def exact_precision(numbers, degree, terms):
    """Digits sufficient for exact sums of at most 'terms' decimal products.

    Every value is an integer times a power of ten. Bound the smallest
    product exponent and the largest magnitude of their sum, not just the
    input mantissas (cancellation between disparate exponents matters).
    Additional decimal places cover division/multiplication by 2.
    """
    least = 0
    most = 1
    for number in numbers:
        if number:
            least = min(least, number.as_tuple().exponent)
            most = max(most, number.adjusted() + 1)
    needed = max(80, degree * (most - least) + len(str(terms)) + 16)
    if needed > 10_000:
        raise Invalid("decimal exponent span exceeds the reference verifier's guard limit")
    return needed


def reported_ratio(numerator, denominator):
    with localcontext() as report:
        report.prec = 42
        return numerator / denominator


def cone_gate(matrix, tolerance, guard):
    """Sufficient check for lambda_min(sym(matrix)) >= -tol*(1+||matrix||_inf)."""
    n = len(matrix)
    scale = ONE + max(sum(map(abs, row), ZERO) for row in matrix)
    skew = max(abs(matrix[i][j] - matrix[j][i]) for i in range(n) for j in range(i + 1))
    sym = [[(matrix[i][j] + matrix[j][i]) / TWO for j in range(i + 1)] for i in range(n)]
    shift = tolerance * scale / TWO
    for i in range(n):
        sym[i][i] += shift
    symmetric = skew <= tolerance * scale
    # Cheap exact test on small diagonally dominant / zero matrices.
    off = [ZERO] * n
    for i in range(n):
        for j in range(i):
            off[i] += abs(sym[i][j])
            off[j] += abs(sym[i][j])
    if all(sym[i][i] >= off[i] for i in range(n)):
        return symmetric, True, reported_ratio(skew, scale)

    # Compute an independent Decimal Cholesky candidate. A successful
    # factor alone is not relied on: verify a reconstruction bound below.
    lower = []
    with localcontext() as work:
        work.prec = guard
        for i in range(n):
            row = []
            for j in range(i + 1):
                other = row if i == j else lower[j]
                delta = sym[i][j] - sum((row[k] * other[k] for k in range(j)), ZERO)
                if i == j:
                    if delta <= ZERO:
                        return symmetric, False, reported_ratio(skew, scale)
                    row.append(delta.sqrt())
                else:
                    row.append(delta / other[j])
            lower.append(row)
    # LL' is PSD in exact arithmetic. Bound ||LL'-(S+shift*I)||_inf
    # in a context large enough that the product/sum is exact on the
    # represented decimals. Error <= shift implies S >= -2*shift*I.
    with localcontext() as proof:
        proof.prec = exact_precision(chain.from_iterable(chain(lower, sym)), 2, n * n)
        errors = [ZERO] * n
        for i in range(n):
            for j in range(i + 1):
                err = abs(sum((lower[i][k] * lower[j][k] for k in range(j + 1)), ZERO) - sym[i][j])
                errors[i] += err
                if i != j:
                    errors[j] += err
        psd = max(errors) <= shift
    return symmetric, psd, reported_ratio(skew, scale)


def validate(problem, witness, tolerance, guard):
    """Return normalized witness metrics and a conjunctive acceptance test.

    The affine residuals, normalization products, and gap comparisons are
    exact on the finite decimals. Ratios are rounded only for reporting.
    """
    b, blocks, entries = problem.b, problem.blocks, problem.entries
    x, X, Y = witness.x, witness.X, witness.Y
    number_stream = chain(b, [tolerance], x, values(X, blocks), values(Y, blocks),
                          *(entry.values() for entry in entries))
    with localcontext() as ctx:
        ctx.prec = exact_precision(number_stream, 3, 2 + sum(len(e) for e in entries)
                                   + sum(bs * bs if bs >= 2 else abs(bs) for bs in blocks))
        slack = {}
        dual_res = ZERO
        dual_obj = ZERO
        for k, Fk in enumerate(entries):
            inner = ZERO
            for (block, i, j), coeff in Fk.items():
                key = (block, i, j)
                slack[key] = slack.get(key, ZERO) + (coeff * x[k - 1] if k else -coeff)
                B = Y[block]
                if blocks[block] < 2:
                    inner += coeff * B[i]
                else:
                    inner += coeff * (B[i][j] + B[j][i] if i != j else B[i][i])
            if not k:
                dual_obj = inner
            else:
                dual_res = max(dual_res, abs(inner - b[k - 1]))

        primal_res = ZERO
        trace = ZERO
        for block, bs in enumerate(blocks):
            P, Q = X[block], Y[block]
            if bs < 2:
                for i, (p, q) in enumerate(zip(P, Q)):
                    primal_res = max(primal_res, abs(slack.get((block, i, i), ZERO) - p))
                    trace += p * q
            else:
                for i in range(bs):
                    for j in range(bs):
                        s = slack.get((block, min(i, j), max(i, j)), ZERO)
                        primal_res = max(primal_res, abs(s - P[i][j]))
                        trace += P[i][j] * Q[i][j]
        primal_obj = sum((value * item for value, item in zip(b, x)), ZERO)
        objective_scale = ONE + abs(primal_obj) + abs(dual_obj)
        magnitudes = {
            "primal_residual": (primal_res, ONE + max(map(abs, entries[0].values()), default=ZERO)),
            "dual_residual": (dual_res, ONE + max(map(abs, b), default=ZERO)),
            "relative_gap": (abs(primal_obj - dual_obj), objective_scale),
            "complementarity": (abs(trace), objective_scale),
        }
        algebra_ok = all(value <= tolerance * denom for value, denom in magnitudes.values())
        result = {key: str(reported_ratio(value, denom)) for key, (value, denom) in magnitudes.items()}
        result["primal_objective"] = str(primal_obj)
        result["dual_objective"] = str(dual_obj)

        for axis, matrices in (("primal", X), ("dual", Y)):
            symmetry, cone, skew = True, True, ZERO
            for bs, mat in zip(blocks, matrices):
                if bs < 2:
                    cone &= min(mat) >= -tolerance * (ONE + max(map(abs, mat)))
                else:
                    symmetric, psd, defect = cone_gate(mat, tolerance, guard)
                    symmetry &= symmetric
                    cone &= psd
                    skew = max(skew, defect)
            result[axis + "_symmetric"] = symmetry
            result[axis + "_symmetry_residual"] = str(skew)
            result[axis + "_cone"] = cone
        result["accuracy_passed"] = algebra_ok and all(
            result[k] for k in ("primal_symmetric", "dual_symmetric", "primal_cone", "dual_cone")
        )
        return result
