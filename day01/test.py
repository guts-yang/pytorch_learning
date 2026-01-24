import argparse
import importlib
import importlib.util
from importlib.util import find_spec
import json
import os
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.skipped = False
        self.error = ""
        self.duration_ms = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "skipped": self.skipped,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 3),
        }


def _safe_import(candidate: Optional[str]) -> Optional[ModuleType]:
    if not candidate:
        for name in ("candidate.py", "intern.py", "solution.py"):
            path = os.path.join(os.getcwd(), name)
            if os.path.isfile(path):
                candidate = path
                break
    if not candidate:
        return None
    try:
        if os.path.isfile(candidate):
            abs_path = os.path.abspath(candidate)
            spec = importlib.util.spec_from_file_location("candidate_mod", abs_path)
            mod = importlib.util.module_from_spec(spec)  # type: ignore
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore
            return mod
        else:
            return importlib.import_module(candidate)
    except Exception:
        return None


def _resolve_func(mod: Optional[ModuleType], name: str, fallback: Callable[..., Any]) -> Callable[..., Any]:
    if mod and hasattr(mod, name):
        attr = getattr(mod, name)
        if callable(attr):
            return attr
    return fallback


def _ref_add(a: Any, b: Any) -> Any:
    return a + b


def _ref_fib(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    a, b = 0, 1 
    # fib(0) = 0, fib(1) = 1
    # fib(2) = 1, fib(3) = 2, fib(4) = 3, ...
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def _ref_reverse_string(s: str) -> str:
    return s[::-1]


def _ref_word_count(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return len(text.split())


def _ref_build_linear(in_features: int, out_features: int):
    torch_spec = find_spec("torch")
    if not torch_spec:
        raise ImportError("torch not available")
    import torch
    nn = torch.nn
    return nn.Linear(in_features, out_features)


def run_add_test(fn: Callable[..., Any]) -> TestResult:
    res = TestResult("add")
    start = time.perf_counter()
    try:
        cases = [
            ((1, 2), 3),
            ((-1, 5), 4),
            ((1.5, 2.5), 4.0),
            (("interview", "test"), "interviewtest"),
        ]
        for args, expected in cases:
            got = fn(*args)
            if isinstance(expected, float):
                if abs(got - expected) > 1e-9:
                    raise AssertionError(f"expected {expected}, got {got}")
            else:
                if got != expected:
                    raise AssertionError(f"expected {expected}, got {got}")
        res.passed = True
    except Exception as e:
        res.error = str(e)
    finally:
        res.duration_ms = (time.perf_counter() - start) * 1000
    return res


def run_fib_test(fn: Callable[..., Any]) -> TestResult:
    res = TestResult("fib")
    start = time.perf_counter()
    try:
        cases = [
            (0, 0),
            (1, 1),
            (2, 1),
            (5, 5),
            (10, 55),
        ]
        for n, expected in cases:
            got = fn(n)
            if got != expected:
                raise AssertionError(f"n={n} expected {expected}, got {got}")
        res.passed = True
    except Exception as e:
        res.error = str(e)
    finally:
        res.duration_ms = (time.perf_counter() - start) * 1000
    return res


def run_reverse_test(fn: Callable[..., Any]) -> TestResult:
    res = TestResult("reverse_string")
    start = time.perf_counter()
    try:
        cases = [
            ("hello", "olleh"),
            ("interview", "weivretni"),
            ("A", "A"),
            ("", ""),
        ]
        for s, expected in cases:
            got = fn(s)
            if got != expected:
                raise AssertionError(f"expected {expected}, got {got}")
        res.passed = True
    except Exception as e:
        res.error = str(e)
    finally:
        res.duration_ms = (time.perf_counter() - start) * 1000
    return res


def run_word_count_test(fn: Callable[..., Any]) -> TestResult:
    res = TestResult("word_count")
    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sample.txt")
            content = "hello world\nnihao shijie\nintern test"
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            expected = len(content.split())
            got = fn(path)
            if got != expected:
                raise AssertionError(f"expected {expected}, got {got}")
        res.passed = True
    except Exception as e:
        res.error = str(e)
    finally:
        res.duration_ms = (time.perf_counter() - start) * 1000
    return res


def run_torch_linear_test(fn: Callable[..., Any]) -> TestResult:
    res = TestResult("torch_linear")
    start = time.perf_counter()
    try:
        torch_spec = find_spec("torch")
        if not torch_spec:
            res.skipped = True
            return res
        import torch
        from torch import nn
        layer = fn(8, 4)
        if not isinstance(layer, nn.Linear):
            raise AssertionError("expected nn.Linear instance")
        x = torch.randn(5, 8, requires_grad=True)
        y = layer(x)
        if y.shape != (5, 4):
            raise AssertionError(f"unexpected output shape {y.shape}")
        s = y.sum()
        s.backward()
        if x.grad is None:
            raise AssertionError("gradient is None")
        res.passed = True
    except ImportError:
        res.skipped = True
    except Exception as e:
        res.error = str(e)
    finally:
        res.duration_ms = (time.perf_counter() - start) * 1000
    return res


def build_tests(mod: Optional[ModuleType]) -> List[Tuple[str, Callable[[], TestResult]]]:
    add_fn = _resolve_func(mod, "add", _ref_add)
    fib_fn = _resolve_func(mod, "fib", _ref_fib)
    rev_fn = _resolve_func(mod, "reverse_string", _ref_reverse_string)
    wc_fn = _resolve_func(mod, "word_count", _ref_word_count)
    lin_fn = _resolve_func(mod, "build_linear", _ref_build_linear)
    return [
        ("add", lambda: run_add_test(add_fn)),
        ("fib", lambda: run_fib_test(fib_fn)),
        ("reverse_string", lambda: run_reverse_test(rev_fn)),
        ("word_count", lambda: run_word_count_test(wc_fn)),
        ("torch_linear", lambda: run_torch_linear_test(lin_fn)),
    ]


def run_all(tests: List[Tuple[str, Callable[[], TestResult]]], only: Optional[List[str]], skip: Optional[List[str]]) -> List[TestResult]:
    name_to_test = {name: test for name, test in tests}
    names = list(name_to_test.keys())
    if only:
        names = [n for n in names if n in set(only)]
    if skip:
        names = [n for n in names if n not in set(skip)]
    results: List[TestResult] = []
    for n in names:
        results.append(name_to_test[n]())
    return results


def print_report(results: List[TestResult], as_json: bool, verbose: bool) -> None:
    if as_json:
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
        return
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    skipped = sum(1 for r in results if r.skipped)
    failed = total - passed - skipped
    print(f"total: {total}  passed: {passed}  skipped: {skipped}  failed: {failed}")
    for r in results:
        status = "PASS" if r.passed else ("SKIP" if r.skipped else "FAIL")
        base = f"[{status}] {r.name} ({round(r.duration_ms,3)} ms)"
        if verbose and r.error:
            base += f" | {r.error}"
        print(base)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Intern interview test runner")
    parser.add_argument("--candidate", type=str, default=None, help="Candidate script path or module name")
    parser.add_argument("--only", nargs="*", default=None, help="Only run these tests")
    parser.add_argument("--skip", nargs="*", default=None, help="Skip these tests")
    parser.add_argument("--json", action="store_true", help="Output results in JSON")
    parser.add_argument("--verbose", action="store_true", help="Show detailed error info")
    args = parser.parse_args(argv)
    mod = _safe_import(args.candidate)
    tests = build_tests(mod)
    results = run_all(tests, args.only, args.skip)
    print_report(results, args.json, args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
