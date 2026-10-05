from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import io
import json
import os
import signal
import traceback
import warnings
from pathlib import Path
from typing import Any


# Graph workers also run as standalone scripts in scientific environments where
# the framework package is not installed.
if __package__:
    from .operator_call_fixtures import API_CALL_FIXTURES
else:
    from operator_call_fixtures import API_CALL_FIXTURES


RISKY_NAME_PARTS = {
    "browser",
    "download",
    "fetch",
    "from_file",
    "from_url",
    "load",
    "open",
    "plot",
    "read",
    "save",
    "show",
    "to_file",
    "upload",
    "viewer",
    "write",
}

RISKY_PARAM_PARTS = {
    "api_key",
    "client",
    "database",
    "filename",
    "filepath",
    "host",
    "path",
    "port",
    "session",
    "token",
    "url",
}


class TimeoutErrorForSymbol(RuntimeError):
    pass


def ensure_runtime_dirs() -> None:
    os.environ.setdefault("HOME", "/tmp/scicode_operator_home")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/scicode_operator_mpl")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/scicode_operator_numba")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/scicode_operator_xdg")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("BOKEH_BROWSER", "none")
    for key in ("HOME", "MPLCONFIGDIR", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def short_text(value: Any, limit: int = 240) -> str:
    text = str(value)
    text = text.replace("\n", " ")
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_symbol(symbol: str) -> dict:
    parts = symbol.split(".")
    errors = []
    for split in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split])
        attrs = parts[split:]
        try:
            obj = importlib.import_module(module_name)
        except BaseException as exc:
            errors.append(f"import {module_name}: {type(exc).__name__}: {short_text(exc)}")
            continue
        try:
            for attr in attrs:
                obj = getattr(obj, attr)
            return {
                "ok": True,
                "module": module_name,
                "object": obj,
                "error": None,
            }
        except BaseException as exc:
            errors.append(
                f"getattr {module_name}.{'.'.join(attrs)}: {type(exc).__name__}: {short_text(exc)}"
            )
    return {
        "ok": False,
        "module": None,
        "object": None,
        "error": " | ".join(errors[-4:]),
    }


def import_numpy():
    import numpy as np

    return np


def import_scipy_sparse():
    import scipy.sparse as sp

    return sp


def rdkit_mol():
    try:
        from rdkit import Chem

        return Chem.MolFromSmiles("CCO")
    except Exception:
        return "CCO"


def qutip_state():
    try:
        import qutip

        return qutip.basis(2, 0)
    except Exception:
        return None


def qiskit_circuit():
    try:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)
        return qc
    except Exception:
        return None


def fixture_for_name(name: str) -> tuple[bool, Any]:
    low = name.lower().strip("_")
    np = import_numpy()

    if low in {
        "n",
        "m",
        "k",
        "dim",
        "dims",
        "dimension",
        "size",
        "rank",
        "degree",
        "order",
        "num",
        "num_qubits",
        "n_qubits",
        "levels",
        "bins",
    }:
        return True, 3
    if (
        low.startswith("n_")
        or low.startswith("num_")
        or low in {"nbits", "bits", "radius", "maxiter", "iterations"}
    ):
        return True, 3
    if low in {"axis", "i", "j", "index", "idx", "q", "qubit", "target", "control"}:
        return True, 0
    if low in {"seed", "random_state"}:
        return True, 0
    if low in {"shape"}:
        return True, (3, 3)
    if low in {"dtype"}:
        return True, float
    if low in {
        "copy",
        "normalize",
        "normalized",
        "full_matrices",
        "return_counts",
        "return_index",
        "return_inverse",
        "check_finite",
    }:
        return True, False
    if low in {
        "alpha",
        "beta",
        "gamma",
        "sigma",
        "scale",
        "loc",
        "mu",
        "mean",
        "std",
        "var",
        "df",
        "p",
        "prob",
        "probability",
        "tol",
        "rtol",
        "atol",
        "eps",
        "theta",
        "phi",
        "angle",
        "mass",
        "charge",
        "energy",
        "temperature",
    }:
        return True, 1.0
    if low in {
        "x",
        "y",
        "z",
        "a",
        "b",
        "c",
        "r",
        "u",
        "v",
        "w",
        "arr",
        "array",
        "data",
        "values",
        "sample",
        "samples",
        "vector",
        "vec",
        "input",
        "signal",
        "observations",
        "weights",
    }:
        return True, np.array([1.0, 2.0, 3.0])
    if low in {
        "matrix",
        "mat",
        "m",
        "a_matrix",
        "operator",
        "hamiltonian",
        "h",
        "jac",
        "jacobian",
        "hessian",
    }:
        return True, np.eye(3)
    if low in {"points", "coords", "coordinates", "positions", "vertices"}:
        return True, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    if low in {"x0", "p0", "initial", "initial_guess", "guess", "y0"}:
        return True, np.array([1.0, 0.5])
    if low in {"t", "time", "times", "grid", "t_eval"}:
        return True, np.linspace(0.0, 1.0, 5)
    if low in {"t_span", "span", "bounds"}:
        return True, (0.0, 1.0)
    if low in {"func", "function", "fun", "f", "callable", "objective"}:
        return True, lambda x, *args: x * x if not isinstance(x, (list, tuple)) else x
    if low in {"ode", "rhs"}:
        return True, lambda t, y: -y
    if low in {"s", "text", "string", "name", "label"}:
        return True, "x"
    if low in {"sequence", "seq", "dna", "protein"}:
        return True, "ACGTACGT"
    if low in {"smiles", "smarts"}:
        return True, "CCO"
    if low in {"formula"}:
        return True, "H2O"
    if low in {"mol", "molecule"}:
        return True, rdkit_mol()
    if low in {"state", "psi", "ket"}:
        state = qutip_state()
        if state is not None:
            return True, state
    if low in {"circuit", "qc"}:
        circuit = qiskit_circuit()
        if circuit is not None:
            return True, circuit
    if low in {"format"}:
        return True, "csr"
    return False, None


def fixture_for_annotation(annotation: Any) -> tuple[bool, Any]:
    if annotation is inspect._empty:
        return False, None
    text = getattr(annotation, "__name__", str(annotation)).lower()
    np = import_numpy()
    if "int" in text:
        return True, 3
    if "float" in text or "double" in text:
        return True, 1.0
    if "bool" in text:
        return True, False
    if "str" in text:
        return True, "x"
    if "array" in text or "ndarray" in text:
        return True, np.array([1.0, 2.0, 3.0])
    return False, None


def safe_to_attempt_call(symbol: str) -> tuple[bool, str | None]:
    low = symbol.lower()
    parts = low.split(".")
    joined_tail = ".".join(parts[-2:])
    for token in RISKY_NAME_PARTS:
        if token in joined_tail:
            return False, f"risky_name:{token}"
    return True, None


def build_call(obj: Any, symbol: str) -> tuple[str, list[Any], dict[str, Any], str | None]:
    ok, reason = safe_to_attempt_call(symbol)
    if not ok:
        return "skipped", [], {}, reason

    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "attempt_no_signature", [], {}, None

    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    required_count = 0
    for param in sig.parameters.values():
        if param.name in {"self", "cls"}:
            continue
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        is_required = param.default is inspect._empty
        if not is_required:
            continue
        required_count += 1
        low_name = param.name.lower()
        if any(part in low_name for part in RISKY_PARAM_PARTS):
            return "skipped", [], {}, f"risky_param:{param.name}"
        ok_value, value = fixture_for_name(param.name)
        if not ok_value:
            ok_value, value = fixture_for_annotation(param.annotation)
        if not ok_value:
            return "skipped", [], {}, f"missing_fixture:{param.name}"
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[param.name] = value

    if required_count > 5:
        return "skipped", [], {}, f"too_many_required:{required_count}"
    return "attempt", args, kwargs, None


def timeout_handler(signum, frame) -> None:  # noqa: ARG001
    raise TimeoutErrorForSymbol("call timed out")


def smoke_call(obj: Any, symbol: str, timeout_seconds: int) -> dict:
    if not callable(obj):
        return {
            "call_status": "not_callable",
            "call_ok": False,
            "call_error": None,
            "return_type": None,
            "return_repr": None,
            "call_fixture": None,
        }
    fixture = API_CALL_FIXTURES.get(symbol)
    fixture_kind = "explicit_api" if fixture is not None else "parameter_name"
    if fixture is None:
        mode, args, kwargs, skip_reason = build_call(obj, symbol)
    else:
        allowed, skip_reason = safe_to_attempt_call(symbol)
        mode, args, kwargs = "attempt" if allowed else "skipped", [], {}
    if mode == "skipped":
        return {
            "call_status": "call_skipped",
            "call_ok": False,
            "call_error": skip_reason,
            "return_type": None,
            "return_repr": None,
            "call_fixture": fixture_kind,
        }

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with (
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
                contextlib.ExitStack() as resources,
            ):
                if fixture is not None:
                    args, kwargs = fixture(resources)
                value = obj(*args, **kwargs)
        return {
            "call_status": "call_ok",
            "call_ok": True,
            "call_error": None,
            "return_type": type(value).__name__,
            "return_repr": short_text(repr(value)),
            "call_fixture": fixture_kind,
        }
    except BaseException as exc:
        return {
            "call_status": "call_error",
            "call_ok": False,
            "call_error": f"{type(exc).__name__}: {short_text(exc)}",
            "return_type": None,
            "return_repr": None,
            "call_fixture": fixture_kind,
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def validate_symbol(symbol: str, timeout_seconds: int) -> dict:
    row = {
        "api_symbol": symbol,
        "resolve_ok": False,
        "module": None,
        "object_type": None,
        "callable": False,
        "signature": None,
        "call_status": "not_attempted",
        "call_ok": False,
        "call_error": None,
        "return_type": None,
        "return_repr": None,
        "call_fixture": None,
        "validation_status": "resolve_error",
        "error": None,
    }
    try:
        resolved = resolve_symbol(symbol)
        if not resolved["ok"]:
            row["error"] = resolved["error"]
            return row
        obj = resolved["object"]
        row["resolve_ok"] = True
        row["module"] = resolved["module"]
        row["object_type"] = type(obj).__name__
        row["callable"] = callable(obj)
        try:
            row["signature"] = short_text(inspect.signature(obj))
        except (TypeError, ValueError):
            row["signature"] = None
        call_row = smoke_call(obj, symbol, timeout_seconds)
        row.update(call_row)
        if row["call_ok"]:
            row["validation_status"] = "call_ok"
        elif row["callable"]:
            row["validation_status"] = "callable_but_not_smoke_called"
        else:
            row["validation_status"] = "resolved_not_callable"
        return row
    except BaseException as exc:
        row["validation_status"] = "worker_error"
        row["error"] = f"{type(exc).__name__}: {short_text(exc)}"
        row["traceback"] = traceback.format_exc(limit=5)
        return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols-json", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=6)
    args = parser.parse_args()

    ensure_runtime_dirs()
    symbols = json.loads(Path(args.symbols_json).read_text(encoding="utf-8"))
    rows = [validate_symbol(symbol, args.timeout_seconds) for symbol in symbols]
    write_jsonl(Path(args.out_jsonl), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
