from __future__ import annotations

from pathlib import Path
from typing import Any
import ast
import json
import re
import tempfile
import textwrap
from .context import (
    contains_hidden_operator_term,
    hidden_operator_terms_to_avoid,
    visible_direct_library_api_hits,
)
from .rules import (
    ALLOWED_FROM_IMPORT_MODULE_NAMES,
    ALLOWED_GENERIC_OPERATOR_WORDS,
    ALLOWED_REQUIRED_IMPORT_MODULES,
    ALLOWED_REQUIRED_IMPORT_TOP_LEVEL,
    BARE_CONSTRUCTOR_OUTPUT_PATTERNS,
    FINAL_OUTPUT_DOMAIN_TERMS,
    FORBIDDEN_STUDENT_DEPENDENCY_TOP_LEVEL,
    FORBIDDEN_VISIBLE_TERMS,
    MAX_GENERAL_TESTS,
    MAX_RAISE_PER_CODE_LINE,
    MAX_RAISE_STATEMENTS,
    MAX_SUBSTEP_TEST_CASES,
    MIN_ANSWER_CHARS,
    MIN_ANSWER_SENTENCES,
    MIN_GENERAL_ASSERTIONS,
    MIN_GENERAL_TESTS,
    MIN_NONTRIVIAL_GENERAL_TESTS,
    MIN_ORACLE_ASSERTIONS,
    MIN_ORACLE_TESTS,
    MIN_STUDENT_CODE_LINES,
    MIN_STUDENT_FUNCTIONS,
    MIN_SUBSTEP_TEST_CASES,
    NONTRIVIAL_GENERAL_TEST_PATTERNS,
    NO_STUDENT_INPUT_PATTERNS,
    PHYSICS_KEYWORDS,
    REQUIRED_TOP_LEVEL_FIELDS,
    SHALLOW_GENERAL_TEST_PATTERNS,
    TEMPLATE_PREFIX_PATTERNS,
)


def canonicalize_problem(problem: dict[str, Any], chain: dict[str, Any]) -> dict[str, Any]:
    item = dict(problem)
    seed_id = str(chain.get("scicode_seed_id") or item.get("problem_id") or "SCICODE-SEED")
    item["problem_id"] = str(item.get("problem_id") or seed_id)
    if item["problem_id"] != seed_id:
        item["problem_id"] = seed_id
    name = str(item.get("problem_name") or seed_id.lower().replace("-", "_"))
    item["problem_name"] = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()
    return item


def count_assertions(code: str) -> int:
    return len(re.findall(r"\bassert\b", code))


def count_raise_statements(code: str) -> int:
    return len(re.findall(r"(?m)^\s*raise\b", code))


def validation_guard_density_errors(code: str) -> list[str]:
    lines = nonempty_code_lines(code)
    if not lines:
        return []
    raise_count = count_raise_statements(code)
    if raise_count <= MAX_RAISE_STATEMENTS:
        return []
    ratio = raise_count / max(1, len(lines))
    if ratio > MAX_RAISE_PER_CODE_LINE:
        return [
            f"general_solution_excessive_validation_raise_guards:{raise_count}_raises_in_{len(lines)}_code_lines"
        ]
    return []


def looks_nontrivial_general_test(code: str) -> bool:
    lowered = code.lower()
    has_assert = "assert" in lowered
    if not has_assert:
        return False
    if any(re.search(pattern, lowered) for pattern in NONTRIVIAL_GENERAL_TEST_PATTERNS):
        return True
    shallow_hits = sum(
        bool(re.search(pattern, lowered)) for pattern in SHALLOW_GENERAL_TEST_PATTERNS
    )
    return shallow_hits == 0 and count_assertions(code) >= 2


def sentence_like_count(text: str) -> int:
    normalized = re.sub(r"\s+", " ", text.strip())
    return len([part for part in re.split(r"[.!?。！？]+", normalized) if len(part.strip()) >= 25])


def has_template_prefix(problem_name: str) -> bool:
    lowered = problem_name.lower()
    return any(re.search(pattern, lowered) for pattern in TEMPLATE_PREFIX_PATTERNS)


def final_function_names(solution: str) -> list[str]:
    names = re.findall(r"(?m)^\s*def\s+([A-Za-z_]\w*)\s*\(", solution)
    return names[-2:] if names else []


def function_names_from_code(code: str) -> list[str]:
    return re.findall(r"(?m)^\s*def\s+([A-Za-z_]\w*)\s*\(", code)


def nonempty_code_lines(code: str) -> list[str]:
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


def quality_difficulty_metrics(problem: dict[str, Any]) -> dict[str, Any]:
    solution = strip_code_fences(str(problem.get("general_solution", "")))
    general_tests = problem.get("general_tests")
    oracle_tests = problem.get("oracle_tests")
    sub_steps = problem.get("sub_steps")

    general_test_texts = (
        [strip_code_fences(str(item)) for item in general_tests]
        if isinstance(general_tests, list)
        else []
    )
    oracle_test_texts = (
        [strip_code_fences(str(item)) for item in oracle_tests]
        if isinstance(oracle_tests, list)
        else []
    )

    sub_step_solution_lines = 0
    sub_step_test_counts: list[int] = []
    sub_step_assert_counts: list[int] = []
    if isinstance(sub_steps, list):
        for step in sub_steps:
            if not isinstance(step, dict):
                continue
            sub_step_solution_lines += len(
                nonempty_code_lines(strip_code_fences(str(step.get("ground_truth_code", ""))))
            )
            tests = step.get("test_cases")
            if isinstance(tests, list):
                sub_step_test_counts.append(len(tests))
                sub_step_assert_counts.append(
                    sum(count_assertions(strip_code_fences(str(test))) for test in tests)
                )
            else:
                sub_step_test_counts.append(0)
                sub_step_assert_counts.append(0)

    solution_lines = len(nonempty_code_lines(solution))
    raise_count = count_raise_statements(solution)
    return {
        "student_general_solution_nonempty_lines": solution_lines,
        "student_general_solution_function_defs": len(function_names_from_code(solution)),
        "student_general_solution_raise_statements": raise_count,
        "student_validation_raise_ratio": round(raise_count / max(1, solution_lines), 3),
        "sub_step_count": len(sub_steps) if isinstance(sub_steps, list) else 0,
        "sub_step_solution_nonempty_lines_total": sub_step_solution_lines,
        "sub_step_test_snippets_total": sum(sub_step_test_counts),
        "sub_step_test_snippets_min_per_step": min(sub_step_test_counts)
        if sub_step_test_counts
        else 0,
        "sub_step_test_assertions_total": sum(sub_step_assert_counts),
        "general_test_snippets": len(general_test_texts),
        "general_test_assertions": sum(count_assertions(text) for text in general_test_texts),
        "general_test_nontrivial_snippets": sum(
            1 for text in general_test_texts if looks_nontrivial_general_test(text)
        ),
        "oracle_test_snippets": len(oracle_test_texts),
        "oracle_test_assertions": sum(count_assertions(text) for text in oracle_test_texts),
    }


def import_module_path(node: ast.AST) -> str:
    if isinstance(node, ast.Import):
        return ""
    if isinstance(node, ast.ImportFrom):
        return str(node.module or "")
    return ""


def required_dependency_import_errors(code: str) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        return [f"required_dependencies_syntax_error:{exc.msg}"]
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            if not (
                isinstance(node, ast.Expr)
                and isinstance(getattr(node, "value", None), ast.Constant)
            ):
                errors.append(f"required_dependencies_non_import_statement:{type(node).__name__}")
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                top = module.split(".", 1)[0].lower()
                if top in FORBIDDEN_STUDENT_DEPENDENCY_TOP_LEVEL:
                    errors.append(f"forbidden_dependency:{module}")
                elif top not in ALLOWED_REQUIRED_IMPORT_TOP_LEVEL:
                    errors.append(f"required_dependency_not_allowed:{module}")
                elif module.lower() not in ALLOWED_REQUIRED_IMPORT_MODULES and top in {
                    "numpy",
                    "scipy",
                    "sympy",
                }:
                    errors.append(f"required_dependency_module_too_specific:{module}")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").lower()
            top = module.split(".", 1)[0]
            if top in FORBIDDEN_STUDENT_DEPENDENCY_TOP_LEVEL:
                errors.append(f"forbidden_dependency:{module}")
                continue
            if (
                top not in ALLOWED_REQUIRED_IMPORT_TOP_LEVEL
                or module not in ALLOWED_REQUIRED_IMPORT_MODULES
            ):
                errors.append(f"required_dependency_not_allowed:{module}")
                continue
            for alias in node.names:
                name = alias.name
                if name == "*":
                    errors.append(f"required_dependency_star_import:{module}")
                elif module in {"scipy"} and name not in ALLOWED_FROM_IMPORT_MODULE_NAMES:
                    errors.append(f"required_dependency_direct_function_import:{module}.{name}")
                elif module == "scipy.sparse" and name == "linalg":
                    continue
                elif module in {"numpy", "sympy"}:
                    errors.append(f"required_dependency_direct_function_import:{module}.{name}")
                elif module.startswith(("scipy.", "numpy.", "sympy.")):
                    errors.append(f"required_dependency_direct_function_import:{module}.{name}")
    return errors


def solution_import_errors(code: str, *, label: str) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            errors.append(f"{label}_contains_import_statement")
            break
    return errors


def ast_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = ast_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def call_names_from_code(code: str) -> set[str]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast_call_name(node.func)
            if name:
                names.add(name)
    return names


def hidden_api_call_terms(chain: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    symbols: list[str] = []
    for node in chain.get("operator_nodes", []) or []:
        if isinstance(node, dict):
            symbols.append(str(node.get("api_symbol") or ""))
    for edge in chain.get("edge_trace", []) or []:
        if isinstance(edge, dict):
            symbols.append(str(edge.get("source_api_symbol") or ""))
            symbols.append(str(edge.get("target_api_symbol") or ""))
    for symbol in symbols:
        cleaned = symbol.strip()
        if not cleaned:
            continue
        normalized = cleaned.lower()
        terms.add(normalized)
        terminal = re.split(r"[^0-9A-Za-z_]+", cleaned)[-1].strip().lower()
        if (
            len(terminal) >= 4
            and terminal not in ALLOWED_GENERIC_OPERATOR_WORDS
            and terminal not in {"array", "table", "value", "model", "data", "time"}
        ):
            terms.add(terminal)
    return terms


def student_hidden_api_call_errors(problem: dict[str, Any], chain: dict[str, Any]) -> list[str]:
    terms = hidden_api_call_terms(chain)
    if not terms:
        return []
    code_chunks = [strip_code_fences(str(problem.get("general_solution", "")))]
    for step in problem.get("sub_steps") or []:
        if isinstance(step, dict):
            code_chunks.append(strip_code_fences(str(step.get("ground_truth_code", ""))))
    call_names: set[str] = set()
    for chunk in code_chunks:
        call_names.update(name.lower() for name in call_names_from_code(chunk))
    errors: list[str] = []
    for name in sorted(call_names):
        suffix = name.rsplit(".", 1)[-1]
        if name in terms or suffix in terms:
            errors.append(f"student_code_direct_hidden_api_call:{name}")
    return errors


def code_calls_function(text: str, function_name: str) -> bool:
    if not function_name:
        return False
    return bool(re.search(rf"\b{re.escape(function_name)}\s*\(", text))


def oracle_test_has_differential_check(text: str, student_name: str, oracle_name: str) -> bool:
    if not (code_calls_function(text, student_name) and code_calls_function(text, oracle_name)):
        return False
    lowered = text.lower()
    comparison_markers = [
        "allclose",
        "isclose",
        "array_equal",
        "assert_close",
        "cmp_tuple_or_list",
        "student",
        "reference",
    ]
    if any(marker in lowered for marker in comparison_markers):
        return True
    return bool(re.search(r"assert\s+.+(==|<=|>=|<|>)", text, flags=re.S))


def helper_name_leak_errors(problem: dict[str, Any], chain: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    names: list[str] = []
    names.extend(
        function_names_from_code(strip_code_fences(str(problem.get("general_solution", ""))))
    )
    for step in problem.get("sub_steps") or []:
        if isinstance(step, dict):
            names.extend(
                function_names_from_code(strip_code_fences(str(step.get("ground_truth_code", ""))))
            )
            header = str(step.get("function_header", ""))
            names.extend(function_names_from_code(header))
    hidden_terms = hidden_operator_terms_to_avoid(chain)
    for name in sorted(set(names)):
        lowered = name.lower()
        for term in hidden_terms:
            compact = re.sub(r"[^a-z0-9]+", "", term.lower())
            if compact and compact in re.sub(r"[^a-z0-9]+", "", lowered):
                errors.append(f"helper_function_name_leaked_operator:{name}:{term}")
                break
    return errors


def substep_dependency_errors(problem: dict[str, Any]) -> list[str]:
    steps = problem.get("sub_steps")
    if not isinstance(steps, list):
        return []
    solution = strip_code_fences(str(problem.get("general_solution", "")))
    errors: list[str] = []
    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        code = strip_code_fences(str(step.get("ground_truth_code", "")))
        names = function_names_from_code(code)
        if not names:
            continue
        helper = names[-1]
        later_chunks = []
        for later in steps[idx:]:
            if isinstance(later, dict):
                later_chunks.append(strip_code_fences(str(later.get("ground_truth_code", ""))))
                later_chunks.append(str(later.get("step_description_prompt", "")))
        later_chunks.append(solution)
        later_text = "\n".join(later_chunks)
        if not re.search(rf"\b{re.escape(helper)}\s*\(", later_text):
            if idx < len(steps):
                errors.append(f"sub_step_{idx}_helper_not_used_downstream:{helper}")
    return errors


def parameter_sensitivity_evidence_missing(problem: dict[str, Any]) -> bool:
    solution = strip_code_fences(str(problem.get("general_solution", "")))
    final_names = final_function_names(solution)
    tests = problem.get("general_tests") if isinstance(problem.get("general_tests"), list) else []
    test_blob = "\n".join(strip_code_fences(str(test)) for test in tests)
    lowered = test_blob.lower()
    if any(
        word in lowered for word in ["sensitivity", "perturb", "changed", "changes", "different"]
    ):
        if any(name and name in test_blob for name in final_names):
            return False
    for name in final_names:
        calls = re.findall(rf"\b{re.escape(name)}\s*\(([^)]*)\)", test_blob, flags=re.S)
        normalized_calls = {re.sub(r"\s+", "", call) for call in calls}
        if len(normalized_calls) >= 2:
            return False
    return True


def parameter_sensitivity_runtime_error(
    *,
    dependencies: str,
    solution: str,
    general_tests: list[str],
    timeout: int,
    validation_python: str,
) -> str:
    final_names = final_function_names(solution)
    if not final_names:
        return "parameter_sensitivity_runtime_missing_final_function"
    final_name = final_names[-1]
    test_list_json = json.dumps([strip_code_fences(str(test)) for test in general_tests])
    checker = textwrap.dedent(
        f"""
        _SCICODE_ORIGINAL_FINAL = {final_name}
        _SCICODE_CALLS = []

        def _scicode_numeric_array(value):
            try:
                arr = np.asarray(value)
                if arr.dtype.kind not in "biufc":
                    return None
                arr = arr.astype(np.complex128).reshape(-1)
                if arr.size > 10000:
                    arr = arr[:10000]
                if not np.all(np.isfinite(arr.real)) or not np.all(np.isfinite(arr.imag)):
                    return None
                return arr
            except Exception:
                return None

        def _scicode_value_changed(left, right):
            left_arr = _scicode_numeric_array(left)
            right_arr = _scicode_numeric_array(right)
            if left_arr is not None and right_arr is not None:
                if left_arr.shape != right_arr.shape:
                    return True
                return not np.allclose(left_arr, right_arr, rtol=1e-7, atol=1e-9)
            return repr(left) != repr(right)

        def _scicode_arg_signature(args, kwargs):
            parts = []
            for value in args:
                arr = _scicode_numeric_array(value)
                if arr is None:
                    parts.append(repr(value))
                else:
                    parts.append((arr.shape, tuple(np.round(arr[:32], 10))))
            for key in sorted(kwargs):
                value = kwargs[key]
                arr = _scicode_numeric_array(value)
                if arr is None:
                    parts.append((key, repr(value)))
                else:
                    parts.append((key, arr.shape, tuple(np.round(arr[:32], 10))))
            return repr(parts)

        def {final_name}(*args, **kwargs):
            output = _SCICODE_ORIGINAL_FINAL(*args, **kwargs)
            _SCICODE_CALLS.append((_scicode_arg_signature(args, kwargs), output))
            return output

        for _scicode_test_code in {test_list_json}:
            exec(_scicode_test_code, globals())

        _SCICODE_CHANGED = False
        for _i in range(len(_SCICODE_CALLS)):
            for _j in range(_i + 1, len(_SCICODE_CALLS)):
                if _SCICODE_CALLS[_i][0] != _SCICODE_CALLS[_j][0] and _scicode_value_changed(_SCICODE_CALLS[_i][1], _SCICODE_CALLS[_j][1]):
                    _SCICODE_CHANGED = True
                    break
            if _SCICODE_CHANGED:
                break
        assert len(_SCICODE_CALLS) >= 2, "final function must be called in at least two general tests"
        assert _SCICODE_CHANGED, "different final-function inputs did not produce a changed final output"
        """
    ).strip()
    return run_python(
        dependencies + "\n\n" + solution + "\n\n" + checker,
        timeout,
        validation_python,
    )


def hardcoded_random_seed_errors(problem: dict[str, Any]) -> list[str]:
    solution = strip_code_fences(str(problem.get("general_solution", "")))
    problem_io = str(problem.get("problem_io", "")).lower()
    if "seed" in problem_io or "random_state" in problem_io or "rng" in problem_io:
        return []
    patterns = [
        r"\bnp\.random\.seed\s*\(\s*[0-9]+",
        r"\bnumpy\.random\.seed\s*\(\s*[0-9]+",
        r"\bdefault_rng\s*\(\s*[0-9]+",
        r"\bRandomState\s*\(\s*[0-9]+",
    ]
    if any(re.search(pattern, solution) for pattern in patterns):
        return ["hardcoded_random_seed_without_seed_input"]
    return []


def output_domain_meaning_errors(problem: dict[str, Any], returns_section: str) -> list[str]:
    errors: list[str] = []
    text = "\n".join(
        [
            str(problem.get("problem_name", "")),
            returns_section,
            str(problem.get("problem_answer_main", "")),
            str(problem.get("problem_description_main", ""))[:1200],
        ]
    ).lower()
    if any(token in text for token in ["bool", "boolean", "true/false", "flag"]):
        if not any(term in text for term in FINAL_OUTPUT_DOMAIN_TERMS):
            errors.append("final_output_boolean_without_numeric_diagnostic")
    if not any(term in text for term in FINAL_OUTPUT_DOMAIN_TERMS):
        errors.append("final_output_not_named_domain_quantity")
    return errors


def validate_problem(
    problem: dict[str, Any],
    chain: dict[str, Any],
    timeout: int,
    validation_python: str,
) -> list[str]:
    errors: list[str] = []
    for location, tests in [
        ("general_tests", problem.get("general_tests")),
        ("oracle_tests", problem.get("oracle_tests")),
    ]:
        if isinstance(tests, list):
            for index, test in enumerate(tests):
                error = test_snippet_error(test)
                if error:
                    errors.append(f"{location}_{index + 1}:{error}")
    for index, step in enumerate(problem.get("sub_steps") or []):
        if isinstance(step, dict) and isinstance(step.get("test_cases"), list):
            for case_index, test in enumerate(step["test_cases"]):
                error = test_snippet_error(test)
                if error:
                    errors.append(f"sub_step_{index + 1}_test_{case_index + 1}:{error}")
    if errors:
        return errors
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in problem:
            errors.append(f"missing_top_level_field:{field}")
    if errors:
        return errors

    for field in [
        "problem_description_main",
        "problem_background_main",
        "problem_io",
        "required_dependencies",
        "general_solution",
        "oracle_solution",
        "problem_answer_main",
    ]:
        if not isinstance(problem.get(field), str) or not problem[field].strip():
            errors.append(f"empty_or_non_string:{field}")

    visible_text = "\n".join(
        str(problem.get(field, ""))
        for field in [
            "problem_name",
            "problem_description_main",
            "problem_background_main",
            "problem_io",
            "problem_answer_main",
        ]
    )
    sub_steps = problem.get("sub_steps")
    if isinstance(sub_steps, list):
        for step in sub_steps:
            if isinstance(step, dict):
                visible_text += "\n" + "\n".join(
                    str(step.get(field, ""))
                    for field in ["step_description_prompt", "step_background"]
                )

    lower_visible = visible_text.lower()
    if has_template_prefix(str(problem.get("problem_name", ""))):
        errors.append(f"problem_name_template_prefix:{problem.get('problem_name')}")
    for term in FORBIDDEN_VISIBLE_TERMS:
        if term.lower() in lower_visible:
            errors.append(f"forbidden_visible_term:{term}")
    for term in hidden_operator_terms_to_avoid(chain):
        if contains_hidden_operator_term(lower_visible, term):
            errors.append(f"leaked_operator_name:{term}")

    if not any(keyword in lower_visible for keyword in PHYSICS_KEYWORDS):
        errors.append("problem_not_clearly_scientific_domain_facing")
    if len(str(problem.get("problem_description_main", ""))) < 500:
        errors.append("problem_description_too_short_for_scicode_style")
    desc_lower = str(problem.get("problem_description_main", "")).lower()
    for phrase in ["write a python function", "implement a function", "write a function"]:
        if phrase in desc_lower:
            errors.append(f"main_problem_is_too_function_implementation_framed:{phrase}")
    for phrase in ["write a python function", "implement a function", "write a function"]:
        if phrase in lower_visible:
            errors.append(f"visible_problem_is_too_function_implementation_framed:{phrase}")
    for hit in visible_direct_library_api_hits(visible_text)[:5]:
        errors.append(f"visible_text_direct_library_api_instruction:{hit}")

    all_student_text = (
        visible_text
        + "\n"
        + "\n".join(
            str(problem.get(field, "")) for field in ["required_dependencies", "general_solution"]
        )
    )
    if isinstance(sub_steps, list):
        for step in sub_steps:
            if isinstance(step, dict):
                all_student_text += "\n" + "\n".join(
                    str(step.get(field, ""))
                    for field in [
                        "ground_truth_code",
                        "function_header",
                        "return_line",
                    ]
                )
                tests = step.get("test_cases")
                if isinstance(tests, list):
                    all_student_text += "\n" + "\n".join(str(test) for test in tests)
    lower_all_student_text = all_student_text.lower()
    errors.extend(helper_name_leak_errors(problem, chain))
    for phrase in ["placeholder", "student must implement", "you may return"]:
        if phrase in lower_all_student_text:
            errors.append(f"student_artifact_contains_placeholder:{phrase}")
    if re.search(r"return\s+\[\s*['\"]u['\"]\s*\]", lower_all_student_text):
        errors.append("student_artifact_contains_placeholder_gate_return")
    # The student-facing prose and helper names must not reveal the hidden chain.
    # The executable solution/tests may still use ordinary NumPy/SciPy/SymPy calls;
    # otherwise common words such as solve, det, variance, or spectrum cause large
    # false-positive repair loops.

    problem_io = str(problem.get("problem_io", ""))
    if "Parameters" not in problem_io or "Returns" not in problem_io:
        errors.append("problem_io_missing_parameters_or_returns")
    parameters_section = (
        problem_io.split("Parameters", 1)[-1].split("Returns", 1)[0].strip()
        if "Parameters" in problem_io
        else ""
    )
    normalized_parameters = re.sub(r"\s+", " ", parameters_section).strip().lower()
    if not normalized_parameters or any(
        re.search(pattern, normalized_parameters, flags=re.I)
        for pattern in NO_STUDENT_INPUT_PATTERNS
    ):
        errors.append("problem_has_no_meaningful_student_inputs")
    returns_section = problem_io.split("Returns", 1)[-1].strip() if "Returns" in problem_io else ""
    if not returns_section or re.fullmatch(r'[:\s"`\']*result\s*[:\s"`\']*', returns_section, re.I):
        errors.append("problem_io_return_is_ambiguous")
    if "result" in returns_section.lower() and len(returns_section) < 80:
        errors.append("problem_io_return_result_not_described")

    answer = str(problem.get("problem_answer_main", ""))
    deliverable_text = "\n".join(
        [
            str(problem.get("problem_name", "")),
            returns_section,
            answer,
        ]
    ).lower()
    bare_constructor_hits = [
        pattern for pattern in BARE_CONSTRUCTOR_OUTPUT_PATTERNS if pattern in deliverable_text
    ]
    if bare_constructor_hits:
        errors.append(
            "final_output_is_bare_constructor_like:" + ",".join(bare_constructor_hits[:4])
        )
    errors.extend(output_domain_meaning_errors(problem, returns_section))

    if "result" in answer.lower() and len(answer) < 120:
        errors.append("problem_answer_main_output_ambiguous")
    if len(answer.strip()) < MIN_ANSWER_CHARS:
        errors.append("problem_answer_main_too_short")
    if sentence_like_count(answer) < MIN_ANSWER_SENTENCES:
        errors.append("problem_answer_main_too_few_sentences")
    if len(answer.strip()) < MIN_ANSWER_CHARS or sentence_like_count(answer) < MIN_ANSWER_SENTENCES:
        errors.append("problem_answer_main_too_thin")

    dependencies = strip_code_fences(str(problem.get("required_dependencies", "")))
    if "import numpy as np" not in dependencies:
        errors.append("required_dependencies_missing_numpy_alias")
    errors.extend(required_dependency_import_errors(dependencies))

    if not isinstance(sub_steps, list) or len(sub_steps) < 3:
        errors.append("sub_steps_must_have_at_least_three_steps")
    elif len(sub_steps) > 6:
        errors.append("sub_steps_too_many_for_seed_generation")
    else:
        errors.extend(substep_dependency_errors(problem))
        cumulative_code = dependencies + "\n\n"
        for idx, step in enumerate(sub_steps, start=1):
            if not isinstance(step, dict):
                errors.append(f"sub_step_{idx}_not_object")
                continue
            for field in [
                "step_number",
                "step_description_prompt",
                "step_background",
                "ground_truth_code",
                "function_header",
                "test_cases",
                "return_line",
            ]:
                if field not in step:
                    errors.append(f"sub_step_{idx}_missing:{field}")
            code = strip_code_fences(str(step.get("ground_truth_code", "")))
            if "def " not in code:
                errors.append(f"sub_step_{idx}_ground_truth_missing_def")
            if "target" in code:
                errors.append(f"sub_step_{idx}_ground_truth_uses_target")
            cumulative_code += "\n\n" + code + "\n"
            tests = step.get("test_cases", [])
            if not isinstance(tests, list) or not tests:
                errors.append(f"sub_step_{idx}_missing_test_cases")
                continue
            if len(tests) < MIN_SUBSTEP_TEST_CASES:
                errors.append(f"sub_step_{idx}_too_few_test_cases")
            if len(tests) > MAX_SUBSTEP_TEST_CASES:
                errors.append(f"sub_step_{idx}_too_many_test_cases")
            for case_idx, case in enumerate(tests, start=1):
                case_text = strip_code_fences(str(case))
                if "target" in case_text:
                    errors.append(f"sub_step_{idx}_test_{case_idx}_uses_target")
                    continue
                if count_assertions(case_text) < 1:
                    errors.append(f"sub_step_{idx}_test_{case_idx}_has_no_assert")
                    continue
                run_error = run_python(
                    cumulative_code + "\n\n" + case_text,
                    timeout,
                    validation_python,
                )
                if run_error:
                    if run_error == "timeout":
                        errors.append(f"sub_step_{idx}_test_{case_idx}_timeout")
                    else:
                        errors.append(f"sub_step_{idx}_test_{case_idx}_failed:{run_error}")
                    break

    solution = strip_code_fences(str(problem.get("general_solution", "")))
    general_tests = problem.get("general_tests")
    oracle_dependencies = strip_code_fences(str(problem.get("oracle_dependencies", "")))
    oracle_solution = strip_code_fences(str(problem.get("oracle_solution", "")))
    oracle_tests = problem.get("oracle_tests")
    errors.extend(hardcoded_random_seed_errors(problem))
    if "def " not in solution:
        errors.append("general_solution_missing_function_definitions")
    student_function_names = function_names_from_code(solution)
    if len(student_function_names) < MIN_STUDENT_FUNCTIONS:
        errors.append("general_solution_too_few_functions")
    if len(nonempty_code_lines(solution)) < MIN_STUDENT_CODE_LINES:
        errors.append("general_solution_too_short")
    errors.extend(validation_guard_density_errors(solution))
    errors.extend(solution_import_errors(solution, label="general_solution"))
    if "target" in solution:
        errors.append("general_solution_uses_target")
    errors.extend(student_hidden_api_call_errors(problem, chain))
    if not isinstance(general_tests, list) or not general_tests:
        errors.append("general_tests_missing")
    else:
        if len(general_tests) < MIN_GENERAL_TESTS:
            errors.append("general_tests_too_few")
        if len(general_tests) > MAX_GENERAL_TESTS:
            errors.append("general_tests_too_many_not_scicode_like")
        test_code_parts: list[str] = []
        assertion_count = 0
        nontrivial_test_count = 0
        for idx, test in enumerate(general_tests, start=1):
            text = strip_code_fences(str(test))
            if "target" in text:
                errors.append(f"general_test_{idx}_uses_target")
            assertion_count += count_assertions(text)
            if looks_nontrivial_general_test(text):
                nontrivial_test_count += 1
            test_code_parts.append(text)
        if assertion_count < MIN_GENERAL_ASSERTIONS:
            errors.append("general_tests_too_few_assertions")
        if nontrivial_test_count < MIN_NONTRIVIAL_GENERAL_TESTS:
            errors.append("general_tests_too_shallow")
        if parameter_sensitivity_evidence_missing(problem):
            errors.append("parameter_sensitivity_test_missing")
        run_error = run_python(
            dependencies + "\n\n" + solution + "\n\n" + "\n\n".join(test_code_parts),
            timeout,
            validation_python,
        )
        if run_error:
            if run_error == "timeout":
                errors.append("general_tests_timeout")
            else:
                errors.append(f"general_tests_failed:{run_error}")
        sensitivity_error = parameter_sensitivity_runtime_error(
            dependencies=dependencies,
            solution=solution,
            general_tests=test_code_parts,
            timeout=timeout,
            validation_python=validation_python,
        )
        if sensitivity_error:
            if sensitivity_error == "timeout":
                errors.append("parameter_sensitivity_runtime_timeout")
            else:
                errors.append(f"parameter_sensitivity_runtime_failed:{sensitivity_error}")

    if "def " not in oracle_solution:
        errors.append("oracle_solution_missing_function_definitions")
    errors.extend(solution_import_errors(oracle_solution, label="oracle_solution"))
    oracle_function_names = function_names_from_code(oracle_solution)
    final_student_name = student_function_names[-1] if student_function_names else ""
    if final_student_name and not any(
        name == f"oracle_{final_student_name}" for name in oracle_function_names
    ):
        errors.append(f"oracle_solution_missing_oracle_final_function:oracle_{final_student_name}")
    if not isinstance(oracle_tests, list) or not oracle_tests:
        errors.append("oracle_tests_missing")
    else:
        if len(oracle_tests) < MIN_ORACLE_TESTS:
            errors.append("oracle_tests_too_few")
        oracle_test_parts: list[str] = []
        oracle_assertion_count = 0
        for idx, test in enumerate(oracle_tests, start=1):
            text = strip_code_fences(str(test))
            if "target" in text:
                errors.append(f"oracle_test_{idx}_uses_target")
            if final_student_name and not code_calls_function(text, final_student_name):
                errors.append(f"oracle_test_{idx}_does_not_call_student_final:{final_student_name}")
            oracle_final_name_for_item = (
                f"oracle_{final_student_name}" if final_student_name else ""
            )
            if oracle_final_name_for_item and not code_calls_function(
                text, oracle_final_name_for_item
            ):
                errors.append(
                    f"oracle_test_{idx}_does_not_call_oracle_final:{oracle_final_name_for_item}"
                )
            if count_assertions(text) < 2:
                errors.append(f"oracle_test_{idx}_too_few_assertions")
            oracle_assertion_count += count_assertions(text)
            oracle_test_parts.append(text)
        if oracle_assertion_count < MIN_ORACLE_ASSERTIONS:
            errors.append("oracle_tests_too_few_assertions")
        joined_oracle_tests = "\n\n".join(oracle_test_parts)
        oracle_final_name = f"oracle_{final_student_name}" if final_student_name else ""
        if final_student_name and not code_calls_function(joined_oracle_tests, final_student_name):
            errors.append(f"oracle_tests_do_not_call_student_final:{final_student_name}")
        if oracle_final_name and not code_calls_function(joined_oracle_tests, oracle_final_name):
            errors.append(f"oracle_tests_do_not_call_oracle_final:{oracle_final_name}")
        if final_student_name and oracle_final_name:
            paired_checks = sum(
                1
                for text in oracle_test_parts
                if oracle_test_has_differential_check(text, final_student_name, oracle_final_name)
            )
            if paired_checks < min(MIN_ORACLE_TESTS, len(oracle_test_parts)):
                errors.append(f"oracle_tests_too_few_paired_differential_checks:{paired_checks}")
        oracle_run_error = run_python(
            dependencies
            + "\n\n"
            + solution
            + "\n\n"
            + oracle_dependencies
            + "\n\n"
            + oracle_solution
            + "\n\n"
            + "\n\n".join(oracle_test_parts),
            timeout,
            validation_python,
        )
        if oracle_run_error:
            if oracle_run_error == "timeout":
                errors.append("oracle_tests_timeout")
            else:
                errors.append(f"oracle_tests_failed:{oracle_run_error}")

    return errors


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def run_python(code: str, timeout: int, python_executable: str) -> str:
    """Execute a test in a fresh working directory with a wall-clock timeout.

    This isolates temporary files; it is not a security sandbox.
    """
    from .execution import execute_python

    with tempfile.TemporaryDirectory(prefix="scidata-test-") as workdir:
        script = Path(workdir) / "test.py"
        script.write_text(code, encoding="utf-8")
        return execute_python(script, timeout, python_executable)


def test_snippet_error(value: Any) -> str:
    """Reject non-code test containers and assertions hidden in strings/comments."""
    if not isinstance(value, str):
        return "test_must_be_a_string"
    try:
        source = (
            strip_code_fences(value) if value.lstrip().startswith("```") else value.strip("\r\n")
        )
        tree = ast.parse(source)
    except SyntaxError:
        return "test_is_not_valid_module_code"

    def has_check(nodes):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(node, ast.Assert):
                return True
            if isinstance(node, ast.Call) and ast_call_name(node.func).split(".")[-1].startswith(
                "assert_"
            ):
                return True
            if has_check(list(ast.iter_child_nodes(node))):
                return True
        return False

    return "" if has_check(tree.body) else "test_has_no_executable_assertion"
