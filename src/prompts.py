from __future__ import annotations

from typing import Any
import json
import textwrap
from .context import (
    background_options_for_chain,
    chain_context_for_prompt,
    summarize_diversity_context,
)
from .rules import (
    ALLOWED_IMPORT_HINT,
    MIN_GENERAL_ASSERTIONS,
    MIN_ORACLE_ASSERTIONS,
    MIN_STUDENT_CODE_LINES,
    MIN_STUDENT_FUNCTIONS,
    QUALITY_DIMENSION_MIN_SCORES,
    QUALITY_FEWSHOT_SUMMARY,
    SA_RUBRIC_TEXT,
)
from .validation import (
    quality_difficulty_metrics,
)


def build_background_selection_prompt(
    chain: dict[str, Any],
    rejected: list[dict[str, Any]],
    *,
    candidate_count: int,
    diversity_context: dict[str, Any],
) -> tuple[str, str]:
    system_prompt = textwrap.dedent(
        """
        You choose a natural subdomain-specific setting for a hidden scientific operator chain.
        Return one valid JSON object only.
        """
    ).strip()
    chain_info = chain_context_for_prompt(chain)
    background_options = "\n".join(
        f"- {item['id']}: {item['title']} - {item['description']}"
        for item in background_options_for_chain(chain)
    )
    rejected_summary = [
        {
            "background_id": item.get("proposal", {}).get("background_id"),
            "reason": item.get("judge", {}).get("reason") or item.get("error") or "",
            "revision_hint": item.get("judge", {}).get("revision_hint", ""),
        }
        for item in rejected[-5:]
    ]
    diversity_summary = summarize_diversity_context(diversity_context)
    user_prompt = textwrap.dedent(
        f"""
        Hidden chain context:
        {json.dumps(chain_info, ensure_ascii=False, indent=2)}

        Subdomain-specific background directions:
        {background_options}

        Previously rejected choices, if any:
        {json.dumps(rejected_summary, ensure_ascii=False, indent=2)}

        Diversity history from already accepted/generated problems:
        {json.dumps(diversity_summary, ensure_ascii=False, indent=2)}

        Task:
        Propose {candidate_count} distinct background candidates that make the hidden chain
        feel like a natural workflow in this exact `subdomain`, rather than a decorative
        story. Use the subdomain-specific directions above as examples, but do not copy
        them mechanically. If the hidden chain suggests a better setting in the same
        subdomain, propose that instead. Do not force a physics setting unless this
        subdomain is physics-specific.

        If `generation_variant_index` is present, this chain is intentionally being used
        multiple times because it has high potential. For this variant, choose a background
        and observable that are materially different from other likely variants of the same
        hidden chain, while still remaining natural for the computation.

        Requirements:
        - The selected background must explain why each hidden operator node can correspond
          to a meaningful domain role.
        - Do not write the final problem. Only choose and justify the scientific background.
        - Operator ids are hidden implementation inspiration; they must not be exposed
          in the eventual student-facing problem.
        - Prefer a background with concrete inputs, observable outputs, and a plausible
          sequence of computational steps.
        - Avoid a vague story where the operators are merely attached after the fact.
        - Avoid repeating overused background ids, titles, observables, and problem-name
          prefixes shown in the diversity history. Prefer less-used domains when natural.
        - Make the candidates meaningfully different from one another.

        Return JSON with exactly this shape:
        {{
          "candidates": [
            {{
              "background_id": "one_catalog_id_or_combined_id",
              "background_title": "short title",
              "background_domain": "scientific domain",
              "domain_story": "concrete scenario in 3-5 sentences",
              "observable_to_compute": "named domain output and type/shape",
              "input_quantities": ["quantity 1", "quantity 2"],
              "operator_role_mapping": [
                {{
                  "operator_id": "hidden operator id",
                  "scientific_role": "natural role in this workflow without exposing it to students"
                }}
              ],
              "why_natural": "why this background is not forced",
              "student_visible_terms_to_avoid": ["terms that would reveal hidden operators"]
            }}
          ]
        }}
        """
    ).strip()
    return system_prompt, user_prompt


def build_background_judge_prompt(
    chain: dict[str, Any],
    proposals: list[dict[str, Any]],
    diversity_context: dict[str, Any],
) -> tuple[str, str]:
    system_prompt = textwrap.dedent(
        """
        You are a strict reviewer of scientific/domain backgrounds for SciCode-style coding problems.
        Return one valid JSON object only.
        """
    ).strip()
    chain_info = chain_context_for_prompt(chain)
    user_prompt = textwrap.dedent(
        f"""
        Hidden chain context:
        {json.dumps(chain_info, ensure_ascii=False, indent=2)}

        Proposed scientific background candidates:
        {json.dumps(proposals, ensure_ascii=False, indent=2)}

        Diversity history:
        {json.dumps(summarize_diversity_context(diversity_context), ensure_ascii=False, indent=2)}

        Judge all candidates and select the best one for a high-quality SciCode-style
        problem in the chain's subdomain. Prefer the candidate that is both natural
        for the hidden operator graph walk and least template-like relative to the
        diversity history.

        Accept if:
        - The scenario is scientifically meaningful for the stated subdomain.
        - The observable is concrete and computable.
        - The hidden operator nodes have plausible scientific roles.
        - The story is not just a generic wrapper around unrelated operations.
        - The final problem can hide operator ids while still being clear.

        Reject if:
        - The operator roles are unrelated to the domain story.
        - The proposed output is vague or not scientifically interpretable.
        - The background would likely force the final problem to mention hidden operators.
        - The problem would mostly be about calling library functions rather than doing
          a scientific calculation.

        Domain words such as quantum, density matrix, mutual information, spin,
        entropy, wave, diffusion, or oscillator are allowed when they are natural.

        Use `score` on a 1-10 scale, where 1 means unusable/forced, 4 means
        barely acceptable, 7 means natural, and 10 means an excellent fit.

        Return JSON:
        {{
          "verdict": "accept or reject",
          "selected_index": 0,
          "score": 7,
          "is_forced": false,
          "template_risk": "low/medium/high",
          "diversity_risk": "low/medium/high",
          "reason": "short explanation",
          "candidate_reviews": [
            {{"index": 0, "score": 7, "pros": ["..."], "cons": ["..."]}}
          ],
          "failure_modes": ["only if rejected"],
          "revision_hint": "what kind of background would fit better"
        }}
        """
    ).strip()
    return system_prompt, user_prompt


def build_chain_potential_prompt(chain: dict[str, Any]) -> tuple[str, str]:
    system_prompt = textwrap.dedent(
        """
        You judge whether a hidden per-subdomain scientific operator graph walk is
        worth using as the seed for an S/A-quality SciCode-style coding problem.
        Return JSON only.
        """
    ).strip()
    chain_info = chain_context_for_prompt(chain)
    user_prompt = textwrap.dedent(
        f"""
        Hidden chain:
        {json.dumps(chain_info, ensure_ascii=False, indent=2)}

        Decide whether this chain has enough S/A potential. A high-potential chain
        should naturally support:
        - a meaningful domain output, not only a helper/constructor/boolean;
        - at least three dependent scientific sub-steps;
        - concrete student inputs that genuinely affect the final output;
        - a student implementation using only NumPy/SciPy/math/similar base libraries,
          without exposing hidden operator names or importing specialized domain packages;
        - SciCode-style tests: roughly 3 executable test snippets per sub-step, plus
          4-7 final/general snippets with numeric comparisons, invariants, or small
          deterministic simulation checks.

        Low-potential chains often force a zero/identity/basis-vector final output,
        require named SDK/package semantics to explain, only check shape/type, or are
        just a library API composition with no natural SciCode-style deliverable.

        Reject or give a low score if a reasonable problem would require students to
        directly call one of the hidden specialized operators, rather than rebuilding
        the needed behavior from lower-level numerical steps.
        `operator_runtime_env_id` and `operator_runtime_python` are for hidden operator
        inspection only; they must not become student-facing dependencies.

        Return JSON:
        {{
          "verdict": "accept/reject/uncertain",
          "score": 0,
          "likely_grade_ceiling": "S/A/B/C/D",
          "can_make_physical_observable": true,
          "constructor_or_boolean_risk": "low/medium/high",
          "api_leakage_risk": "low/medium/high",
          "natural_substep_potential": "low/medium/high",
          "reason": "2-4 sentences",
          "suggested_backgrounds": ["short domain suggestion"],
          "avoid_as_final_output": ["constructor/helper/boolean if any"]
        }}
        """
    ).strip()
    return system_prompt, user_prompt


def build_quality_judge_prompt(
    *,
    problem: dict[str, Any],
    chain: dict[str, Any],
    selected_background: dict[str, Any] | None,
    min_grade: str,
) -> tuple[str, str]:
    system_prompt = textwrap.dedent(
        """
        You are a strict human-quality reviewer for SciCode-style scientific coding problems.
        Return one valid JSON object only.
        """
    ).strip()
    chain_info = chain_context_for_prompt(chain)
    chain_info["selected_domain_background"] = selected_background or {}
    visible_problem = {
        key: problem.get(key)
        for key in [
            "problem_name",
            "problem_id",
            "problem_description_main",
            "problem_background_main",
            "problem_io",
            "required_dependencies",
            "problem_answer_main",
            "sub_steps",
            "general_solution",
            "general_tests",
            "oracle_dependencies",
            "oracle_solution",
            "oracle_tests",
        ]
    }
    difficulty_metrics = quality_difficulty_metrics(problem)
    user_prompt = textwrap.dedent(
        f"""
        Minimum acceptable grade: {min_grade}

        {SA_RUBRIC_TEXT}

        {QUALITY_FEWSHOT_SUMMARY}

        Hidden chain context, for reviewer use only:
        {json.dumps(chain_info, ensure_ascii=False, indent=2)}

        Derived implementation and test metrics, for reviewer use only:
        {json.dumps(difficulty_metrics, ensure_ascii=False, indent=2)}

        Problem JSON to review:
        {json.dumps(visible_problem, ensure_ascii=False, indent=2)}

        Grade the problem as S, A, B, C, or D.

        Apply these reviewer standards:
        - Inspect the prose, equations, IO, all sub-step code, final solution, and tests.
        - Do not grade as A/S merely because the code is executable or the story uses
          domain terms. A usable but thin or stitched-together task is B.
        - S/A requires a natural domain calculation, not a decorative story around code.
        - S/A requires a named scientific output with clear type/shape and domain meaning.
        - S/A requires meaningful parameters: no dummy inputs, unused inputs, or circular
          recovery of an already supplied answer.
        - S/A requires the final returned quantity to actually vary for at least one
          scientifically meaningful parameter change. Penalize diagnostics that are constant
          by construction, such as always-one determinants, always-normalized norms, or
          invariants used as the only deliverable.
        - S/A stochastic problems require explicit reproducibility: a seed/random_state
          input or explicit sample data. Penalize final functions that hard-code random
          seeds while the IO has no seed parameter.
        - S/A requires sub-steps that form a real dependency chain and feed the final answer.
        - S/A requires nontrivial numeric tests or scientific invariants, not just shape/type.
        - S/A requires enough difficulty for a SciCode-style benchmark item. Penalize tasks
          whose final solution is mostly a few vectorized lines, a direct closed-form formula,
          a wrapper around a common solver, or a simple filter/rank/count/statistic.
        - S/A requires student code length and structure to reflect real scientific
          implementation work. Use the derived metrics. As a rule of thumb, A/S should have
          at least {MIN_STUDENT_CODE_LINES} nonempty student-code lines and at least
          {MIN_STUDENT_FUNCTIONS} meaningful helper/final functions, unless the algorithmic
          difficulty is clearly high despite compact code.
        - Do not reward length inflation. Boilerplate, repeated validation branches, copied
          formulas, and unused helpers do not count as difficulty.
        - S/A requires SciCode-like test organization: sub-step tests should usually have
          about 3 executable snippets per step; final `general_tests` should usually have
          4-7 snippets, not one giant monolithic test block. Keep about 16 or more
          meaningful assert checks in general tests, plus oracle-vs-student differential
          checks on small deterministic cases.
        - S/A requires strong tests. `test_strength_score_1_to_10` must account for
          independent expected values, parameter sensitivity, edge cases, numerical
          tolerances, invariants, oracle-vs-student checks, and whether tests would catch
          plausible wrong implementations.
        - S/A oracle tests must be true differential tests: each oracle-test snippet should
          call the student final function and the oracle final function on the same case
          and directly compare the returned values or fields.
        - S/A requires `required_dependencies` to stay within base SciCode-style
          dependencies. Penalize any specialized domain package or direct one-call
          hidden-operator import in student code.
        - S/A requires student code with real implementation complexity; a few lines
          delegating to a library routine or hidden operator is B/C even if correct.
        - Penalize student code dominated by repetitive input-validation branches. Shape,
          finite-value, domain-range, and feasibility checks are useful, but many nearly
          identical `if ... raise ValueError` branches are not scientific complexity.
        - Grade as B, not A, when the task is mostly a wrapper around a standard
          matrix decomposition, filter, shift, rank, p-value, central-interval
          proportion, or "compute this statistic" workflow, even if the wrapper
          is domain themed.
        - Grade as B, not A, when the problem combines several unrelated numerical
          operations with a busy domain story but the scientific necessity of
          each operation is not compelling.
        - Grade as B, not A, when the final observable is too thin for the amount
          of setup, for example a single rank, yes/no diagnostic, central interval
          proportion, or a shifted image without a deeper scientific conclusion.
        - Grade as A only when the problem would still look like a natural domain
          computation if the hidden operator chain were unknown.
        - Grade as S only for polished tasks with a rich scientific output such as
          a spectrum, response curve, fitted domain parameter, validated state
          with diagnostics, energy/rate/probability map, or mode structure, plus
          strong tests.
        - It is fine to use normal domain words such as entropy, density matrix, Fourier,
          mutual information, wave, oscillator, spectrum, diffusion, or spin when natural.
          Do not penalize ordinary domain terminology. Penalize only direct hidden
          operator/function names or explicit instructions to call a specific hidden API.
        - Penalize overused generic two-qubit, spectroscopy, thermal-rod, or diffusion
          templates if the chain does not naturally support them.
        - Penalize final outputs that are bare constructors, boolean flags, identity
          matrices, basis vectors, or other intermediate helpers.
        - Penalize expensive or brittle tests that would likely timeout on small CPU runs.

        Return JSON exactly in this shape:
        {{
          "grade": "S/A/B/C/D",
          "accept": true,
          "score_1_to_10": 8,
          "problem_difficulty_score_1_to_10": 7,
          "student_code_complexity_score_1_to_10": 7,
          "student_code_length_adequacy_score_1_to_10": 7,
          "test_strength_score_1_to_10": 8,
          "strengths": ["short concrete strength"],
          "issues": ["concrete issue if any"],
          "required_fixes": ["specific fix needed to reach {min_grade}, empty if accepted"],
          "evidence": [
            {{"field": "problem_io/general_solution/general_tests/sub_steps/difficulty_metrics", "finding": "specific evidence"}}
          ],
          "reason": "2-4 sentence judgment"
        }}

        For grade A or S with accept=true, all of these must be true:
        - problem_difficulty_score_1_to_10 >= {QUALITY_DIMENSION_MIN_SCORES["problem_difficulty_score_1_to_10"]}
        - student_code_complexity_score_1_to_10 >= {QUALITY_DIMENSION_MIN_SCORES["student_code_complexity_score_1_to_10"]}
        - student_code_length_adequacy_score_1_to_10 >= {QUALITY_DIMENSION_MIN_SCORES["student_code_length_adequacy_score_1_to_10"]}
        - test_strength_score_1_to_10 >= {QUALITY_DIMENSION_MIN_SCORES["test_strength_score_1_to_10"]}
        If any dimension is below threshold, set accept=false and list concrete fixes.
        """
    ).strip()
    return system_prompt, user_prompt


def build_quality_repair_prompt(
    *,
    chain: dict[str, Any],
    selected_background: dict[str, Any] | None,
    problem: dict[str, Any],
    review: dict[str, Any],
    semantic_review: dict[str, Any],
    min_grade: str,
) -> tuple[str, str]:
    system_prompt = textwrap.dedent(
        """
        You repair a SciCode-style scientific problem so it reaches the requested quality grade.
        Return one valid JSON object only.
        """
    ).strip()
    chain_info = chain_context_for_prompt(chain)
    chain_info["selected_domain_background"] = selected_background or {}
    user_prompt = textwrap.dedent(
        f"""
        Target minimum grade: {min_grade}

        {SA_RUBRIC_TEXT}

        Hidden chain context:
        {json.dumps(chain_info, ensure_ascii=False, indent=2)}

        Reviewer feedback:
        {json.dumps(review, ensure_ascii=False, indent=2)}

        Semantic consistency feedback:
        {json.dumps(semantic_review, ensure_ascii=False, indent=2)}

        Previous problem JSON:
        {json.dumps(problem, ensure_ascii=False, indent=2)}

        Repair task:
        Return a complete corrected problem JSON with the same required schema.
        Preserve the selected scientific background if it is good, but change the
        scenario if the reviewer says it is forced or too template-like.

        Mandatory repair rules:
        - Preserve the required JSON schema exactly. `problem_io`,
          `required_dependencies`, `problem_answer_main`, `general_solution`,
          `oracle_dependencies`, and `oracle_solution` must remain JSON strings.
          `sub_steps`, `general_tests`, and `oracle_tests` must remain arrays.
        - `problem_io` must keep literal `Parameters:` and `Returns:` sections with
          named, used inputs and a named returned scientific quantity.
        - `required_dependencies` must be executable import statements only, one per
          line, and should include `import numpy as np` unless the task is purely scalar.
        - Do not add import statements to `general_solution`, sub-step code, tests, or
          oracle code; imports belong only in dependency fields.
        - Keep multi-line tests executable after JSON parsing; do not leave literal
          `\\n` backslash-n sequences in snippets.
        - Never use an undefined `target` fixture. Tests must call the actual student
          final function and, for oracle tests, `oracle_<student_final_function>`.
        - Strengthen the domain statement, equations/definitions, IO, main answer, sub-step
          dependency chain, and tests until the problem is plausibly {min_grade}+.
        - Raise the actual benchmark difficulty when reviewer feedback indicates a thin
          task. Add a scientifically necessary intermediate computation, inverse step,
          calibration/fitting step, stability or uncertainty diagnostic, coupled quantity,
          or parameter-sensitivity analysis. Do not merely add prose.
        - Keep the final output a meaningful scientific observable or fitted/domain
          quantity, not a constructor, boolean flag, shape, or helper artifact.
        - Make every parameter used by the computation and at least one test sensitive
          to changing a parameter value.
        - If the previous final output is constant by construction, change the final
          deliverable to a response curve, spectrum, rate, energy, fitted parameter, or
          other quantity that genuinely changes with the stated inputs.
        - Keep the oracle/student split: `general_solution` is the nontrivial student
          solution under base dependencies; `oracle_solution` is the hidden reference
          used for differential checks.
        - Make the student solution substantively complex enough for SciCode-style data:
          normally at least {MIN_STUDENT_CODE_LINES} nonempty lines and
          {MIN_STUDENT_FUNCTIONS} meaningful functions, dominated by algorithms,
          formulas, loops, linear algebra, numerical integration, optimization,
          interpolation, simulation, or statistical estimation. Do not inflate length
          with validation boilerplate or unused helpers.
        - Keep input validation concise and meaningful. Preserve checks for finite values,
          shape compatibility, positive domain scales, valid probability/range constraints,
          and feasibility. Remove repetitive or decorative `if ... raise` branches that do
          not affect scientific correctness.
        - Replace shape/type-only tests with small numeric checks, independently derived
          expected values, or scientific invariants.
        - Make tests more SciCode-like: each sub-step should normally have 3 test snippets;
          final `general_tests` should have 4-7 snippets with at least
          {MIN_GENERAL_ASSERTIONS} total asserts. Oracle tests must have at least
          {MIN_ORACLE_ASSERTIONS} asserts and compare student and oracle outputs directly.
          Add tests that would catch plausible wrong algorithms: parameter sensitivity,
          edge cases, independent closed-form values, monotonicity/conservation/bounds,
          and oracle-vs-student comparisons on small deterministic cases.
        - Remove any specialized domain package or direct high-level one-call operator
          import from `required_dependencies`.
        - If semantic feedback says an input is unused or decorative, either make it
          affect the returned observable and add a sensitivity test, or remove it from
          `problem_io`.
        - If semantic feedback says a sub-step does not feed downstream, rewrite it so
          its returned quantity is used by a later step or by the final function.
        - If any validation error contains `timeout`, keep the scientific task and only shrink
          the slow sub-step implementation/test: smaller arrays, fewer iterations,
          lower symbolic degree, and deterministic closed-form checks.
        - Keep all student code executable with only NumPy/SciPy/math-style base
          dependencies visible to students.
        - Do not expose hidden operator ids, hidden operator names, "operator chain",
          QuTiP, Cirq, Qiskit, or exact hidden API names.
        - Do not write "Write a Python function"; present the task as a scientific
          computation, with the function interface only in `problem_io`.
        - Output JSON only; no markdown fences.
        """
    ).strip()
    return system_prompt, user_prompt


def build_generation_prompt(
    chain: dict[str, Any],
    selected_background: dict[str, Any] | None,
) -> tuple[str, str]:
    system_prompt = textwrap.dedent(
        """
        You create SciCode-style scientific coding problems. Return one valid JSON object only.
        The problem must be a domain-facing numerical/scientific coding problem, not a
        request to implement a named library operator. The per-subdomain operator graph
        walk is hidden inspiration only.
        """
    ).strip()

    chain_info = chain_context_for_prompt(chain)
    chain_info["selected_domain_background"] = selected_background or {}
    background_instruction = ""
    if selected_background:
        background_instruction = textwrap.dedent(
            """
            Use `selected_domain_background` as the student-facing scientific setting.
            Preserve its concrete observable and domain story, but do not expose hidden
            operator ids or say that the task is based on an operator chain. Translate the
            hidden operator roles into ordinary scientific steps.
            """
        ).strip()
    user_prompt = textwrap.dedent(
        f"""
        Generate one SciCode-style problem from this hidden chain context:
        {json.dumps(chain_info, ensure_ascii=False, indent=2)}

        Background instruction:
        {background_instruction or "Choose a natural domain-facing scientific setting from the hidden context."}

        Variant instruction:
        {chain.get("variant_instruction") or "No special variant instruction."}

        Quality target:
        {SA_RUBRIC_TEXT}

        {QUALITY_FEWSHOT_SUMMARY}

        Internal generation order for this single response:
        A. First design the student-facing scientific task from the hidden chain:
           choose the domain setting, inputs, final output, and sub-step decomposition.
        B. Then write the student implementation and the oracle implementation for that
           exact task. The student implementation must use only the base dependencies
           in `required_dependencies`; the oracle is only for hidden differential checks.
        C. Then write broad general tests and oracle-vs-student tests that exercise the
           implemented behavior, edge cases, invariants, and parameter sensitivity.
        D. Finally revise the problem description, background, IO, sub-step prose, and
           answer so the prose matches the already written code and tests. Do this
           polishing inside the JSON you return; do not expose these internal steps.

        Hard requirements:
        1. The main problem must be a concrete SciCode-style computation in the chain's
           own subdomain. It may be physics, mathematics, chemistry, biology, finance,
           astronomy, geometry, materials, statistics, or another represented field.
           The final output must be a meaningful domain quantity, not a thin wrapper
           around the hidden API.
        2. The problem is NOT "implement this function/operator". Do not use the phrase
           "Write a Python function" in `problem_description_main`. Frame the main
           statement as a scientific calculation with equations, inputs, and named outputs;
           put the concrete function interface in `problem_io` and sub-step headers. Avoid
           "write a function" phrasing in every student-visible field, including sub-steps.
        3. Do not mention the hidden operator ids, "operator chain", QuTiP, Cirq, Qiskit,
           or direct library/operator-function names in the visible problem text, problem
           name, subproblem text, helper names, comments, or code. Do not use any exact
           string listed in `hidden_terms_to_avoid_in_student_problem`, including underscore,
           space, and hyphen variants. Domain concepts such as density matrices, spin systems, mutual
           information, wave packets, Bessel functions, or Fourier analysis are allowed when
           they are the natural scientific background; just do not tell students which hidden
           operator to call. Also avoid every item in
           `selected_domain_background.student_visible_terms_to_avoid`; use descriptive
           scientific helper names instead of hidden operator/function names.
           If a hidden operator comes from a quantum SDK or a specialized synthesis package,
           do not turn the task into named circuit/operator synthesis. Use a low-dimensional
           small domain calculation that can be solved with NumPy/SciPy/math-style base tools instead.
        4. {ALLOWED_IMPORT_HINT}
           Natural-language problem fields and sub-step prompts must describe the
           mathematical or domain operation, not the exact library API to call. Do not
           write instructions such as "Use scipy.linalg.expm", "compute with np.kron",
           or "apply sympy.simplify" in visible text. Exact imports and library calls are
           allowed inside code only when they are base-library building blocks, not a
           direct call to the hidden complex operator. `required_dependencies` should
           import modules, not individual high-level functions.
        5. `problem_io` must list at least one meaningful student input parameter. Do not
           create an all-constants task whose Parameters section is "None" or says all
           parameters are fixed. `problem_io` and `problem_answer_main` must explicitly name the output quantity
           and its type/shape. Do not write only "result". `problem_answer_main` must be
           3-5 substantive sentences and at least about 300 characters: state the returned
           quantity, summarize how it is computed from sub-step outputs, give its physical
           interpretation and units/shape, and mention one sanity-check interpretation.
           The final output must be a scientific observable, statistic, response curve,
           energy/rate/probability, spectrum, or fitted parameter. Do not make the final
           deliverable a bare identity matrix, basis/Fock state vector, tensor product,
           or boolean normalization flag; those may appear only as intermediate helpers.
           If a boolean diagnostic is scientifically useful, return it only alongside a
           numerical diagnostic curve/value; never make a lone boolean the final output.
           Every input parameter in `problem_io` must affect the final returned quantity
           or be explicitly removed. Add at least one general test that changes one
           parameter while holding the others fixed and verifies a scientifically reasonable
           change in the final output. Do not choose a final diagnostic that is constant
           by construction, such as a determinant that is always 1, a norm that is always
           normalized, or a conservation flag that never changes; such quantities may only
           be secondary sanity checks.
           If the problem uses randomness or simulated samples, include either the sample
           data itself or a `seed`/`random_state` input in `problem_io`; do not hard-code a
           random seed inside the final student function.
        6. `required_dependencies` constrains the student solution only. It must contain
           executable imports from the allowed base libraries and must not contain
           specialized domain packages or direct one-call hidden-operator imports.
        7. `general_solution` is the student solution. It must be complete executable
           Python code with enough real implementation detail: multiple helper functions,
           explicit formulas/loops/linear algebra/numerical routines, and not just a few
           lines delegating to one high-level solver. It must not import packages itself;
           put all imports in `required_dependencies`.
           Input validation is useful only when it protects a scientifically meaningful
           domain assumption: finite arrays, matching shapes, positive domain scales,
           ordered time grids, valid probabilities, or feasible bounds. Do not fill the
           student solution with repetitive `if ... raise ValueError` guards for every
           minor condition. Prefer compact validation helpers, and keep the main code
           dominated by the scientific computation rather than exception branches.
        8. `oracle_dependencies`, `oracle_solution`, and `oracle_tests` are hidden
           reference artifacts used for fast differential checking. The oracle may be a
           compact independent implementation, but it should still be deterministic and
           CPU-small. Name the oracle final function `oracle_<student_final_function>`.
           `oracle_tests` must compare the student final function against the oracle
           across several small cases. `oracle_tests` must contain at least 4 separate
           list entries. Every oracle-test entry must call both final functions on the
           same inputs, store or compare the two results, and contain at least 2 assert
           statements. At least one assertion in each entry must be a direct differential
           check such as `np.allclose(student_value, oracle_value)`, `np.isclose(...)`,
           a recursive comparison of dict/list fields, or equivalent numeric comparison.
        9. `general_tests` must be a SciCode-like list of self-contained Python assertion
           snippets that run after `required_dependencies` and `general_solution`. Do not
           use an undefined variable named `target`. Prefer 4-7 final-function test snippets,
           matching SciCode's `general_tests` scale, but keep them strong: at least sixteen
           total assert statements. At least four snippets must check nontrivial numerical
           values, algorithmic properties, or scientific invariants with `np.isclose`,
           `np.allclose`, `abs(...) < tolerance`, conservation/normalization, monotonicity,
           positivity, or boundedness; do not rely only on shape/type checks. Include at
           least one edge-case or parameter-sensitivity test.
           Tests must not contain guessed constants. Derive each expected numerical value
          from a small hand-checkable formula in the snippet, or assert a domain invariant
           that follows from the equations.
           Use deterministic small inputs. If a statistical test has a minimum sample-size
           requirement, either use enough samples or avoid that test. If a graph or sparse
           matrix uses missing links, define the convention explicitly and make the code
           convert it consistently before numerical checks.
           Keep sub-step tests deliberately small: small matrices, low polynomial degree,
           few grid points, few iterations, no broad root scans, no large Monte Carlo, and
           no symbolic expressions that grow combinatorially.
           Every test input must obey the parameter domain stated in the main problem and
           `problem_io`; if you want to test periodic wrapping or clipping, state that
           behavior explicitly in the input contract first.
        10. Each sub-step must be dependent on earlier steps and include:
           `step_number`, `step_description_prompt`, `step_background`,
           `ground_truth_code`, `function_header`, `test_cases`, and `return_line`.
           Sub-step test cases must be self-contained assertions, also without `target`.
           Use the SciCode-style distribution: normally exactly 3 test snippets per
           sub-step, and never fewer than 3 or more than 7 unless the step is unusually
           complex. Prefer numeric expected values or invariants over shape-only checks.
           Every variable used in an assertion must be assigned inside that same snippet.
           Provide at least three sub-steps. No sub-step may be a placeholder or an
           unused imitation of a hidden operator; every sub-step output must feed into
          a later sub-step or the final returned scientific quantity.
        11. Prefer SciCode-like domains and decomposition: formula first, numerical helper
           steps next, final scientific output last. Keep array sizes small in tests.
        12. Keep the chosen background coherent across the main problem, sub-steps, code,
            tests, and final answer.
        13. Choose problem names and helper names that describe domain quantities, not
            hidden operator names or generic templates. Avoid repeated prefixes such as
            `two_qubit_*`, `spectroscopy_peak_*`, `thermal_diffusion_*`, or `laser_beam_*`
            unless the chain truly requires that exact scenario.
        14. Output JSON only; no markdown fences.
        15. If this is a repeated variant of a high-potential chain, do not merely rename a
            previous-style task. Change at least two of: scientific background, input
            quantities, observable, decomposition, or diagnostic tests.
        16. After writing code and tests, revise the problem prose in this same response
            so the final description, background, IO, and answer accurately reflect the
            implemented code and tests. The problem should read like a meaningful
            SciCode task, not like a post-hoc wrapper around operators.

        Schema and code-field invariants:
        - `problem_io`, `required_dependencies`, `problem_answer_main`,
          `general_solution`, `oracle_dependencies`, and `oracle_solution` must be
          JSON strings, never objects, lists, null, or markdown blocks.
        - `problem_io` must be a docstring-like string with literal `Parameters:` and
          `Returns:` sections. Use named parameters, types/shapes, valid ranges, and a
          named returned quantity. Never write `None` for parameters.
        - `required_dependencies` must be only executable import statements, one per
          line. It must include `import numpy as np` unless the whole problem is purely
          scalar math, and must not include explanations, bullets, comments, code fences,
          assignments, or prose.
        - `general_solution`, every sub-step `ground_truth_code`, `general_tests`,
          `oracle_solution`, and `oracle_tests` must not contain import statements.
          All imports belong only in `required_dependencies` or `oracle_dependencies`.
        - Never use an undefined variable or fixture named `target`. Tests must call the
          actual student final function by name, and oracle tests must call both the
          actual student final function and `oracle_<student_final_function>`.
        - Multi-line test snippets must execute as Python after JSON parsing. Do not
          put literal backslash-n text such as `try:\\n    ...` inside a snippet; use
          real newline characters in the JSON string so the parsed snippet contains
          valid Python lines.
        - Every sub-step helper returned value must be consumed by a later helper or by
          the final student function. Do not include decorative helper functions that
          are tested but unused.

        Required JSON schema:
        {{
          "problem_name": "snake_case_name",
          "problem_id": "{chain.get("scicode_seed_id") or "SCICODE-SEED"}",
          "problem_description_main": "Main scientific/domain problem with equations or definitions and deliverable.",
          "problem_background_main": "Scientific background, concise but useful.",
          "problem_io": "\\\"\\\"\\\"\\nParameters:\\n    ...\\nReturns:\\n    output_name (type/shape): scientific meaning.\\n\\\"\\\"\\\"",
          "required_dependencies": "import numpy as np\\nfrom scipy import ...",
          "problem_answer_main": "The final function returns output_name (...): ...",
          "sub_steps": [
            {{
              "step_number": "{chain.get("scicode_seed_id") or "SCICODE-SEED"}.1",
              "step_description_prompt": "...",
              "step_background": "...",
              "ground_truth_code": "def helper(...):\\n    ...",
              "function_header": "def helper(...):\\n    '''...'''",
              "test_cases": ["assert ..."],
              "return_line": "    return ..."
            }}
          ],
          "general_solution": "def helper(...):\\n    ...\\n\\ndef final_function(...):\\n    ...",
          "general_tests": ["assert ...", "assert ..."],
          "oracle_dependencies": "import numpy as np\\n...",
          "oracle_solution": "def oracle_final_function(...):\\n    ...",
          "oracle_tests": [
            "student = final_function(case1); reference = oracle_final_function(case1); assert ...; assert ...",
            "student = final_function(case2); reference = oracle_final_function(case2); assert ...; assert ..."
          ]
        }}
        """
    ).strip()
    return system_prompt, user_prompt


def build_final_polish_prompt(
    *,
    chain: dict[str, Any],
    selected_background: dict[str, Any] | None,
    problem: dict[str, Any],
) -> tuple[str, str]:
    system_prompt = textwrap.dedent(
        """
        You polish a validated SciCode-style problem after its student code,
        oracle code, and tests already execute. Return one valid JSON object only.
        """
    ).strip()
    chain_info = chain_context_for_prompt(chain)
    chain_info["selected_domain_background"] = selected_background or {}
    user_prompt = textwrap.dedent(
        f"""
        Hidden chain context:
        {json.dumps(chain_info, ensure_ascii=False, indent=2)}

        Validated problem JSON:
        {json.dumps(problem, ensure_ascii=False, indent=2)}

        Polish task:
        Return the complete problem JSON with the same required schema. Use the
        existing `general_solution`, `general_tests`, `oracle_solution`, and
        `oracle_tests` as grounding evidence. Improve the student-facing description,
        background, IO contract, sub-step prose, and final answer so they describe
        exactly what the code and tests compute.

        Strict constraints:
        - Preserve all function names, function signatures, return types, code behavior,
          and test intent unless a tiny wording/code consistency fix is unavoidable.
        - Do not remove or weaken tests. Keep SciCode-like test structure: roughly 3 tests
          per sub-step, 4-7 final `general_tests` snippets, at least
          {MIN_GENERAL_ASSERTIONS} general assert statements, and at least
          {MIN_ORACLE_ASSERTIONS} oracle assert statements.
        - Keep `required_dependencies` limited to the allowed base libraries.
        - Keep the student solution nontrivial and decomposed; do not replace it with a
          one-call hidden operator or specialized library wrapper.
        - Keep validation branches concise and scientifically meaningful. The polish should
          not turn the student code into mostly input-error handling.
        - Do not expose hidden operator ids, hidden API symbols, specialized package
          names, or the phrase "operator chain".
        - Do not introduce visible instructions naming exact base-library calls such as
          `np.gradient`, `np.linalg.solve`, `scipy.optimize...`, or `sympy...`; visible
          prose should describe the mathematical operation instead.
        - Make the problem read as a coherent SciCode task in the stated subdomain,
          with a meaningful domain output and clear parameter influence.
        - Output JSON only; no markdown fences.
        """
    ).strip()
    return system_prompt, user_prompt


def build_repair_prompt(
    chain: dict[str, Any],
    selected_background: dict[str, Any] | None,
    previous_problem: dict[str, Any] | None,
    errors: list[str],
) -> tuple[str, str]:
    system_prompt = textwrap.dedent(
        """
        You repair SciCode-style scientific coding problem JSON. Return one valid JSON object only.
        Preserve the scientific intent, but fix every validation error.
        """
    ).strip()
    user_prompt = textwrap.dedent(
        f"""
        Hidden chain context:
        {json.dumps({**chain_context_for_prompt(chain), "selected_domain_background": selected_background or {}}, ensure_ascii=False, indent=2)}

        Validation errors to fix:
        {json.dumps(errors, ensure_ascii=False, indent=2)}

        Previous JSON:
        {json.dumps(previous_problem or {}, ensure_ascii=False, indent=2)}

        Quality target:
        {SA_RUBRIC_TEXT}

        {QUALITY_FEWSHOT_SUMMARY}

        Repair constraints:
        - Output JSON only.
        - Return the complete top-level schema every time. Include all fields:
          `problem_name`, `problem_id`, `problem_description_main`,
          `problem_background_main`, `problem_io`, `required_dependencies`,
          `problem_answer_main`, `sub_steps`, `general_solution`,
          `general_tests`, `oracle_dependencies`, `oracle_solution`, and
          `oracle_tests`.
        - The fields `problem_io`, `required_dependencies`, `problem_answer_main`,
          `general_solution`, `oracle_dependencies`, and `oracle_solution` must be
          JSON strings, not dictionaries, arrays, nulls, markdown fences, or bullets.
        - `problem_io` must be a docstring-like string with literal `Parameters:` and
          `Returns:` sections. It must name at least one meaningful input parameter,
          with type/shape/range details, and must name the returned scientific output.
        - `required_dependencies` must contain executable import statements only, one
          import per line. Include `import numpy as np` unless the task is purely scalar.
          Do not put prose, comments, assignments, bullets, or code fences in this field.
        - Keep the task domain-facing, scientifically meaningful, and SciCode-like.
        - Keep using the selected scientific background if one is provided.
        - Student-visible required dependencies may include only allowed base libraries:
          numpy, scipy modules, math/cmath/itertools/collections/functools/copy/random,
          and sympy. Do not import specialized domain libraries. Do not use direct
          high-level function imports that solve a hidden complex operator in one call.
        - Do not mention hidden operator ids, hidden function names, specialized domain packages,
          or any exact term listed in `hidden_terms_to_avoid_in_student_problem`.
          This applies to `problem_name`, prose, sub-step text, helper names, comments, and tests.
        - In natural-language visible fields, describe the mathematical operation instead of
          naming exact library APIs like `scipy.linalg.expm`, `np.kron`, or `sympy.simplify`.
          Library calls are allowed in imports, solution code, and test snippets.
        - If an error says `visible_text_direct_library_api_instruction`, remove that visible
          library/API reference and replace it with domain language such as "matrix
          exponential", "Kronecker product", "nonlinear root solve", or "orthogonal
          polynomial evaluation".
        - If an error says `problem_has_no_meaningful_student_inputs`, add one or more
          scientifically meaningful input parameters in `problem_io`, use them in the
          solution, and test at least two different values.
        - If an error says `leaked_operator_name`, remove that exact hidden function/operator
          name from all prose, helper names, code, comments, and tests. Replace it with a
          domain quantity name such as `spectrum`, `mode_profile`, `calibration_curve`,
          `response_matrix`, or another domain-specific observable.
        - Remove any named SDK algorithm, circuit-synthesis algorithm, placeholder step,
          or wording like "student must implement" / "you may return". Every sub-step
          must compute a real intermediate quantity that is used downstream.
        - Avoid "write a function" wording in all visible problem and sub-step text; frame
          each step as computing an intermediate domain quantity.
        - Ensure `problem_io` lists at least one meaningful input parameter, and ensure
          `problem_io` and `problem_answer_main` name the output quantity.
        - The final output must be a scientific observable, statistic, response curve,
          energy/rate/probability, spectrum, or fitted parameter. Do not make the final
          deliverable a bare identity matrix, basis/Fock state vector, tensor product,
          or boolean normalization flag; those may appear only as intermediate helpers.
        - If random generation is used, add a `seed` or explicit input data parameter and
          remove hard-coded seeds from the final function. Tests may set deterministic seeds.
        - Expand `problem_answer_main` to 3-5 substantive sentences, at least about
          300 characters, with output quantity, computation path, domain meaning,
          units/shape, and one sanity-check interpretation.
        - Keep `general_solution` as the student solution. It must be nontrivial,
          decomposed, and at least several dozen lines of real implementation, not a
          short wrapper around one high-level library operator.
        - Keep validation meaningful but not dominant. Use a compact validation helper
          for finite/shape/range/feasibility checks and remove repetitive `if ... raise`
          branches that merely restate obvious type assumptions.
        - Remove all import statements from `general_solution`, every sub-step
          `ground_truth_code`, `general_tests`, `oracle_solution`, and `oracle_tests`.
          Put imports only in `required_dependencies` or `oracle_dependencies`.
        - Multi-line tests must be valid Python snippets after JSON parsing. Do not
          double-escape newlines into literal `\\n` text inside snippets; use actual
          newline characters in the JSON string.
        - Do not call any exact hidden API symbol from the student solution or sub-step
          ground-truth code. Rebuild the needed behavior from lower-level base operations.
        - Add or repair `oracle_dependencies`, `oracle_solution`, and `oracle_tests`.
          Name the oracle final function `oracle_<student_final_function>`, and make
          `oracle_tests` directly call both the student final function and the oracle
          final function on small deterministic cases. Every oracle-test snippet should
          contain a direct student-vs-oracle numerical comparison.
        - Strengthen `general_tests` in a SciCode-like shape: 4-7 executable assertion
          snippets, containing at least sixteen total `assert` statements and at least
          four nontrivial numerical/invariant checks.
        - Strengthen sub-step `test_cases`: normally exactly three executable assertion
          snippets per sub-step, never fewer than three.
        - Strengthen `oracle_tests`: at least four executable snippets and at least
          eight total `assert` statements comparing student and oracle behavior.
        - Repair failing tests by making the code and expected values follow the same
          equations. Do not guess magic expected constants. For each assertion, either
          compute the expected value from a short independent formula or test a physical
          invariant such as normalization, conservation, positivity, boundedness, symmetry,
          or monotonic/parameter sensitivity.
        - Use deterministic test inputs. Avoid small-sample statistical tests unless the
          sample is large enough for that test. For graph/sparse-matrix problems, define
          missing-edge conventions explicitly and convert them consistently in code and tests.
        - Keep all tests inside the parameter domains stated in the problem and IO. If a
          test uses a wrapped, clipped, or otherwise out-of-range value, revise the stated
          input contract to make that behavior explicit.
        - Make every test snippet self-contained: every variable referenced by an assertion
          must be defined in that snippet.
        - Each `sub_steps` item must be an object with `step_number`,
          `step_description_prompt`, `step_background`, `ground_truth_code`,
          `function_header`, `test_cases`, and `return_line`.
        - Ensure all code and all tests execute without an undefined `target`.
          Never use `target` as a testing fixture or alias. Use the actual student
          final function name.
        - If an error says a sub-step helper is not used downstream, rewrite the
          later helper/final function so that exact helper output becomes an input to
          a later computation. Do not merely rename the helper or remove its tests.
        """
    ).strip()
    return system_prompt, user_prompt
