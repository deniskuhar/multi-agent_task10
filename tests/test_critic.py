from __future__ import annotations

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from tests._helpers import run_critic


critique_quality = GEval(
    name="Critique Quality",
    evaluation_steps=[
        "Check that the critique identifies specific issues, not vague complaints.",
        "Check that revision requests are actionable and concrete.",
        "If verdict is APPROVE, gaps should be empty or only minor.",
        "If verdict is REVISE, there must be at least one actionable revision request.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-5.4-mini",
    threshold=0.6,
)


business_logic_metric = GEval(
    name="Critique Scope Control",
    evaluation_steps=[
        "Check that the critique evaluates the findings against the original request rather than inventing a new scope.",
        "Penalize critiques that ask for unrelated additions.",
        "Reward critiques that focus on freshness, completeness, and structure in a balanced way.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-5.4-mini",
    threshold=0.7,
)



def test_critique_revise_quality():
    findings = """
    Brief Summary: Naive RAG is simple. Sentence-window adds more local context.
    Key Findings: Parent-child retrieval often preserves broader structure.
    Sources: generic blog posts only.
    """
    _, actual_output = run_critic(findings)
    test_case = LLMTestCase(input=findings, actual_output=actual_output)
    assert_test(test_case, [critique_quality, business_logic_metric])
