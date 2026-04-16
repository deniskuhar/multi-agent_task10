from __future__ import annotations

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from tests._helpers import run_planner


plan_quality = GEval(
    name="Plan Quality",
    evaluation_steps=[
        "Check that the plan contains specific search queries and not only vague phrases.",
        "Check that sources_to_check includes relevant sources for the topic.",
        "Check that the output_format matches what the user asked for.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-5.4-mini",
    threshold=0.6,
)


plan_scope_metric = GEval(
    name="Plan Scope Discipline",
    evaluation_steps=[
        "Check that the plan stays focused on the user's request.",
        "Penalize plans that expand scope into unrelated topics.",
        "Reward plans that are actionable for a researcher agent.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-5.4-mini",
    threshold=0.7,
)



def test_plan_quality_happy_path():
    user_input = "Compare naive RAG vs sentence-window retrieval for document QA."
    _, actual_output = run_planner(user_input)
    test_case = LLMTestCase(input=user_input, actual_output=actual_output)
    assert_test(test_case, [plan_quality, plan_scope_metric])
