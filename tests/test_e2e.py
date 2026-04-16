from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from tests._helpers import load_dataset, run_supervisor_pipeline


goldens = load_dataset()

answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-5.4-mini",
    include_reason=True,
)

correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether the facts in actual output contradict expected output.",
        "Penalize omission of critical details.",
        "Different wording of the same concept is acceptable.",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model="gpt-5.4-mini",
    threshold=0.6,
)

citation_presence = GEval(
    name="Citation Presence",
    evaluation_steps=[
        "Check whether the output gives at least some source grounding, references, or explicit evidence markers when the task is a research task.",
        "Reward outputs that name sources, documents, or evidence.",
        "Do not require formal citation style, but penalize unsupported assertive reports.",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    model="gpt-5.4-mini",
    threshold=0.5,
)


@pytest.mark.parametrize("golden", goldens)
def test_golden_dataset(golden: dict):
    _, actual_output = run_supervisor_pipeline(golden["input"])
    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=actual_output,
        expected_output=golden["expected_output"],
    )
    assert_test(test_case, [answer_relevancy, correctness, citation_presence])
