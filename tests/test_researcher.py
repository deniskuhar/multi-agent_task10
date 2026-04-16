from __future__ import annotations

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from tests._helpers import get_retrieval_context, run_research


groundedness = GEval(
    name="Groundedness",
    evaluation_steps=[
        "Extract factual claims from actual output.",
        "Check whether each factual claim is directly supported by the retrieval context.",
        "Claims not supported by the retrieval context count as ungrounded even if they may be true in general.",
        "Reward outputs that clearly acknowledge uncertainty when evidence is weak.",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model="gpt-5.4-mini",
    threshold=0.7,
)


source_usefulness = GEval(
    name="Source Usefulness",
    evaluation_steps=[
        "Check whether the answer actually uses the retrieval context instead of ignoring it.",
        "Reward outputs that synthesize evidence into useful findings.",
        "Penalize unsupported generic filler.",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model="gpt-5.4-mini",
    threshold=0.7,
)



def test_research_grounded_happy_path():
    user_input = "Explain RAG for document question answering using retrieved evidence."
    actual_output = run_research(user_input)
    retrieval_context = get_retrieval_context(user_input)

    test_case = LLMTestCase(
        input=user_input,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )
    assert_test(test_case, [groundedness, source_usefulness])
