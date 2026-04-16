from __future__ import annotations

from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall


tool_metric = ToolCorrectnessMetric(
    threshold=0.5,
    model="gpt-5.4-mini",
    include_reason=True,
)



def test_planner_tools():
    test_case = LLMTestCase(
        input="Compare naive RAG vs sentence-window retrieval",
        actual_output="Generated a focused research plan.",
        tools_called=[ToolCall(name="knowledge_search"), ToolCall(name="web_search")],
        expected_tools=[ToolCall(name="knowledge_search"), ToolCall(name="web_search")],
    )
    assert_test(test_case, [tool_metric])



def test_researcher_tools():
    test_case = LLMTestCase(
        input="Research sentence-window retrieval with web and local docs",
        actual_output="Produced findings memo grounded in MCP search tools.",
        tools_called=[
            ToolCall(name="knowledge_search"),
            ToolCall(name="web_search"),
            ToolCall(name="read_url"),
        ],
        expected_tools=[
            ToolCall(name="knowledge_search"),
            ToolCall(name="web_search"),
            ToolCall(name="read_url"),
        ],
    )
    assert_test(test_case, [tool_metric])



def test_supervisor_save():
    test_case = LLMTestCase(
        input="Supervisor receives APPROVE from Critic",
        actual_output="Report saved to output/rag_comparison.md",
        tools_called=[ToolCall(name="save_report")],
        expected_tools=[ToolCall(name="save_report")],
    )
    assert_test(test_case, [tool_metric])
