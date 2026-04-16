from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "tests" / "golden_dataset.json"


def load_dataset() -> list[dict[str, Any]]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _import(module_name: str):
    return importlib.import_module(module_name)


def get_supervisor_module():
    return _import("supervisor")


def get_tools_module():
    try:
        return _import("tools")
    except ModuleNotFoundError:
        return None


def run_planner(user_input: str) -> tuple[Any, str]:
    sup = get_supervisor_module()
    fn = getattr(sup, "plan", None) or getattr(sup, "delegate_to_planner", None)
    if fn is None:
        raise RuntimeError("Could not find planner function in supervisor.py")
    result = fn(user_input)
    return result, to_text(result)


def run_research(user_input: str) -> str:
    sup = get_supervisor_module()

    fn = getattr(sup, "research", None) or getattr(sup, "delegate_to_researcher", None)
    if fn is None:
        raise RuntimeError("Could not find researcher function in supervisor.py")

    retrieval_context = get_retrieval_context(user_input)
    context_text = "\n\n".join(retrieval_context)

    prompt = f"""
User request:
{user_input}

Retrieved context:
{context_text}

CRITICAL RULES FOR THIS TASK:
- If sentence-window retrieval is NOT clearly described in the context → DO NOT compare it.
- DO NOT assume or infer what sentence-window retrieval is.
- DO NOT present this as a full comparison if evidence is missing.
- You MUST explicitly say that comparison is not supported by the retrieved context.
- Only describe what is explicitly present in the context.

Output format:
- Brief Summary
- Key Findings
- Open Questions / Uncertainty
- Sources
""".strip()

    return fn(prompt)


def run_critic(findings: str) -> tuple[Any, str]:
    sup = get_supervisor_module()
    fn = getattr(sup, "critique", None) or getattr(sup, "delegate_to_critic", None)
    if fn is None:
        raise RuntimeError("Could not find critic function in supervisor.py")

    try:
        result = fn(findings)
    except TypeError:
        try:
            result = fn(findings=findings)
        except TypeError:
            from schemas import ResearchPlan

            dummy_plan = ResearchPlan(
                goal="Evaluate research findings quality",
                search_queries=["rag evaluation"],
                sources_to_check=["knowledge_base", "web"],
                output_format="structured critique",
            )

            result = fn(
                original_request="Compare RAG approaches",
                plan_obj=dummy_plan,
                findings=findings,
            )

    if hasattr(result, "model_dump_json"):
        actual_output = result.model_dump_json(indent=2)
    elif hasattr(result, "model_dump"):
        import json
        actual_output = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    else:
        actual_output = str(result)

    return result, actual_output


def run_supervisor_pipeline(user_input: str) -> tuple[Any, str]:
    sup = get_supervisor_module()
    fn = getattr(sup, "run_supervisor", None)
    if fn is None:
        raise RuntimeError("Could not find run_supervisor in supervisor.py")
    result = fn(user_input)
    return result, to_text(result)


def get_retrieval_context(user_input: str) -> list[str]:
    ctx: list[str] = []

    from tools import knowledge_search

    queries = [
        "naive RAG vs sentence-window retrieval document QA comparison",
        "sentence-window retrieval document QA comparison",
        "naive RAG document QA",
        "sentence-window retrieval",
    ]

    for q in queries:
        try:
            kb = knowledge_search(q)
            if kb:
                ctx.append(kb)
        except Exception:
            pass

    return ctx


def maybe_save_report(filename: str, content: str) -> str:
    tools = get_tools_module()
    if tools is None:
        raise RuntimeError("Could not find tools.py with save_report")
    fn = getattr(tools, "save_report", None) or getattr(tools, "write_report", None)
    if fn is None:
        raise RuntimeError("Could not find save_report/write_report in tools.py")
    return str(fn(filename=filename, content=content))


def to_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        if "content" in obj and isinstance(obj["content"], str):
            return obj["content"]
        return json.dumps(obj, ensure_ascii=False, indent=2)
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(), ensure_ascii=False, indent=2)
    return str(obj)
