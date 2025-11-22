import json

import pytest

from app.agent import LLMSolverAgent, build_tool_catalog
from app.llm import TaskParseResult
from app.tools import registry


@pytest.mark.anyio
async def test_agent_executes_tool_and_returns_answer(tmp_path):
    tool_name = "test_fake_tool_agent"
    calls = []

    async def fake_tool(value: str):
        calls.append(value)
        return {"echo": value.upper()}

    registry.register(tool_name, fake_tool)

    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "thought": "Need to inspect helper.",
                                "action": tool_name,
                                "input": {"value": "hello"},
                            }
                        )
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "thought": "Ready to answer.",
                                "action": "final",
                                "input": {
                                    "answer": "42",
                                    "reasoning": "Test reasoning",
                                },
                            }
                        )
                    }
                }
            ]
        },
    ]

    class StubClient:
        def __init__(self, queued):
            self._queued = queued
            self.settings = type("Settings", (), {"llm_default_model": "stub-model"})

        async def chat_completion(self, **kwargs):
            if not self._queued:
                raise AssertionError("No more stub responses.")
            return self._queued.pop(0)

    agent = LLMSolverAgent(
        client=StubClient(responses.copy()),  # type: ignore[arg-type]
        tool_allowlist=[tool_name],
    )

    task = TaskParseResult(
        question="What is the answer?",
        submission_url=None,
        answer_format="number",
        resources=[],
        instructions="Use helper, then answer.",
        reasoning=None,
    )
    result = await agent.solve(task, workspace=tmp_path)

    assert calls == ["hello"]
    assert result.answer == "42"
    assert result.reasoning == "Test reasoning"
    assert len(result.steps) == 2
    assert result.steps[0].action == tool_name
    assert result.steps[1].action == "final"


def test_build_tool_catalog_returns_registered_entries():
    tool_name = "test_catalog_tool"

    async def noop():
        return "ok"

    registry.register(tool_name, noop)
    catalog = build_tool_catalog([tool_name])

    assert len(catalog) == 1
    entry = catalog[0]
    assert entry["name"] == tool_name
    assert tool_name in entry["signature"]
    assert isinstance(entry["doc"], str)
