from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.llm import AIPipeClient, TaskParseResult
from app.tools import registry

AGENT_SYSTEM_PROMPT = """You are the Phase 6 quiz-solving agent for the TDS Project 2 system.
You must analyze the quiz question, reason step-by-step, and decide which tool to call next.
DO NOT attempt to submit answers or call submission URLs - that is handled by the orchestrator. Your job is only to solve the quiz and return the answer.
Whenever you respond you MUST emit compact JSON only (no prose) matching this schema:
{"thought":"string","action":"tool_name|final","input":{"arg":"value", ...}}
Rules:
- Keep thought concise (<=40 tokens).
- When action == "final", the input object must contain {"answer":"...", "reasoning":"..."}.
- Tools may accept only the arguments declared in the catalog provided.
- Never hallucinate data; always inspect tool outputs before concluding.
"""


@dataclass(slots=True)
class PlannerDirective:
    """Structured directive emitted by the planner model."""

    thought: str
    action: str
    arguments: Dict[str, Any]


@dataclass(slots=True)
class AgentStep:
    """Trace entry capturing a single reasoning + tool invocation loop."""

    index: int
    thought: str
    action: str
    arguments: Dict[str, Any]
    observation: Any = None


@dataclass(slots=True)
class AgentResult:
    """Final artifact returned by the solver agent."""

    answer: Optional[str]
    reasoning: Optional[str]
    steps: List[AgentStep] = field(default_factory=list)
    transcript: List[Dict[str, Any]] = field(default_factory=list)


class AgentProtocolError(RuntimeError):
    """Raised when the planner emits malformed JSON."""


class ToolExecutionError(RuntimeError):
    """Raised when a requested tool fails."""


class AgentError(RuntimeError):
    """Raised for high-level agent failures (timeouts, run-away reasoning, etc.)."""


def _tool_summary(name: str, func: Any) -> Dict[str, str]:
    doc = inspect.getdoc(func) or "No description provided."
    signature = inspect.signature(func)
    params = ", ".join(signature.parameters)
    return {"name": name, "doc": doc, "signature": f"{name}({params})"}


def build_tool_catalog(
    allowlist: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    names = list(allowlist) if allowlist else registry.names()
    catalog: List[Dict[str, str]] = []
    for name in names:
        try:
            func = registry.get(name)
        except KeyError:
            continue
        catalog.append(_tool_summary(name, func))
    return catalog


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value) or asyncio.isfuture(value):
        return await value
    return value


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        try:
            candidate = value.to_dict()
            json.dumps(candidate)
            return candidate
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


class LLMSolverAgent:
    """Planner/executor loop that coordinates the registry-backed tools."""

    def __init__(
        self,
        *,
        client: Optional[AIPipeClient] = None,
        model: Optional[str] = None,
        max_turns: int = 8,
        tool_allowlist: Optional[Iterable[str]] = None,
    ) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self._client = client
        self._model_override = model
        self.max_turns = max_turns
        self.tool_catalog = build_tool_catalog(tool_allowlist)
        self._current_task_context: Optional[TaskParseResult] = None

    async def solve(
        self,
        task: TaskParseResult,
        *,
        workspace: Path,
    ) -> AgentResult:
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        # Store task context for tools to access (e.g., quiz_url for relative URL resolution)
        self._current_task_context = task

        client = self._client or AIPipeClient()
        transcript = self._bootstrap_transcript(task, workspace)
        steps: List[AgentStep] = []

        try:
            for turn in range(1, self.max_turns + 1):
                response = await client.chat_completion(
                    model=self._model_override or client.settings.llm_default_model,
                    messages=transcript,
                    temperature=0.2,
                    max_output_tokens=768,
                )
                directive = self._parse_response(response)

                if directive.action == "final":
                    answer = (
                        str(directive.arguments.get("answer"))
                        if directive.arguments.get("answer") is not None
                        else None
                    )
                    reasoning = directive.arguments.get("reasoning")
                    steps.append(
                        AgentStep(
                            index=len(steps),
                            thought=directive.thought,
                            action=directive.action,
                            arguments=directive.arguments,
                        )
                    )
                    return AgentResult(
                        answer=answer,
                        reasoning=reasoning,
                        steps=steps,
                        transcript=transcript,
                    )

                observation = await self._invoke_tool(
                    directive.action, directive.arguments
                )
                steps.append(
                    AgentStep(
                        index=len(steps),
                        thought=directive.thought,
                        action=directive.action,
                        arguments=directive.arguments,
                        observation=observation,
                    )
                )

                transcript.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "thought": directive.thought,
                                "action": directive.action,
                                "input": directive.arguments,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
                transcript.append(
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"observation": _json_safe(observation)},
                            ensure_ascii=False,
                        ),
                    }
                )

            raise AgentError("Turn budget exhausted without reaching a final answer.")
        finally:
            if self._client is None:
                await client.close()

    def _bootstrap_transcript(
        self, task: TaskParseResult, workspace: Path
    ) -> List[Dict[str, Any]]:
        catalog_lines = [
            f"- {entry['name']}: {entry['doc']} | Signature: {entry['signature']}"
            for entry in self.tool_catalog
        ]
        catalog_text = (
            "\n".join(catalog_lines) if catalog_lines else "No tools registered."
        )

        context_payload = {
            "question": task.question,
            "answer_format": task.answer_format,
            "instructions": task.instructions,
            "resources": task.resources,
            "submission_url": task.submission_url,
            "workspace": str(workspace),
        }

        user_prompt = (
            "Quiz metadata:\n"
            f"{json.dumps(context_payload, ensure_ascii=False)}\n\n"
            "Tool catalog:\n"
            f"{catalog_text}\n\n"
            "Follow the JSON schema strictly; do not include extra text."
        )

        return [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _parse_response(payload: Dict[str, Any]) -> PlannerDirective:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentProtocolError(f"Unexpected LLM payload: {payload}") from exc

        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content)

        if not isinstance(content, str):
            raise AgentProtocolError("Planner response was not textual JSON.")

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise AgentProtocolError(
                f"Planner emitted invalid JSON: {content}"
            ) from exc

        if "thought" not in data or "action" not in data:
            raise AgentProtocolError(
                "Planner JSON must include 'thought' and 'action'."
            )

        arguments = data.get("input") or {}
        if not isinstance(arguments, dict):
            raise AgentProtocolError("'input' field must be an object.")

        return PlannerDirective(
            thought=str(data["thought"]),
            action=str(data["action"]),
            arguments=arguments,
        )

    async def _invoke_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        try:
            func = registry.get(name)
        except KeyError as exc:
            raise ToolExecutionError(f"Tool '{name}' is not registered.") from exc

        try:
            # Inject quiz_url as base_url for call_json_api to resolve relative URLs
            if name == "call_json_api" and self._current_task_context:
                if "base_url" not in arguments and self._current_task_context.quiz_url:
                    arguments = {
                        **arguments,
                        "base_url": self._current_task_context.quiz_url,
                    }

            return await _maybe_await(func(**arguments))
        except Exception as exc:  # pragma: no cover - delegated tool logic
            raise ToolExecutionError(f"Tool '{name}' failed: {exc}") from exc


__all__ = [
    "AgentResult",
    "AgentStep",
    "AgentError",
    "AgentProtocolError",
    "ToolExecutionError",
    "LLMSolverAgent",
    "build_tool_catalog",
]
