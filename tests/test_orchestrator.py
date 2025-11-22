import asyncio
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.agent import AgentResult, AgentStep
from app.browser import ScrapeResult
from app.llm import TaskParseResult
from app.orchestrator import OrchestratorTimeout, QuizOrchestrator


class StubScraper:
    def __init__(self, result: ScrapeResult):
        self.result = result
        self.calls: List[str] = []

    async def __call__(self, url: str) -> ScrapeResult:
        self.calls.append(url)
        return self.result


class StubParser:
    def __init__(self, result: TaskParseResult):
        self.result = result
        self.calls: List[str] = []

    async def __call__(self, html: str) -> TaskParseResult:
        self.calls.append(html)
        return self.result


class StubAgent:
    def __init__(self, result: AgentResult, delay: float = 0.0):
        self.result = result
        self.calls: List[TaskParseResult] = []
        self.delay = delay

    async def solve(self, task: TaskParseResult, *, workspace: Path) -> AgentResult:
        self.calls.append(task)
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.result


class StubSubmitter:
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        self.calls: List[Dict[str, Any]] = []

    async def __call__(
        self,
        *,
        submission_url: str,
        email: str,
        secret: str,
        quiz_url: str,
        answer: Any,
        extras: Dict[str, Any] | None = None,
        payload_template: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        self.calls.append(
            {
                "submission_url": submission_url,
                "email": email,
                "secret": secret,
                "quiz_url": quiz_url,
                "answer": answer,
                "extras": extras,
                "payload_template": payload_template,
            }
        )
        return dict(self.payload)


def _scrape_result(
    submission_url: str | None = "https://example.com/submit",
) -> ScrapeResult:
    return ScrapeResult(
        requested_url="https://example.com/quiz",
        final_url="https://example.com/quiz",
        html="<html></html>",
        text="Question?",
        links=[],
        resource_links=["https://example.com/data.csv"],
        submit_urls=[submission_url] if submission_url else [],
        submission_url=submission_url,
        question_snippet="Q?",
        instructions_snippet="Submit via link",
    )


def _parser_result(
    submission_url: str | None = "https://example.com/submit",
) -> TaskParseResult:
    return TaskParseResult(
        question="What is 2+2?",
        submission_url=submission_url,
        answer_format="number",
        resources=[],
        instructions="Add numbers",
        reasoning=None,
    )


def _agent_result(answer: str = "4") -> AgentResult:
    return AgentResult(
        answer=answer,
        reasoning="basic math",
        steps=[
            AgentStep(
                index=0,
                thought="thought",
                action="none",
                arguments={},
                observation=None,
            )
        ],
        transcript=[{"role": "assistant", "content": "done"}],
    )


@pytest.mark.anyio
async def test_orchestrator_success_single_attempt():
    scraper = StubScraper(_scrape_result())
    parser = StubParser(_parser_result())
    agent = StubAgent(_agent_result())
    submitter = StubSubmitter({"correct": True, "url": None, "reason": None})

    orchestrator = QuizOrchestrator(
        scraper=scraper,
        parser=parser,
        agent_factory=lambda: agent,
        submitter=submitter,
        total_timeout_seconds=60,
        per_attempt_timeout_seconds=10,
        max_chain_depth=2,
    )

    result = await orchestrator.process_quiz(
        initial_url="https://example.com/quiz",
        email="student@example.com",
        secret="s3cr3t",
    )

    assert result.solved is True
    assert result.exhausted is False
    assert len(result.attempts) == 1

    attempt = result.attempts[0]
    assert attempt.quiz_url == "https://example.com/quiz"
    assert attempt.submission_url == "https://example.com/submit"
    assert attempt.agent_answer == "4"
    assert attempt.correct is True
    assert attempt.next_url is None
    assert attempt.steps_taken == 1
    assert attempt.scrape_snapshot is not None
    assert attempt.scrape_snapshot["resource_links"] == ["https://example.com/data.csv"]

    assert scraper.calls == ["https://example.com/quiz"]
    assert parser.calls == ["<html></html>"]
    assert submitter.calls[0]["answer"] == "4"
    assert submitter.calls[0]["extras"] == {"reasoning": "basic math"}


@pytest.mark.anyio
async def test_orchestrator_uses_scrape_submission_url_when_parser_lacks():
    scraper = StubScraper(
        _scrape_result(submission_url="https://example.com/fallback-submit")
    )
    parser = StubParser(_parser_result(submission_url=None))
    agent = StubAgent(_agent_result("42"))
    submitter = StubSubmitter(
        {"correct": False, "url": "https://example.com/next", "reason": "try again"}
    )

    orchestrator = QuizOrchestrator(
        scraper=scraper,
        parser=parser,
        agent_factory=lambda: agent,
        submitter=submitter,
        total_timeout_seconds=30,
        per_attempt_timeout_seconds=5,
    )

    result = await orchestrator.process_quiz(
        initial_url="https://example.com/quiz",
        email="student@example.com",
        secret="s3cr3t",
    )

    attempt = result.attempts[0]
    assert attempt.submission_url == "https://example.com/fallback-submit"
    assert submitter.calls[0]["submission_url"] == "https://example.com/fallback-submit"
    assert result.solved is False
    assert result.final_url == "https://example.com/next"


@pytest.mark.anyio
async def test_orchestrator_raises_timeout_when_attempt_exceeds_budget():
    scraper = StubScraper(_scrape_result())
    parser = StubParser(_parser_result())
    agent = StubAgent(_agent_result(), delay=0.2)
    submitter = StubSubmitter({"correct": True, "url": None, "reason": None})

    orchestrator = QuizOrchestrator(
        scraper=scraper,
        parser=parser,
        agent_factory=lambda: agent,
        submitter=submitter,
        total_timeout_seconds=1,
        per_attempt_timeout_seconds=0.05,
    )

    with pytest.raises(OrchestratorTimeout):
        await orchestrator.process_quiz(
            initial_url="https://example.com/quiz",
            email="student@example.com",
            secret="s3cr3t",
        )
