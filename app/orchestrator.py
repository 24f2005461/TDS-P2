from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from app.agent import AgentResult, LLMSolverAgent
from app.browser import ScrapeResult
from app.config import Settings, get_settings
from app.llm import TaskParseResult
from app.logging_utils import get_logger

logger = get_logger(__name__)


class Scraper(Protocol):
    async def __call__(self, url: str) -> ScrapeResult: ...


class Parser(Protocol):
    async def __call__(self, html: str) -> TaskParseResult: ...


class SolverAgent(Protocol):
    """Protocol for any agent that can solve quiz tasks."""

    async def solve(self, task: TaskParseResult, *, workspace: Path) -> AgentResult: ...


class AgentFactory(Protocol):
    def __call__(self) -> SolverAgent: ...


class Submitter(Protocol):
    async def __call__(
        self,
        *,
        submission_url: str,
        email: str,
        secret: str,
        quiz_url: str,
        answer: Any,
        extras: Optional[Dict[str, Any]] = None,
        payload_template: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...


class OrchestratorError(RuntimeError):
    """Base class for orchestration failures."""


class OrchestratorTimeout(OrchestratorError):
    """Raised when the overall deadline is exceeded."""


@dataclass(slots=True)
class SubmissionAttempt:
    """Represents a single scrape → solve → submit attempt."""

    quiz_url: str
    submission_url: Optional[str]
    agent_answer: Optional[Any]
    reasoning: Optional[str]
    correct: Optional[bool]
    next_url: Optional[str]
    reason: Optional[str]
    steps_taken: int
    elapsed_seconds: float
    agent_transcript_tail: List[Any] = field(default_factory=list)
    scrape_snapshot: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass(slots=True)
class OrchestratorResult:
    """Top-level result covering the entire quiz chain run."""

    attempts: List[SubmissionAttempt]
    solved: bool
    total_elapsed_seconds: float
    exhausted: bool
    final_url: Optional[str]


class QuizOrchestrator:
    """
    Coordinates the Phase 8 workflow:

    1. Scrape the quiz page.
    2. Parse instructions/resources with the LLM parser.
    3. Invoke the Phase 6 solver agent inside a workspace.
    4. Submit the computed answer and inspect responses for follow-up URLs.
    5. Repeat until completion, max chain depth, or timeout.
    """

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        scraper: Optional[Scraper] = None,
        parser: Optional[Parser] = None,
        agent_factory: Optional[AgentFactory] = None,
        submitter: Optional[Submitter] = None,
        workspace_root: Path | str = Path("demo_workspace"),
        total_timeout_seconds: float = 180.0,
        per_attempt_timeout_seconds: Optional[float] = None,
        max_chain_depth: int = 5,
    ) -> None:
        self.settings = settings or get_settings()
        self.scraper = scraper
        self.parser = parser
        self.agent_factory = agent_factory or (lambda: LLMSolverAgent())
        self.submitter = submitter
        self.workspace_root = Path(workspace_root)
        self.total_timeout_seconds = total_timeout_seconds
        self.per_attempt_timeout_seconds = per_attempt_timeout_seconds
        self.max_chain_depth = max_chain_depth
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    async def process_quiz(
        self,
        *,
        initial_url: str,
        email: Optional[str] = None,
        secret: Optional[str] = None,
    ) -> OrchestratorResult:
        resolved_email = email or self.settings.email
        resolved_secret = secret or self.settings.secret

        attempts: List[SubmissionAttempt] = []
        current_url: Optional[str] = initial_url
        solved = False
        exhausted = False
        start_time = time.monotonic()

        for depth in range(1, self.max_chain_depth + 1):
            if not current_url:
                break

            remaining = self._remaining_time(start_time)
            if remaining <= 0:
                raise OrchestratorTimeout("Overall time budget exhausted.")

            attempt_timeout = min(
                remaining, self.per_attempt_timeout_seconds or remaining
            )
            logger.info("Phase 8 attempt %d for %s", depth, current_url)

            try:
                attempt = await asyncio.wait_for(
                    self._attempt(
                        quiz_url=current_url,
                        attempt_index=depth,
                        email=resolved_email,
                        secret=resolved_secret,
                    ),
                    timeout=attempt_timeout,
                )
            except asyncio.TimeoutError as exc:
                exhausted = True
                raise OrchestratorTimeout("Attempt timed out.") from exc
            except OrchestratorError as exc:
                attempt = SubmissionAttempt(
                    quiz_url=current_url,
                    submission_url=None,
                    agent_answer=None,
                    reasoning=None,
                    correct=None,
                    next_url=None,
                    reason=str(exc),
                    steps_taken=0,
                    elapsed_seconds=self.total_timeout_seconds - remaining,
                    error=str(exc),
                )

            attempts.append(attempt)

            if attempt.correct:
                solved = True

            if not attempt.next_url:
                current_url = attempt.next_url
                break

            current_url = attempt.next_url

        total_elapsed = time.monotonic() - start_time
        if len(attempts) >= self.max_chain_depth and current_url:
            exhausted = True

        return OrchestratorResult(
            attempts=attempts,
            solved=solved,
            total_elapsed_seconds=total_elapsed,
            exhausted=exhausted,
            final_url=current_url,
        )

    async def _attempt(
        self,
        *,
        quiz_url: str,
        attempt_index: int,
        email: str,
        secret: str,
    ) -> SubmissionAttempt:
        attempt_start = time.monotonic()
        workspace = self.workspace_root / f"attempt_{attempt_index:02d}"
        workspace.mkdir(parents=True, exist_ok=True)

        scrape_result = await self._scrape(quiz_url)
        parser_result = await self._parse(scrape_result.html, quiz_url=quiz_url)
        submission_url = self._pick_submission_url(parser_result, scrape_result)
        self._merge_resources(parser_result, scrape_result)

        agent = self.agent_factory()
        agent_result = await agent.solve(parser_result, workspace=workspace)

        if not agent_result.answer:
            raise OrchestratorError("Agent produced no answer.")

        submission_payload = await self._submit(
            submission_url=submission_url,
            email=email,
            secret=secret,
            quiz_url=quiz_url,
            answer=agent_result.answer,
            reasoning=agent_result.reasoning,
            payload_template=parser_result.payload_template,
        )

        elapsed = time.monotonic() - attempt_start

        return SubmissionAttempt(
            quiz_url=quiz_url,
            submission_url=submission_url,
            agent_answer=agent_result.answer,
            reasoning=agent_result.reasoning,
            correct=submission_payload.get("correct"),
            next_url=submission_payload.get("url"),
            reason=submission_payload.get("reason"),
            steps_taken=len(agent_result.steps),
            elapsed_seconds=elapsed,
            agent_transcript_tail=agent_result.transcript[-4:],
            scrape_snapshot=self._snapshot_scrape(scrape_result),
        )

    async def _scrape(self, url: str) -> ScrapeResult:
        if self.scraper:
            return await self.scraper(url)
        from app.browser import scrape_quiz_page

        return await scrape_quiz_page(url)

    async def _parse(
        self, html: str, quiz_url: Optional[str] = None
    ) -> TaskParseResult:
        if self.parser:
            return await self.parser(html)
        from app.llm import parse_task

        return await parse_task(html, quiz_url=quiz_url)

    async def _submit(
        self,
        *,
        submission_url: str,
        email: str,
        secret: str,
        quiz_url: str,
        answer: Any,
        reasoning: Optional[str],
        payload_template: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        extras = {"reasoning": reasoning} if reasoning else None
        if self.submitter:
            return await self.submitter(
                submission_url=submission_url,
                email=email,
                secret=secret,
                quiz_url=quiz_url,
                answer=answer,
                extras=extras,
                payload_template=payload_template,
            )
        from app.submission import submit_answer

        return (
            await submit_answer(
                submission_url=submission_url,
                email=email,
                secret=secret,
                quiz_url=quiz_url,
                answer=answer,
                extras=extras,
                payload_template=payload_template,
            )
        ).payload

    @staticmethod
    def _snapshot_scrape(scrape: ScrapeResult) -> Dict[str, Any]:
        return {
            "requested_url": scrape.requested_url,
            "final_url": scrape.final_url,
            "submission_url": scrape.submission_url,
            "resource_links": scrape.resource_links,
            "question_snippet": scrape.question_snippet,
            "instructions_snippet": scrape.instructions_snippet,
        }

    @staticmethod
    def _pick_submission_url(
        parsed: TaskParseResult,
        scrape: ScrapeResult,
    ) -> str:
        if parsed.submission_url:
            return parsed.submission_url
        if scrape.submission_url:
            parsed.submission_url = scrape.submission_url
            return scrape.submission_url
        if scrape.submit_urls:
            parsed.submission_url = scrape.submit_urls[0]
            return scrape.submit_urls[0]
        raise OrchestratorError("Submission URL could not be determined.")

    @staticmethod
    def _merge_resources(task: TaskParseResult, scrape: ScrapeResult) -> None:
        resources = list(task.resources or [])
        existing = set(resources)
        for link in scrape.resource_links:
            if link not in existing:
                resources.append(link)
                existing.add(link)
        task.resources = resources

    def _remaining_time(self, start_time: float) -> float:
        elapsed = time.monotonic() - start_time
        return max(0.0, self.total_timeout_seconds - elapsed)


__all__ = [
    "QuizOrchestrator",
    "OrchestratorResult",
    "SubmissionAttempt",
    "OrchestratorError",
    "OrchestratorTimeout",
]
