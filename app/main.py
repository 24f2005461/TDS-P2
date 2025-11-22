from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.logging_utils import configure_logging, get_logger
from app.orchestrator import QuizOrchestrator
from app.schemas import HealthResponse, SolveRequest, SolveResponse

logger = get_logger(__name__)
settings: Settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "FastAPI entrypoint for the TDS Project 2 LLM quiz solver. "
        "Phase 2 implements authentication, validation, and structured logging."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def _build_request_id(request: Request) -> str:
    header_id = request.headers.get("x-request-id")
    request_id = header_id or str(uuid4())
    request.state.request_id = request_id
    return request_id


def request_id_dependency(request: Request) -> str:
    return getattr(request.state, "request_id", _build_request_id(request))


@app.on_event("startup")
def startup_event() -> None:
    from typing import Literal, cast

    log_level = cast(
        Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        settings.log_level.upper(),
    )
    configure_logging(log_level)
    logger.info(
        "Application startup complete",
        extra={"request_id": "-", "environment": settings.environment},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    logger.warning(
        "Malformed request payload",
        extra={"request_id": request_id, "errors": exc.errors()},
    )
    response = SolveResponse(
        status="error",
        accepted=False,
        reason="Invalid or malformed JSON payload.",
        details={"errors": exc.errors()},
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST, content=response.model_dump()
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health probe",
    tags=["Diagnostics"],
)
async def health_check() -> HealthResponse:
    message = settings.describe_llm_capabilities()
    return HealthResponse(
        status="healthy",
        llm_provider=settings.llm_default_model,
        message=message,
    )


@app.post(
    "/solve",
    response_model=SolveResponse,
    summary="Primary quiz-solving endpoint",
    tags=["Quiz"],
)
async def solve_endpoint(
    payload: SolveRequest,
    request: Request,
    request_id: str = Depends(request_id_dependency),
) -> JSONResponse:
    logger.info(
        "Solve request received",
        extra={
            "request_id": request_id,
            "url": str(payload.url),
            "email": payload.email,
        },
    )

    if payload.secret != settings.secret:
        logger.warning(
            "Rejected request due to invalid secret",
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid secret provided."
        )

    # Acknowledge receipt immediately and solve asynchronously
    # The TDS platform expects HTTP 200 to confirm we received the request
    # Our system then solves the quiz and submits to the quiz's endpoint

    # Start background task to solve the quiz
    async def solve_quiz_background():
        """Background task to solve the quiz and submit answer."""
        try:
            orchestrator = QuizOrchestrator(
                workspace_root="workspace",
                total_timeout_seconds=180,
                max_chain_depth=5,
            )

            logger.info(
                "Starting quiz solving in background",
                extra={"request_id": request_id, "url": str(payload.url)},
            )

            result = await orchestrator.process_quiz(
                initial_url=str(payload.url),
                email=payload.email,
                secret=payload.secret,
            )

            logger.info(
                "Quiz solving completed",
                extra={
                    "request_id": request_id,
                    "attempts": result.attempts,
                    "time": result.total_elapsed_seconds,
                    "status": result.solved,
                },
            )

        except Exception as e:
            logger.error(
                "Error solving quiz in background",
                extra={"request_id": request_id, "error": str(e)},
                exc_info=True,
            )

    # Spawn background task (fire and forget)
    import asyncio

    asyncio.create_task(solve_quiz_background())

    # Return 200 immediately to acknowledge receipt
    details: Dict[str, Any] = {
        "message": "Quiz request received and being processed",
        "quiz_url": str(payload.url),
        "request_id": request_id,
    }

    response = SolveResponse(
        status="ok",
        accepted=True,
        reason=None,
        details=details,
    )

    logger.info(
        "Quiz request accepted, solving in background",
        extra={"request_id": request_id},
    )

    return JSONResponse(status_code=status.HTTP_200_OK, content=response.model_dump())


@app.post("/", include_in_schema=False)
async def solve_root(
    payload: SolveRequest,
    request: Request,
    request_id: str = Depends(request_id_dependency),
) -> JSONResponse:
    """Support POST requests to the root path by delegating to /solve."""
    return await solve_endpoint(payload, request, request_id)


@app.get("/", include_in_schema=False)
async def root_redirect() -> Dict[str, str]:
    return {"message": "Refer to /solve for quiz submissions and /healthz for status."}


def get_application() -> FastAPI:
    return app


__all__ = ["app", "get_application"]
