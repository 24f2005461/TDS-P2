#!/usr/bin/env python3
"""
Local test server for TDS Quiz Solver pipeline testing.

Serves a sample quiz page with JSON payload structure and a mock submission endpoint.
Use this to test the end-to-end flow locally without hitting real quiz endpoints.

Usage:
    python scripts/run_test_server.py --port 8888
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS Quiz Test Server")


# Sample quiz HTML with embedded JSON structure
SAMPLE_QUIZ_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Sample Quiz #834</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .question { background: #f5f5f5; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .resources { background: #e8f4f8; padding: 15px; border-radius: 5px; }
        code { background: #333; color: #0f0; padding: 2px 5px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>Quiz #834: Data Analysis Challenge</h1>

    <div class="question">
        <h2>Question</h2>
        <p>Given the CSV file below, what is the sum of all values in the 'amount' column?</p>
        <p><strong>Format:</strong> Return a single number (integer or float)</p>
    </div>

    <div class="resources">
        <h3>Resources</h3>
        <ul>
            <li><a href="/data.csv">data.csv</a> - Dataset for analysis</li>
        </ul>
    </div>

    <div id="result">
        <h3>Submission Instructions</h3>
        <p>Submit your answer to: <code>/submit-quiz-834</code></p>
        <p>Use the following JSON structure:</p>
        <pre id="payload-structure">
{
    "quiz_id": "quiz-834",
    "email": "{{email}}",
    "secret": "{{secret}}",
    "answer": "{{answer}}",
    "metadata": {
        "reasoning": "{{reasoning}}"
    }
}
        </pre>
    </div>

    <script>
        // This script would normally handle form submission
        console.log("Quiz #834 loaded");
    </script>
</body>
</html>
"""

# Sample CSV data
SAMPLE_CSV_DATA = """name,amount,category
Alice,100,A
Bob,150,B
Charlie,75,A
David,200,C
Eve,125,B
"""


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "TDS Quiz Test Server",
        "endpoints": {
            "quiz": "/quiz-834",
            "data": "/data.csv",
            "submit": "/submit-quiz-834",
        },
    }


@app.get("/quiz-834", response_class=HTMLResponse)
async def get_quiz():
    """Serve the sample quiz page."""
    logger.info("Serving quiz-834 page")
    return SAMPLE_QUIZ_HTML


@app.get("/data.csv")
async def get_data():
    """Serve the sample CSV file."""
    logger.info("Serving data.csv")
    return SAMPLE_CSV_DATA


@app.post("/submit-quiz-834")
async def submit_quiz(request: Request):
    """
    Mock quiz submission endpoint.

    Validates the submission and returns a response indicating correctness.
    The correct answer for the sample quiz is 650 (sum of amounts).
    """
    try:
        payload = await request.json()
    except Exception as e:
        logger.error(f"Invalid JSON payload: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON", "correct": False},
        )

    logger.info(f"Received submission: {json.dumps(payload, indent=2)}")

    # Extract answer
    answer = payload.get("answer")
    email = payload.get("email")
    secret = payload.get("secret")

    # Validate required fields
    if not email or not secret:
        logger.warning("Missing email or secret")
        return JSONResponse(
            status_code=403,
            content={"error": "Missing credentials", "correct": False},
        )

    # Check answer (correct answer is 650)
    correct_answer = 650
    is_correct = False

    try:
        # Handle various answer formats
        if isinstance(answer, (int, float)):
            is_correct = abs(float(answer) - correct_answer) < 0.01
        elif isinstance(answer, str):
            is_correct = abs(float(answer) - correct_answer) < 0.01
    except (ValueError, TypeError):
        logger.warning(f"Invalid answer format: {answer}")

    # Build response
    response: Dict[str, Any] = {
        "correct": is_correct,
        "url": None,
        "reason": None,
    }

    if is_correct:
        logger.info(f"✓ Correct answer received: {answer}")
        response["message"] = "Congratulations! Your answer is correct."
    else:
        logger.info(
            f"✗ Incorrect answer received: {answer} (expected {correct_answer})"
        )
        response["message"] = f"Incorrect. Expected {correct_answer}, got {answer}"
        response["reason"] = "Answer does not match expected value"

    return JSONResponse(content=response)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Run the test server."""
    parser = argparse.ArgumentParser(
        description="Run local test server for TDS Quiz Solver"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="Port to run the server on (default: 8888)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes",
    )

    args = parser.parse_args()

    logger.info(f"Starting test server on http://{args.host}:{args.port}")
    logger.info(f"Quiz available at: http://{args.host}:{args.port}/quiz-834")
    logger.info(f"Data file at: http://{args.host}:{args.port}/data.csv")
    logger.info(f"Submit endpoint: http://{args.host}:{args.port}/submit-quiz-834")

    uvicorn.run(
        "scripts.run_test_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
