#!/usr/bin/env python3
"""
Test script for the official TDS demo endpoint.

This script validates your /solve endpoint implementation against the official
demo quiz provided by TDS at https://tds-llm-analysis.s-anand.net/demo

The demo endpoint simulates the quiz process and is the recommended way to test
your implementation before submitting.

Usage:
    # Test with credentials from .env
    python scripts/test_demo_endpoint.py

    # Test with explicit credentials
    python scripts/test_demo_endpoint.py --email your@email.com --secret your-secret

    # Test with custom API endpoint
    python scripts/test_demo_endpoint.py --api http://localhost:8000

    # Verbose output
    python scripts/test_demo_endpoint.py -v
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict

import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Official demo endpoint
DEMO_QUIZ_URL = "https://tds-llm-analysis.s-anand.net/demo"


async def test_demo_endpoint(
    api_base_url: str,
    email: str,
    secret: str,
    timeout_seconds: float = 60.0,
) -> Dict[str, Any]:
    """
    Test the /solve endpoint against the official demo quiz.

    Args:
        api_base_url: Base URL of your API server
        email: Your email for authentication
        secret: Your secret key for authentication
        timeout_seconds: Request timeout in seconds

    Returns:
        Dict containing test results
    """
    results: Dict[str, Any] = {
        "success": False,
        "status_code": None,
        "response_time_ms": None,
        "response_body": None,
        "error": None,
    }

    # Build request payload
    payload = {
        "email": email,
        "secret": secret,
        "url": DEMO_QUIZ_URL,
    }

    logger.info("=" * 70)
    logger.info("Testing Official TDS Demo Endpoint")
    logger.info("=" * 70)
    logger.info(f"Demo Quiz URL: {DEMO_QUIZ_URL}")
    logger.info(f"Your API Endpoint: {api_base_url}/solve")
    logger.info(f"Email: {email}")
    logger.info(f"Timeout: {timeout_seconds}s")
    logger.info("")

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            logger.info("Sending POST request to /solve endpoint...")
            logger.info(f"Payload: {json.dumps(payload, indent=2)}")
            logger.info("")

            start_time = time.time()

            response = await client.post(
                f"{api_base_url}/solve",
                json=payload,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            results["status_code"] = response.status_code
            results["response_time_ms"] = round(elapsed_ms, 2)

            try:
                results["response_body"] = response.json()
            except Exception:
                results["response_body"] = response.text

            # Check response status
            if response.status_code == 200:
                results["success"] = True
                logger.info(f"✓ SUCCESS: /solve returned 200 OK in {elapsed_ms:.2f}ms")
                logger.info("")
                logger.info("Response body:")
                logger.info(json.dumps(results["response_body"], indent=2))
                logger.info("")
                logger.info("Your endpoint is working correctly!")
                logger.info("The quiz is being solved in the background.")
            elif response.status_code == 400:
                logger.error("✗ ERROR: Bad Request (400)")
                logger.error("The request payload is malformed.")
                logger.error(f"Response: {results['response_body']}")
                results["error"] = "Bad Request - check payload format"
            elif response.status_code == 403:
                logger.error("✗ ERROR: Forbidden (403)")
                logger.error("The email/secret combination is invalid.")
                logger.error("Check your .env file or command-line arguments.")
                results["error"] = "Forbidden - invalid credentials"
            else:
                logger.error(f"✗ ERROR: Unexpected status code {response.status_code}")
                logger.error(f"Response: {results['response_body']}")
                results["error"] = f"Unexpected status code: {response.status_code}"

    except httpx.TimeoutException as e:
        logger.error(f"✗ TIMEOUT: Request timed out after {timeout_seconds}s")
        logger.error("This could indicate:")
        logger.error("  - Your API server is not responding")
        logger.error("  - The demo quiz is taking too long to solve")
        logger.error("  - Network connectivity issues")
        results["error"] = f"Timeout: {e}"
    except httpx.ConnectError as e:
        logger.error(f"✗ CONNECTION ERROR: Cannot connect to {api_base_url}")
        logger.error("Make sure your API server is running:")
        logger.error("  uvicorn app.main:app --port 8000")
        results["error"] = f"Connection error: {e}"
    except Exception as e:
        logger.error(f"✗ UNEXPECTED ERROR: {e}")
        results["error"] = str(e)

    return results


async def check_health(api_base_url: str) -> bool:
    """Check if the API server is healthy."""
    logger.info("Checking API server health...")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{api_base_url}/health")

            if response.status_code == 200:
                logger.info("✓ API server is healthy")
                return True
            else:
                logger.warning(
                    f"⚠ API server returned {response.status_code} for health check"
                )
                return False

    except Exception as e:
        logger.error(f"✗ Health check failed: {e}")
        return False


def load_credentials_from_env() -> tuple[str | None, str | None]:
    """Load email and secret from environment variables."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    email = os.getenv("EMAIL")
    secret = os.getenv("SECRET")

    return email, secret


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test your /solve endpoint against the official TDS demo quiz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Official Demo Endpoint:
  The official TDS demo quiz is available at:
  https://tds-llm-analysis.s-anand.net/demo

  This endpoint simulates the quiz process and is the recommended way to test
  your implementation before final submission.

Expected Behavior:
  1. Your /solve endpoint should return HTTP 200 immediately (acknowledge receipt)
  2. Your system should solve the quiz in the background
  3. The background process should:
     - Scrape the quiz page
     - Parse the task with LLM
     - Solve using the agent
     - Submit the answer to the quiz's submission endpoint
     - Handle any chained quizzes

Sample Payload:
  {
    "email": "your@email.com",
    "secret": "your-secret-key",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
  }

Examples:
  # Test with credentials from .env
  python scripts/test_demo_endpoint.py

  # Test with explicit credentials
  python scripts/test_demo_endpoint.py --email your@email.com --secret your-secret

  # Test with custom API endpoint
  python scripts/test_demo_endpoint.py --api http://my-server.com

  # Verbose logging
  python scripts/test_demo_endpoint.py -v
        """,
    )

    parser.add_argument(
        "--api",
        type=str,
        default="http://localhost:8000",
        help="Base URL of your API server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--email",
        type=str,
        help="Email for authentication (optional, uses .env default)",
    )
    parser.add_argument(
        "--secret",
        type=str,
        help="Secret for authentication (optional, uses .env default)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip health check",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load credentials
    env_email, env_secret = load_credentials_from_env()

    email = args.email or env_email
    secret = args.secret or env_secret

    # Validate credentials
    if not email or not secret:
        logger.error("✗ ERROR: Email and secret are required")
        logger.error("")
        logger.error("Provide credentials via:")
        logger.error("  1. Command-line arguments: --email and --secret")
        logger.error("  2. Environment variables: EMAIL and SECRET in .env file")
        logger.error("")
        logger.error("Example .env file:")
        logger.error("  EMAIL=your@email.com")
        logger.error("  SECRET=your-secret-key")
        sys.exit(1)

    # Run tests
    try:
        # Health check
        if not args.skip_health:
            asyncio.run(check_health(args.api))
            logger.info("")

        # Run demo endpoint test
        results = asyncio.run(
            test_demo_endpoint(
                api_base_url=args.api,
                email=email,
                secret=secret,
                timeout_seconds=args.timeout,
            )
        )

        # Summary
        logger.info("=" * 70)
        logger.info("Test Summary")
        logger.info("=" * 70)

        if results["success"]:
            logger.info("✓ Test PASSED")
            logger.info("")
            logger.info("Next Steps:")
            logger.info(
                "  1. Monitor your API server logs to verify background processing"
            )
            logger.info("  2. Look for successful quiz solving and submission messages")
            logger.info(
                "  3. If everything works, you're ready to submit your endpoint!"
            )
            logger.info("")
            logger.info("Submission Instructions:")
            logger.info(
                "  Submit your endpoint URL via the Google Form provided by TDS"
            )
            logger.info(
                "  Make sure your endpoint is publicly accessible (not localhost)"
            )
            sys.exit(0)
        else:
            logger.error("✗ Test FAILED")
            logger.error("")
            logger.error(f"Error: {results['error']}")
            logger.error("")
            logger.error("Troubleshooting:")
            logger.error("  1. Make sure your API server is running")
            logger.error("  2. Check that EMAIL and SECRET are correct")
            logger.error("  3. Verify your /solve endpoint implementation")
            logger.error("  4. Check API server logs for errors")
            logger.error("  5. Ensure all dependencies are installed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
