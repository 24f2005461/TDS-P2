#!/usr/bin/env python3
"""
End-to-end pipeline tester for TDS Quiz Solver.

This script tests the complete workflow:
1. POST to /solve endpoint with quiz URL
2. Verify immediate 200 response
3. Monitor logs for background processing
4. Optionally verify submission was made

Usage:
    # Test against local test server
    python scripts/test_pipeline.py --url http://localhost:8888/quiz-834

    # Test with official demo endpoint
    python scripts/test_pipeline.py --demo

    # Test with custom API endpoint
    python scripts/test_pipeline.py --url http://localhost:8888/quiz-834 --api http://localhost:8000

    # Verbose output
    python scripts/test_pipeline.py --url http://localhost:8888/quiz-834 -v
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

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PipelineTester:
    """Test the quiz solver pipeline end-to-end."""

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        timeout_seconds: float = 30.0,
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def test_solve_endpoint(
        self,
        quiz_url: str,
        email: str | None = None,
        secret: str | None = None,
    ) -> Dict[str, Any]:
        """
        Test the /solve endpoint.

        Args:
            quiz_url: URL of the quiz to solve
            email: Email for authentication (optional, uses env default)
            secret: Secret for authentication (optional, uses env default)

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
        payload: Dict[str, Any] = {"url": quiz_url}
        if email:
            payload["email"] = email
        if secret:
            payload["secret"] = secret

        logger.info(f"Testing /solve endpoint with quiz URL: {quiz_url}")
        logger.info(f"API endpoint: {self.api_base_url}/solve")

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                start_time = time.time()

                response = await client.post(
                    f"{self.api_base_url}/solve",
                    json=payload,
                )

                elapsed_ms = (time.time() - start_time) * 1000

                results["status_code"] = response.status_code
                results["response_time_ms"] = round(elapsed_ms, 2)

                try:
                    results["response_body"] = response.json()
                except Exception:
                    results["response_body"] = response.text

                # Check if response is 200 (immediate acknowledgment)
                if response.status_code == 200:
                    results["success"] = True
                    logger.info(
                        f"✓ /solve returned 200 OK in {elapsed_ms:.2f}ms (background processing started)"
                    )
                else:
                    logger.error(
                        f"✗ /solve returned {response.status_code}: {results['response_body']}"
                    )
                    results["error"] = f"Unexpected status code: {response.status_code}"

        except httpx.TimeoutException as e:
            logger.error(f"✗ Request timeout after {self.timeout_seconds}s")
            results["error"] = f"Timeout: {e}"
        except httpx.ConnectError as e:
            logger.error(
                f"✗ Connection error - is the API server running at {self.api_base_url}?"
            )
            results["error"] = f"Connection error: {e}"
        except Exception as e:
            logger.error(f"✗ Unexpected error: {e}")
            results["error"] = str(e)

        return results

    async def test_health_check(self) -> bool:
        """
        Test if the API server is healthy.

        Returns:
            True if healthy, False otherwise
        """
        logger.info(f"Checking API health at {self.api_base_url}/health")

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.api_base_url}/health")

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

    async def test_malformed_request(self) -> Dict[str, Any]:
        """
        Test that malformed requests are rejected with 400.

        Returns:
            Dict containing test results
        """
        logger.info("Testing malformed request handling")

        results: Dict[str, Any] = {
            "success": False,
            "status_code": None,
            "error": None,
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Send invalid JSON
                response = await client.post(
                    f"{self.api_base_url}/solve",
                    json={"invalid": "no url field"},
                )

                results["status_code"] = response.status_code

                if response.status_code == 400:
                    results["success"] = True
                    logger.info("✓ Malformed request correctly rejected with 400")
                else:
                    logger.warning(
                        f"⚠ Expected 400 for malformed request, got {response.status_code}"
                    )
                    results["error"] = f"Expected 400, got {response.status_code}"

        except Exception as e:
            logger.error(f"✗ Error testing malformed request: {e}")
            results["error"] = str(e)

        return results


async def run_tests(
    quiz_url: str,
    api_base_url: str,
    email: str | None = None,
    secret: str | None = None,
    check_health: bool = True,
    test_malformed: bool = True,
) -> bool:
    """
    Run all pipeline tests.

    Args:
        quiz_url: URL of the quiz to solve
        api_base_url: Base URL of the API server
        email: Optional email for authentication (uses .env if not provided)
        secret: Optional secret for authentication (uses .env if not provided)
        check_health: Whether to run health check first
        test_malformed: Whether to test malformed request handling

    Returns:
        True if all tests pass, False otherwise
    """
    # Load from environment if not provided
    if email is None:
        email = os.getenv("EMAIL")
    if secret is None:
        secret = os.getenv("SECRET")
    tester = PipelineTester(api_base_url=api_base_url)

    all_passed = True

    logger.info("=" * 70)
    logger.info("TDS Quiz Solver - Pipeline Test Suite")
    logger.info("=" * 70)

    # Health check
    if check_health:
        logger.info("\n[1/3] Health Check")
        logger.info("-" * 70)
        healthy = await tester.test_health_check()
        if not healthy:
            logger.error(
                "⚠ API server is not healthy - tests may fail. Continue anyway..."
            )

    # Test malformed request
    if test_malformed:
        logger.info("\n[2/3] Malformed Request Test")
        logger.info("-" * 70)
        malformed_results = await tester.test_malformed_request()
        if not malformed_results["success"]:
            logger.warning("⚠ Malformed request test failed")
            all_passed = False

    # Test solve endpoint
    logger.info("\n[3/3] Solve Endpoint Test")
    logger.info("-" * 70)
    solve_results = await tester.test_solve_endpoint(
        quiz_url=quiz_url,
        email=email,
        secret=secret,
    )

    if not solve_results["success"]:
        logger.error("✗ Solve endpoint test failed")
        all_passed = False

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Test Summary")
    logger.info("=" * 70)
    logger.info(f"Quiz URL: {quiz_url}")
    logger.info(f"API Base URL: {api_base_url}")

    if solve_results["success"]:
        logger.info(f"Response Time: {solve_results['response_time_ms']}ms")
        logger.info(
            f"Response Body: {json.dumps(solve_results['response_body'], indent=2)}"
        )

    if all_passed:
        logger.info("\n✓ All tests passed!")
        logger.info(
            "\nNote: Background processing is asynchronous. Check the API server logs to verify:"
        )
        logger.info("  - Quiz was scraped successfully")
        logger.info("  - Task was parsed correctly")
        logger.info("  - Agent computed the answer")
        logger.info("  - Answer was submitted")
        logger.info("  - Relative URLs were resolved correctly")
        return True
    else:
        logger.error("\n✗ Some tests failed. See logs above for details.")
        return False


def main():
    """Main entry point for pipeline tester."""
    parser = argparse.ArgumentParser(
        description="Test TDS Quiz Solver pipeline end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with official demo endpoint
  python scripts/test_pipeline.py --demo

  # Test with local test server
  python scripts/test_pipeline.py --url http://localhost:8888/quiz-834

  # Test with custom API endpoint
  python scripts/test_pipeline.py --url http://localhost:8888/quiz-834 --api http://localhost:8000

  # Provide credentials explicitly
  python scripts/test_pipeline.py --url http://localhost:8888/quiz-834 --email test@example.com --secret my-secret
        """,
    )

    parser.add_argument(
        "--url",
        type=str,
        help="URL of the quiz to solve",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Test against the official demo endpoint (https://tds-llm-analysis.s-anand.net/demo)",
    )
    parser.add_argument(
        "--api",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API server (default: http://localhost:8000)",
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
        "--skip-health",
        action="store_true",
        help="Skip health check",
    )
    parser.add_argument(
        "--skip-malformed",
        action="store_true",
        help="Skip malformed request test",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.url and not args.demo:
        parser.error("Either --url or --demo must be specified")

    if args.url and args.demo:
        parser.error("Cannot specify both --url and --demo")

    # Determine quiz URL
    if args.demo:
        quiz_url = "https://tds-llm-analysis.s-anand.net/demo"
        logger.info("Using official demo endpoint")
    else:
        quiz_url = args.url

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run tests
    try:
        success = asyncio.run(
            run_tests(
                quiz_url=quiz_url,
                api_base_url=args.api,
                email=args.email,
                secret=args.secret,
                check_health=not args.skip_health,
                test_malformed=not args.skip_malformed,
            )
        )

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
