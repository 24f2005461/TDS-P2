#!/usr/bin/env python3
"""
Environment validation script for TDS Quiz Solver.

This script checks that all required environment variables are properly set
and that dependencies are installed correctly.

Usage:
    python scripts/check_env.py
    python scripts/check_env.py --fix  # Attempt to install missing dependencies
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}\n")


def print_section(text: str):
    """Print formatted section."""
    print(f"\n{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{'-' * 70}")


def check_pass(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓{Colors.RESET} {message}")


def check_warn(message: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {message}")


def check_fail(message: str):
    """Print error message."""
    print(f"{Colors.RED}✗{Colors.RESET} {message}")


def check_dotenv() -> bool:
    """Check if python-dotenv is installed."""
    if DOTENV_AVAILABLE:
        check_pass("python-dotenv is installed")
        return True
    else:
        check_warn("python-dotenv is not installed (optional but recommended)")
        print("  Install with: pip install python-dotenv")
        return False


def check_env_file() -> bool:
    """Check if .env file exists."""
    env_path = Path(".env")
    if env_path.exists():
        check_pass(".env file exists")
        return True
    else:
        check_fail(".env file not found")
        print("  Create .env file from .env.example:")
        print("    cp .env.example .env")
        print("  Then edit .env and fill in your credentials")
        return False


def check_required_env_vars() -> tuple[bool, list[str]]:
    """Check required environment variables."""
    required_vars = {
        "EMAIL": "Your email address for authentication",
        "SECRET": "Your secret key for authentication",
        "AIPIPE_API_KEY": "API key for AIPipe gateway",
    }

    missing = []
    all_present = True

    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if var in ("SECRET", "AIPIPE_API_KEY"):
                masked = (
                    value[:4] + "*" * (len(value) - 8) + value[-4:]
                    if len(value) > 8
                    else "***"
                )
                check_pass(f"{var} is set ({masked})")
            else:
                check_pass(f"{var} is set ({value})")
        else:
            check_fail(f"{var} is not set - {description}")
            missing.append(var)
            all_present = False

    return all_present, missing


def check_optional_env_vars():
    """Check optional environment variables."""
    optional_vars = {
        "LOG_LEVEL": ("INFO", "Logging level (DEBUG, INFO, WARNING, ERROR)"),
        "PLAYWRIGHT_HEADLESS": ("true", "Run browser in headless mode"),
        "PLAYWRIGHT_NAVIGATION_TIMEOUT_SECONDS": ("30", "Navigation timeout"),
        "LLM_DEFAULT_MODEL": ("openai/gpt-4o-mini", "Default LLM model"),
    }

    for var, (default, description) in optional_vars.items():
        value = os.getenv(var)
        if value:
            check_pass(f"{var} = {value}")
        else:
            check_warn(f"{var} not set (using default: {default})")
            print(f"  {description}")


def check_python_version() -> bool:
    """Check Python version."""
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 11:
        check_pass(f"Python {major}.{minor} (>= 3.11)")
        return True
    else:
        check_fail(f"Python {major}.{minor} (requires >= 3.11)")
        return False


def check_dependency(package: str, import_name: str | None = None) -> bool:
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package

    try:
        __import__(import_name)
        check_pass(f"{package} is installed")
        return True
    except ImportError:
        check_fail(f"{package} is not installed")
        print(f"  Install with: pip install {package}")
        return False


def check_playwright_browsers() -> bool:
    """Check if Playwright browsers are installed."""
    try:
        result = subprocess.run(
            ["playwright", "install", "--dry-run", "chromium"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "is already installed" in result.stdout or result.returncode == 0:
            check_pass("Playwright chromium browser is installed")
            return True
        else:
            check_fail("Playwright chromium browser is not installed")
            print("  Install with: playwright install chromium")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        check_warn("Could not verify Playwright browser installation")
        print("  Install with: playwright install chromium")
        return False


def check_dependencies() -> bool:
    """Check all required dependencies."""
    required_deps = [
        ("httpx", None),
        ("fastapi", None),
        ("uvicorn", None),
        ("playwright", None),
        ("pandas", None),
        ("pillow", "PIL"),
        ("pydantic", None),
        ("pytest", None),
    ]

    all_installed = True
    for package, import_name in required_deps:
        if not check_dependency(package, import_name):
            all_installed = False

    return all_installed


def install_dependencies() -> bool:
    """Attempt to install missing dependencies."""
    print("\nAttempting to install dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
        )
        check_pass("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        check_fail("Failed to install dependencies")
        return False


def install_playwright_browsers() -> bool:
    """Attempt to install Playwright browsers."""
    print("\nAttempting to install Playwright browsers...")
    try:
        subprocess.run(
            ["playwright", "install", "chromium"],
            check=True,
        )
        check_pass("Playwright browsers installed successfully")
        return True
    except subprocess.CalledProcessError:
        check_fail("Failed to install Playwright browsers")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate TDS Quiz Solver environment setup"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to install missing dependencies",
    )

    args = parser.parse_args()

    print_header("TDS Quiz Solver - Environment Check")

    # Check Python version
    print_section("Python Version")
    python_ok = check_python_version()

    # Check dotenv
    print_section("Python Dotenv")
    dotenv_ok = check_dotenv()

    # Check .env file
    print_section("Environment File")
    env_file_ok = check_env_file()

    # Check required environment variables
    print_section("Required Environment Variables")
    required_vars_ok, missing_vars = check_required_env_vars()

    # Check optional environment variables
    print_section("Optional Environment Variables")
    check_optional_env_vars()

    # Check dependencies
    print_section("Python Dependencies")
    deps_ok = check_dependencies()

    # Check Playwright browsers
    print_section("Playwright Browsers")
    browsers_ok = check_playwright_browsers()

    # Summary
    print_section("Summary")

    issues = []
    if not python_ok:
        issues.append("Python version too old (need >= 3.11)")
    if not env_file_ok:
        issues.append(".env file missing")
    if not required_vars_ok:
        issues.append(f"Missing environment variables: {', '.join(missing_vars)}")
    if not deps_ok:
        issues.append("Some dependencies are not installed")
    if not browsers_ok:
        issues.append("Playwright browsers not installed")

    if not issues:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All checks passed!{Colors.RESET}")
        print("\nYou're ready to run the TDS Quiz Solver!")
        print("\nNext steps:")
        print("  1. Start the API: uvicorn app.main:app --reload --port 8000")
        print("  2. Test with demo: python scripts/test_demo_endpoint.py")
        return 0
    else:
        print(
            f"\n{Colors.RED}{Colors.BOLD}✗ Found {len(issues)} issue(s):{Colors.RESET}"
        )
        for issue in issues:
            print(f"  • {issue}")

        if args.fix:
            print_section("Attempting to Fix Issues")

            if not deps_ok:
                install_dependencies()

            if not browsers_ok:
                install_playwright_browsers()

            if not env_file_ok:
                print("\nCreating .env file from template...")
                try:
                    if Path(".env.example").exists():
                        import shutil

                        shutil.copy(".env.example", ".env")
                        check_pass("Created .env file from .env.example")
                        print(
                            f"{Colors.YELLOW}  ⚠ Edit .env and fill in your credentials{Colors.RESET}"
                        )
                    else:
                        check_fail(".env.example not found")
                except Exception as e:
                    check_fail(f"Failed to create .env: {e}")

            print("\n" + "=" * 70)
            print("Re-run this script to verify fixes:")
            print("  python scripts/check_env.py")
        else:
            print("\nRun with --fix to attempt automatic fixes:")
            print("  python scripts/check_env.py --fix")

        return 1


if __name__ == "__main__":
    sys.exit(main())
