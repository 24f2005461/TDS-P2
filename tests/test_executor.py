import sys

import pytest

from app.executor import AsyncSubprocessExecutor, execute_python, run_trusted_callable


@pytest.mark.anyio
async def test_execute_python_success():
    code = """
print("hello")
value = 21 * 2
"""
    result = await execute_python(code)

    assert result.success is True
    assert "hello" in result.stdout
    assert result.stderr == ""
    assert result.return_code == 0
    assert result.error is None


@pytest.mark.anyio
async def test_executor_exposes_nonzero_return_code():
    code = """
raise ValueError("boom")
"""
    executor = AsyncSubprocessExecutor(
        python_executable=sys.executable, timeout_seconds=2.0
    )
    result = await executor.execute(code)

    assert result.success is False
    assert "ValueError" in result.stderr
    assert result.return_code != 0
    assert result.error == f"Return code {result.return_code}"


@pytest.mark.anyio
async def test_executor_enforces_timeout():
    code = """
import time
time.sleep(3)
"""
    executor = AsyncSubprocessExecutor(
        python_executable=sys.executable, timeout_seconds=0.2
    )
    result = await executor.execute(code)

    assert result.success is False
    assert result.return_code == -1
    assert "timed out" in (result.error or "").lower()


@pytest.mark.anyio
async def test_run_trusted_callable_success():
    def add(a, b):
        return a + b

    result = run_trusted_callable(add, 2, 3)

    assert result.success is True
    assert result.return_code == 0
    assert result.stdout == "5"
    assert result.error is None


@pytest.mark.anyio
async def test_run_trusted_callable_failure():
    def explode():
        raise RuntimeError("boom")

    result = run_trusted_callable(explode)

    assert result.success is False
    assert result.return_code == -1
    assert "RuntimeError" in (result.error or "")
    assert "boom" in result.stderr
