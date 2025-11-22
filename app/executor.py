from __future__ import annotations

import asyncio
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(slots=True)
class ExecutionResult:
    stdout: str
    stderr: str
    success: bool
    return_code: int
    error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "success": self.success,
            "return_code": self.return_code,
            "error": self.error,
        }


class AsyncSubprocessExecutor:
    """
    Execute Python code in an isolated subprocess (python -I) with a timeout.
    """

    def __init__(
        self,
        *,
        python_executable: str = sys.executable,
        timeout_seconds: float = 5.0,
        extra_args: Optional[list[str]] = None,
    ) -> None:
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds
        self.extra_args = extra_args or ["-I"]

    async def execute(self, code: str) -> ExecutionResult:
        """
        Write `code` to a temp file and run it via the configured interpreter.
        Captures stdout/stderr verbatim without any JSON wrapping.
        """
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(code)

        try:
            cmd = [self.python_executable, *self.extra_args, str(tmp_path)]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return ExecutionResult(
                    stdout="",
                    stderr="",
                    success=False,
                    return_code=-1,
                    error=f"Execution timed out after {self.timeout_seconds} seconds.",
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            success = proc.returncode == 0
            error_message = None if success else f"Return code {proc.returncode}"

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                success=success,
                return_code=proc.returncode,
                error=error_message,
            )
        finally:
            tmp_path.unlink(missing_ok=True)


async def execute_python(code: str, *, timeout_seconds: float = 5.0) -> ExecutionResult:
    executor = AsyncSubprocessExecutor(timeout_seconds=timeout_seconds)
    return await executor.execute(code)


def run_trusted_callable(func, *args, **kwargs):
    """
    Execute a trusted Python callable directly without sandboxing.

    Returns an ExecutionResult so callers can handle trusted runs the same
    way they handle sandboxed runs.
    """
    try:
        output = func(*args, **kwargs)
        stdout = "" if output is None else str(output)
        return ExecutionResult(
            stdout=stdout,
            stderr="",
            success=True,
            return_code=0,
            error=None,
        )
    except Exception as exc:
        return ExecutionResult(
            stdout="",
            stderr=str(exc),
            success=False,
            return_code=-1,
            error=f"{exc.__class__.__name__}: {exc}",
        )


__all__ = [
    "AsyncSubprocessExecutor",
    "ExecutionResult",
    "execute_python",
    "run_trusted_callable",
]
