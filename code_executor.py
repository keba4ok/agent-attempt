"""
Code executor for generated Python code.
Executes code in a sandbox environment with LIB.py support.
"""
import os
import shutil
import subprocess
import tempfile
import asyncio
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    error_output: str


class CodeExecutor:
    """Executes generated Python code in a sandbox environment."""
    
    def __init__(self, lib_path: str, server_url: str, project_path: Optional[str] = None):
        """
        Initialize code executor.
        
        Args:
            lib_path: Path to LIB.py file
            server_url: MCP server URL for LIB initialization
            project_path: Optional project path to set as working directory
        """
        self.lib_path = Path(lib_path).expanduser().resolve()
        self.server_url = server_url
        self.project_path = Path(project_path).expanduser().resolve() if project_path else None
        
        if not self.lib_path.exists():
            raise FileNotFoundError(f"LIB file not found: {self.lib_path}")
    
    def execute(self, code: str, timeout: int = 300) -> ExecutionResult:
        """
        Execute generated Python code in a sandbox.
        
        Args:
            code: Python code to execute (should be complete with async def solve_task(), etc.)
            timeout: Execution timeout in seconds
        
        Returns:
            ExecutionResult with success status, output, and errors
        """
        sandbox_dir = Path(tempfile.mkdtemp(prefix="code_executor_"))
        
        try:
            sandbox_lib = sandbox_dir / "LIB.py"
            shutil.copyfile(self.lib_path, sandbox_lib)
            
            code_file = sandbox_dir / "generated_code.py"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            cwd = str(self.project_path) if self.project_path else str(sandbox_dir)
            
            try:
                process = subprocess.run(
                    ["python", str(code_file)],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    encoding='utf-8',
                    errors='replace'
                )
                
                success = process.returncode == 0
                stdout = process.stdout or ""
                stderr = process.stderr or ""
                
                error_indicators = [
                    "Error occurred:",
                    "Traceback (most recent call last):",
                    "Exception:",
                    "Error:",
                    "missing",
                    "required",
                    "TypeError:",
                    "AttributeError:",
                    "NameError:",
                    "SyntaxError:",
                    "ImportError:",
                    "ModuleNotFoundError:",
                    "Could not find",
                    "Could not determine",
                    "Could not find Feature entity file",
                    "Could not find",
                ]
                
                failure_indicators = [
                    "Could not find",
                    "Could not determine",
                    "Could not find Feature entity file",
                ]
                
                has_error_in_output = any(indicator in stdout or indicator in stderr 
                                         for indicator in error_indicators)
                
                completion_indicators = [
                    "Tag entity created",
                    "Migration script created",
                    "IMPLEMENTATION COMPLETED",
                    "SUCCESSFULLY",
                ]
                
                has_completion = any(indicator in stdout for indicator in completion_indicators)
                
                # If exit code is 0 but no completion indicators and has failure indicators, mark as failed
                if success and not has_completion:
                    # Check if there are failure indicators suggesting early return
                    has_failure = any(indicator in stdout for indicator in failure_indicators)
                    if has_failure:
                        has_error_in_output = True
                
                if not success or has_error_in_output:
                    error_output = f"Exit code: {process.returncode}\n\n"
                    if stderr:
                        error_output += f"STDERR:\n{stderr}\n\n"
                    if stdout:
                        error_output += f"STDOUT:\n{stdout}"
                    else:
                        error_output += stderr
                    success = False
                else:
                    error_output = ""
                
                return ExecutionResult(
                    success=success,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=process.returncode,
                    error_output=error_output
                )
                
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="",
                    exit_code=-1,
                    error_output=f"Execution timed out after {timeout} seconds"
                )
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="",
                    exit_code=-1,
                    error_output=f"Execution failed with exception: {str(e)}"
                )
        
        finally:
            try:
                shutil.rmtree(sandbox_dir, ignore_errors=True)
            except Exception:
                pass

