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
from dataclasses import dataclass, field


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    error_output: str
    tool_calls_count: int = 0
    tool_calls_stats: Dict[str, int] = field(default_factory=dict)
    tool_calls_log_path: Optional[str] = None


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
    
    def execute(self, code: str, timeout: int = 300, instance_id: str = None, phase: str = None) -> ExecutionResult:
        """
        Execute generated Python code in a sandbox.
        
        Args:
            code: Python code to execute (should be complete with async def solve_task(), etc.)
            timeout: Execution timeout in seconds
            instance_id: Optional instance ID for organizing log files
            phase: Optional phase name for organizing log files
        
        Returns:
            ExecutionResult with success status, output, errors, and tool call count
        """
        sandbox_dir = Path(tempfile.mkdtemp(prefix="code_executor_"))
        
        if instance_id and phase:
            if self.project_path:
                log_dir = Path(self.project_path).parent.parent / "tool_calls_logs"
            else:
                log_dir = Path.cwd() / "tool_calls_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            tool_calls_log_path = log_dir / f"{instance_id}_{phase}_tool_calls.log"
        else:
            tool_calls_log_path = sandbox_dir / "tool_calls.log"
        
        tool_calls_log_path = tool_calls_log_path.resolve()
        
        try:
            sandbox_lib = sandbox_dir / "LIB.py"
            shutil.copyfile(self.lib_path, sandbox_lib)
            
            log_setup_line = f'LIB.hidden_setup_logs("{tool_calls_log_path}")'
            
            lines = code.split('\n')
            solve_task_found = False
            
            for i, line in enumerate(lines):
                if line.strip().startswith("async def solve_task"):
                    for j in range(i + 1, min(i + 15, len(lines))):
                        if "hidden_fun_create_session_parts" in lines[j]:
                            indent = len(lines[j]) - len(lines[j].lstrip())
                            log_setup_indented = " " * indent + log_setup_line
                            lines.insert(j + 1, log_setup_indented)
                            solve_task_found = True
                            break
                    if solve_task_found:
                        break
            
            if not solve_task_found:
                lib_import_found = False
                for i, line in enumerate(lines):
                    if line.strip() == "import LIB" or line.strip().startswith("import LIB"):
                        lib_import_found = True
                        lines.insert(i + 1, log_setup_line)
                        break
                
                if not lib_import_found:
                    lines.insert(0, "import LIB")
                    lines.insert(1, log_setup_line)
            
            code = '\n'.join(lines)
            
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
                
                if success and not has_completion:
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
                
                tool_calls_count = 0
                tool_calls_stats = {}
                if tool_calls_log_path.exists():
                    try:
                        with open(tool_calls_log_path, 'r', encoding='utf-8') as f:
                            log_content = f.read()
                            for line in log_content.split('\n'):
                                if line.strip().startswith('Called:'):
                                    tool_calls_count += 1
                                    try:
                                        tool_name = line.split('Called:')[1].split(',')[0].strip()
                                        tool_calls_stats[tool_name] = tool_calls_stats.get(tool_name, 0) + 1
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                
                return ExecutionResult(
                    success=success,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=process.returncode,
                    error_output=error_output,
                    tool_calls_count=tool_calls_count,
                    tool_calls_stats=tool_calls_stats,
                    tool_calls_log_path=str(tool_calls_log_path) if tool_calls_log_path.exists() else None
                )
                
            except subprocess.TimeoutExpired:
                tool_calls_count = 0
                tool_calls_stats = {}
                if tool_calls_log_path.exists():
                    try:
                        with open(tool_calls_log_path, 'r', encoding='utf-8') as f:
                            log_content = f.read()
                            for line in log_content.split('\n'):
                                if line.strip().startswith('Called:'):
                                    tool_calls_count += 1
                                    try:
                                        tool_name = line.split('Called:')[1].split(',')[0].strip()
                                        tool_calls_stats[tool_name] = tool_calls_stats.get(tool_name, 0) + 1
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="",
                    exit_code=-1,
                    error_output=f"Execution timed out after {timeout} seconds",
                    tool_calls_count=tool_calls_count,
                    tool_calls_stats=tool_calls_stats,
                    tool_calls_log_path=str(tool_calls_log_path) if tool_calls_log_path.exists() else None
                )
            except Exception as e:
                tool_calls_count = 0
                tool_calls_stats = {}
                if tool_calls_log_path.exists():
                    try:
                        with open(tool_calls_log_path, 'r', encoding='utf-8') as f:
                            log_content = f.read()
                            for line in log_content.split('\n'):
                                if line.strip().startswith('Called:'):
                                    tool_calls_count += 1
                                    try:
                                        tool_name = line.split('Called:')[1].split(',')[0].strip()
                                        tool_calls_stats[tool_name] = tool_calls_stats.get(tool_name, 0) + 1
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="",
                    exit_code=-1,
                    error_output=f"Execution failed with exception: {str(e)}",
                    tool_calls_count=tool_calls_count,
                    tool_calls_stats=tool_calls_stats,
                    tool_calls_log_path=str(tool_calls_log_path) if tool_calls_log_path.exists() else None
                )
        
        finally:
            try:
                shutil.rmtree(sandbox_dir, ignore_errors=True)
            except Exception:
                pass

