"""
Simple error parser - extracts all error output for LLM feedback.
"""
from typing import Optional
from code_executor import ExecutionResult


def extract_error_feedback(execution_result: ExecutionResult) -> Optional[str]:
    """
    Extract error feedback from execution result.
    Very simple: just return all error output if execution failed.
    
    Args:
        execution_result: Result from code execution
    
    Returns:
        Error feedback string if execution failed, None if successful
    """
    if execution_result.success:
        return None
    
    feedback = f"Code execution failed with exit code {execution_result.exit_code}.\n\n"
    
    if execution_result.error_output:
        feedback += execution_result.error_output
    elif execution_result.stderr:
        feedback += f"STDERR:\n{execution_result.stderr}\n"
    elif execution_result.stdout:
        feedback += f"STDOUT:\n{execution_result.stdout}\n"
    
    return feedback

