import asyncio
import json
import os
import sys
import subprocess
import argparse
import logging
import importlib.util
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from datetime import datetime

from anthropic import Anthropic
from dotenv import load_dotenv

from code_executor import CodeExecutor, ExecutionResult
from error_parser import extract_error_feedback


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H%M%S'
)
logger = logging.getLogger(__name__)


def setup_repository(repo: str, base_path: Optional[str] = None, instance_id: Optional[str] = None) -> str:
    """
    Clone repository into a unique instance folder.
    Each run gets its own fresh copy to avoid mixing results.
    """
    if base_path is None:
        script_dir = Path(__file__).parent.resolve()
        base_path = script_dir / "repos"
    else:
        base_path = Path(base_path).expanduser().resolve()
    
    if instance_id is None:
        instance_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    repo_name = repo.split('/')[-1]
    instance_path = base_path / repo_name / instance_id
    instance_path.mkdir(parents=True, exist_ok=True)
    
    repo_path = instance_path / repo_name

    logger.info(f"Cloning {repo} into new instance: {instance_id}...")
    github_url = f"https://github.com/{repo}.git"
    try:
        subprocess.run(["git", "clone", github_url, str(repo_path)], check=True, capture_output=True)
        logger.info(f"Cloned to {repo_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository: {e.stderr}")
        raise Exception(f"Failed to clone: {e.stderr}")

    try:
        subprocess.run(["git", "checkout", "main"], cwd=repo_path, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        try:
            subprocess.run(["git", "checkout", "master"], cwd=repo_path, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.warning("Could not checkout main or master branch")

    return str(repo_path)


class CodeGenAgent:
    """Agent that generates Python code using generated LIB functions from MCP tools."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 8000, 
                 lib_path: str = None, server_url: str = None, save_code_dir: Optional[str] = None):
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        if lib_path is None:
            raise ValueError("lib_path is required. LIB.py must be generated separately and provided.")
        self.lib_path = Path(lib_path).expanduser().resolve()
        if server_url is None:
            raise ValueError("server_url is required for code generation.")
        self.server_url = server_url
        self.lib_module = None
        self.tools = []
        if save_code_dir:
            self.save_code_dir = Path(save_code_dir).expanduser().resolve()
            self.save_code_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_code_dir = None

    def load_lib(self):
        """Load the existing LIB.py module."""
        if not self.lib_path.exists():
            raise FileNotFoundError(f"LIB file not found: {self.lib_path}. Please generate LIB.py first.")
        
        logger.info(f"Loading LIB from: {self.lib_path}")
        self._load_lib_module()

    def _load_lib_module(self):
        """Load the generated LIB module to inspect available functions."""
        if not self.lib_path or not Path(self.lib_path).exists():
            raise FileNotFoundError(f"LIB file not found: {self.lib_path}")
        
        spec = importlib.util.spec_from_file_location("LIB", self.lib_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load LIB from {self.lib_path}")
        
        self.lib_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.lib_module)
        
        self.tools = [
            name for name in dir(self.lib_module)
            if (callable(getattr(self.lib_module, name)) 
                and not name.startswith('_')
                and name not in ['tool_caller', 'hidden_fun_create_session_parts', 'hidden_fun_cleanup'])
        ]
        
        logger.info(f"Loaded LIB with {len(self.tools)} available functions")

    def _load_tool_documentation(self) -> str:
        """Load documentation from the generated markdown file."""
        doc_path = self.lib_path.parent / (self.lib_path.stem + '_doc.md')
        
        if not doc_path.exists():
            logger.warning(f"Documentation file not found: {doc_path}. Falling back to module inspection.")
            return self._generate_tool_wrappers_doc_fallback()
        
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                doc_content = f.read()
            
            header = """AVAILABLE TOOLS (use as LIB.function_name() with await):

All tools are async functions available in the LIB module.
Example: result = await LIB.find_jpa_entity(entity_name='User', projectPath='/path/to/project')

IMPORTANT: All LIB function calls must be prefixed with 'await' since they are async.

"""
            return header + doc_content
        except Exception as e:
            logger.warning(f"Failed to load documentation from {doc_path}: {e}. Falling back to module inspection.")
            return self._generate_tool_wrappers_doc_fallback()
    

    def _get_server_url_code(self) -> str:
        """Get the code snippet for initializing LIB session."""
        return f'await LIB.hidden_fun_create_session_parts("{self.server_url}")'

    async def generate_code(self, task: Dict, max_iterations: int = 10) -> str:
        """Generate Python code to solve the task using MCP tools."""
        repo = task["repo"]
        issue_url = task["issue_url"]
        issue_title = task["issue_title"]
        issue_body = task["issue_body"]
        repo_path = task["repo_path"]

        tool_docs = self._load_tool_documentation()
        server_url_code = self._get_server_url_code()

        system_prompt = f"""Generate COMPLETE, EXECUTABLE Python code to solve this Spring Boot task.

Repository: {repo} | Issue: {issue_url} | Title: {issue_title}
Description: {issue_body}
Project Path: {repo_path}

{tool_docs}

CODE STRUCTURE REQUIREMENTS:
- Import LIB: `import LIB`
- Wrap code in async function: `async def solve_task():`
- Initialize LIB session: {server_url_code}
- Call tools with await: `result = await LIB.function_name(...)`
- Cleanup: `await LIB.hidden_fun_cleanup()`
- Run with: `asyncio.run(solve_task())`

WORKFLOW: 1) EXPLORE (find entities/repos, read files) → 2) ANALYZE (extract patterns/package names) → 3) IMPLEMENT (use discovered values) → 4) VERIFY (build/check)

KEY RULES:
- ALL LIB calls must use 'await': `result = await LIB.function_name(...)`
- CHAIN CALLS: Store results, parse (may be JSON), extract values, use in next call
- READ FIRST: Always read files with `await LIB.get_file_text_by_path(...)` before modifying
- DISCOVER VALUES: Extract package names from file content, don't hardcode
- PARSE RESULTS: Tool outputs may be JSON strings (use json.loads) or dicts

EXAMPLE:
```python
import LIB
import json
import re
import asyncio

async def solve_task():
    project_path = "{repo_path}"
    
    # Initialize LIB session
    {server_url_code}
    
    try:
        # 1. Explore
        entities = await LIB.find_jpa_entities(projectPath=project_path)
        feature_file = await LIB.get_file_text_by_path(
            pathInProject="src/.../Feature.java", 
            projectPath=project_path
        )
        
        # 2. Extract package from file content
        package_match = re.search(r'package\\s+([\\w.]+)', feature_file)
        package_name = package_match.group(1) if package_match else "com.example"
        
        # 3. Use discovered value
        await LIB.create_jpa_entity(
            entityPackage=package_name, 
            projectPath=project_path, 
            ...
        )
        
        # 4. Read before modifying
        old_content = await LIB.get_file_text_by_path(
            pathInProject="...", 
            projectPath=project_path
        )
        new_content = modify_content(old_content)
        await LIB.replace_text_in_file(
            oldText=old_content, 
            newText=new_content, 
            ...
        )
        
        # 5. Verify
        await LIB.build_project(projectPath=project_path)
    finally:
        await LIB.hidden_fun_cleanup()

if __name__ == "__main__":
    asyncio.run(solve_task())
```

NO placeholders, NO "...", use discovered values. Return ONLY executable Python code with proper async/await structure."""

        messages = [{"role": "user", "content": system_prompt}]

        logger.info("Generating code...")
        
        for iteration in range(max_iterations):
            try:
                response = await asyncio.to_thread(
                    self.anthropic.messages.create,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=messages,
                )

                generated_code = ""
                for content in response.content:
                    if content.type == "text":
                        generated_code += content.text

                if "```python" in generated_code:
                    start = generated_code.find("```python") + 9
                    end = generated_code.find("```", start)
                    if end > start:
                        generated_code = generated_code[start:end].strip()
                elif "```" in generated_code:
                    start = generated_code.find("```") + 3
                    end = generated_code.find("```", start)
                    if end > start:
                        generated_code = generated_code[start:end].strip()

                stop_patterns = ["CODE_COMPLETE", "This is not code", "What's wrong", "What's needed"]
                for pattern in stop_patterns:
                    idx = generated_code.find(pattern)
                    if idx > 0:
                        generated_code = generated_code[:idx].strip()

                logger.info(f"Generated code (iteration {iteration + 1}, length: {len(generated_code)}):")
                logger.info(f"{generated_code[:500]}...")

                has_function_calls = any(
                    f"LIB.{tool}" in generated_code or f"await LIB.{tool}" in generated_code
                    for tool in self.tools[:10]
                )
                has_lib_import = "import LIB" in generated_code
                has_async_structure = "async def" in generated_code and "asyncio.run" in generated_code
                has_no_placeholders = "..." not in generated_code
                
                is_complete = (has_function_calls and has_lib_import and has_async_structure 
                              and has_no_placeholders and len(generated_code) > 200)
                
                if is_complete or iteration >= max_iterations - 1:
                    if not is_complete:
                        logger.warning("Code may be incomplete, but returning due to iteration limit")
                    return generated_code

                server_url_code = self._get_server_url_code()
                
                improvement_prompt = f"""The generated code needs improvement. Please generate COMPLETE, EXECUTABLE Python code following the workflow pattern.

Current code (may be incomplete):
```python
{generated_code[:1000]}
```

CRITICAL IMPROVEMENTS NEEDED:
1. **Use LIB module** - Import LIB and use `await LIB.function_name(...)` for all tool calls
2. **Async structure** - Wrap code in `async def solve_task():` and use `asyncio.run(solve_task())`
3. **Initialize LIB** - Call {server_url_code} at start
4. **Cleanup LIB** - Call `await LIB.hidden_fun_cleanup()` in finally block
5. **Follow exploration → analysis → implementation → verification workflow**
6. **Chain tool calls** - Use results from one call in the next (store in variables, parse results, use discovered values)
7. **Read files before modifying** - Always use `await LIB.get_file_text_by_path(...)` first, then `await LIB.replace_text_in_file(...)`
8. **Discover values** - Don't hardcode package names, entity names, etc. Discover them first
9. **NO placeholders** - All function calls must have complete parameters

EXAMPLE OF GOOD WORKFLOW (CHAINING RESULTS):
```python
import LIB
import json
import re
import asyncio

async def solve_task():
    project_path = "{repo_path}"
    
    # Initialize LIB session
    {server_url_code}
    
    try:
        # 1. Explore - discover what exists
        entities_result = await LIB.find_jpa_entities(projectPath=project_path)
        print(f"Step 1 - Found entities: {{entities_result}}")
        
        # 2. Parse result and extract information
        if isinstance(entities_result, str):
            try:
                entities_data = json.loads(entities_result)
            except:
                entities_data = entities_result
        else:
            entities_data = entities_result
        
        # 3. Use discovered information to read specific files
        feature_content = await LIB.get_file_text_by_path(
            pathInProject="src/main/java/com/sivalabs/ft/features/domain/entities/Feature.java",
            projectPath=project_path
        )
        print(f"Step 2 - Read Feature entity")
        
        # 4. Parse content to extract values
        package_match = re.search(r'package\\s+([\\w.]+)', feature_content)
        package_name = package_match.group(1) if package_match else None
        print(f"Step 3 - Extracted package: {{package_name}}")
        
        # 5. Use discovered values in next step
        if package_name:
            result = await LIB.create_jpa_entity(
                entityName="Tag",
                entityPackage=package_name,  # Using discovered value
                projectPath=project_path,
                # ... complete parameters
            )
            print(f"Step 4 - Created entity using package {{package_name}}")
    finally:
        await LIB.hidden_fun_cleanup()

if __name__ == "__main__":
    asyncio.run(solve_task())
```

Generate the complete code following this pattern:"""
                
                messages.append({"role": "assistant", "content": generated_code})
                messages.append({"role": "user", "content": improvement_prompt})

            except Exception as e:
                logger.error(f"Error generating code: {e}")
                import traceback
                traceback.print_exc()
                return ""

        return generated_code

    async def generate_code_with_error_feedback(self, task: Dict, error_feedback: Optional[str] = None, 
                                                max_iterations: int = 10) -> str:
        """Generate code with optional error feedback from previous execution."""
        repo = task["repo"]
        issue_url = task["issue_url"]
        issue_title = task["issue_title"]
        issue_body = task["issue_body"]
        repo_path = task["repo_path"]

        tool_docs = self._load_tool_documentation()
        server_url_code = self._get_server_url_code()

        if error_feedback:
            system_prompt = f"""The previously generated code failed to execute. Fix the errors and generate COMPLETE, EXECUTABLE Python code.

Repository: {repo} | Issue: {issue_url} | Title: {issue_title}
Description: {issue_body}
Project Path: {repo_path}

{tool_docs}

EXECUTION ERRORS FROM PREVIOUS ATTEMPT:
{error_feedback}

CRITICAL: Fix all errors from the execution output above. Common issues to check:
- Syntax errors (missing colons, parentheses, etc.)
- Import errors (missing imports, wrong module names)
- Name errors (undefined variables, typos)
- Type errors (wrong parameter types)
- Indentation errors
- Missing await keywords for async calls
- Incorrect function signatures

CODE STRUCTURE REQUIREMENTS:
- Import LIB: `import LIB`
- Wrap code in async function: `async def solve_task():`
- Initialize LIB session: {server_url_code}
- Call tools with await: `result = await LIB.function_name(...)`
- Cleanup: `await LIB.hidden_fun_cleanup()`
- Run with: `asyncio.run(solve_task())`

Generate the FIXED, COMPLETE, EXECUTABLE code now."""
        else:
            system_prompt = f"""Generate COMPLETE, EXECUTABLE Python code to solve this Spring Boot task.

Repository: {repo} | Issue: {issue_url} | Title: {issue_title}
Description: {issue_body}
Project Path: {repo_path}

{tool_docs}

CODE STRUCTURE REQUIREMENTS:
- Import LIB: `import LIB`
- Wrap code in async function: `async def solve_task():`
- Initialize LIB session: {server_url_code}
- Call tools with await: `result = await LIB.function_name(...)`
- Cleanup: `await LIB.hidden_fun_cleanup()`
- Run with: `asyncio.run(solve_task())`

WORKFLOW: 1) EXPLORE (find entities/repos, read files) → 2) ANALYZE (extract patterns/package names) → 3) IMPLEMENT (use discovered values) → 4) VERIFY (build/check)

KEY RULES:
- ALL LIB calls must use 'await': `result = await LIB.function_name(...)`
- CHAIN CALLS: Store results, parse (may be JSON), extract values, use in next call
- READ FIRST: Always read files with `await LIB.get_file_text_by_path(...)` before modifying
- DISCOVER VALUES: Extract package names from file content, don't hardcode
- PARSE RESULTS: Tool outputs may be JSON strings (use json.loads) or dicts

EXAMPLE:
```python
import LIB
import json
import re
import asyncio

async def solve_task():
    project_path = "{repo_path}"
    
    # Initialize LIB session
    {server_url_code}
    
    try:
        # 1. Explore
        entities = await LIB.find_jpa_entities(projectPath=project_path)
        feature_file = await LIB.get_file_text_by_path(
            pathInProject="src/.../Feature.java", 
            projectPath=project_path
        )
        
        # 2. Extract package from file content
        package_match = re.search(r'package\\s+([\\w.]+)', feature_file)
        package_name = package_match.group(1) if package_match else "com.example"
        
        # 3. Use discovered value
        await LIB.create_jpa_entity(
            entityPackage=package_name, 
            projectPath=project_path, 
            ...
        )
        
        # 4. Read before modifying
        old_content = await LIB.get_file_text_by_path(
            pathInProject="...", 
            projectPath=project_path
        )
        new_content = modify_content(old_content)
        await LIB.replace_text_in_file(
            oldText=old_content, 
            newText=new_content, 
            ...
        )
        
        # 5. Verify
        await LIB.build_project(projectPath=project_path)
    finally:
        await LIB.hidden_fun_cleanup()

if __name__ == "__main__":
    asyncio.run(solve_task())
```

NO placeholders, NO "...", use discovered values. Return ONLY executable Python code with proper async/await structure."""

        messages = [{"role": "user", "content": system_prompt}]

        logger.info("Generating code..." + (" (with error feedback)" if error_feedback else ""))
        
        for iteration in range(max_iterations):
            try:
                response = await asyncio.to_thread(
                    self.anthropic.messages.create,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=messages,
                )

                generated_code = ""
                for content in response.content:
                    if content.type == "text":
                        generated_code += content.text

                if "```python" in generated_code:
                    start = generated_code.find("```python") + 9
                    end = generated_code.find("```", start)
                    if end > start:
                        generated_code = generated_code[start:end].strip()
                elif "```" in generated_code:
                    start = generated_code.find("```") + 3
                    end = generated_code.find("```", start)
                    if end > start:
                        generated_code = generated_code[start:end].strip()

                stop_patterns = ["CODE_COMPLETE", "This is not code", "What's wrong", "What's needed"]
                for pattern in stop_patterns:
                    idx = generated_code.find(pattern)
                    if idx > 0:
                        generated_code = generated_code[:idx].strip()

                logger.info(f"Generated code (iteration {iteration + 1}, length: {len(generated_code)}):")
                logger.info(f"{generated_code[:500]}...")

                has_function_calls = any(
                    f"LIB.{tool}" in generated_code or f"await LIB.{tool}" in generated_code
                    for tool in self.tools[:10]
                )
                has_lib_import = "import LIB" in generated_code
                has_async_structure = "async def" in generated_code and "asyncio.run" in generated_code
                has_no_placeholders = "..." not in generated_code
                
                is_complete = (has_function_calls and has_lib_import and has_async_structure 
                              and has_no_placeholders and len(generated_code) > 200)
                
                if is_complete or iteration >= max_iterations - 1:
                    if not is_complete:
                        logger.warning("Code may be incomplete, but returning due to iteration limit")
                    return generated_code

                server_url_code = self._get_server_url_code()
                
                improvement_prompt = f"""The generated code needs improvement. Please generate COMPLETE, EXECUTABLE Python code following the workflow pattern.

Current code (may be incomplete):
```python
{generated_code[:1000]}
```

CRITICAL IMPROVEMENTS NEEDED:
1. **Use LIB module** - Import LIB and use `await LIB.function_name(...)` for all tool calls
2. **Async structure** - Wrap code in `async def solve_task():` and use `asyncio.run(solve_task())`
3. **Initialize LIB** - Call {server_url_code} at start
4. **Cleanup LIB** - Call `await LIB.hidden_fun_cleanup()` in finally block
5. **Follow exploration → analysis → implementation → verification workflow**
6. **Chain tool calls** - Use results from one call in the next (store in variables, parse results, use discovered values)
7. **Read files before modifying** - Always use `await LIB.get_file_text_by_path(...)` first, then `await LIB.replace_text_in_file(...)`
8. **Discover values** - Don't hardcode package names, entity names, etc. Discover them first
9. **NO placeholders** - All function calls must have complete parameters

Generate the complete code following this pattern:"""
                
                messages.append({"role": "assistant", "content": generated_code})
                messages.append({"role": "user", "content": improvement_prompt})

            except Exception as e:
                logger.error(f"Error generating code: {e}")
                import traceback
                traceback.print_exc()
                return ""

        return generated_code

    def _save_generated_code(self, code: str, iteration: int, instance_id: str = None):
        """Save generated code to file for later review."""
        if not self.save_code_dir or not code:
            return
        
        if instance_id is None:
            instance_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename = f"generated_code_iter_{iteration:02d}_{instance_id}.py"
        file_path = self.save_code_dir / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            logger.info(f"Saved generated code to: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to save generated code: {e}")

    async def run_task(self, task: Dict, execute: bool = False, 
                      max_exec_iterations: int = 5, instance_id: str = None) -> Tuple[str, List[ExecutionResult]]:
        """
        Generate code for the task and optionally execute it with iterative improvement.
        
        Args:
            task: Task configuration dictionary
            execute: Whether to execute the code and iteratively improve
            max_exec_iterations: Maximum number of execution attempts
            instance_id: Instance ID for saving code files
        
        Returns:
            Tuple of (final_code, execution_history)
        """
        self.load_lib()
        
        execution_history = []
        
        if not execute:
            code = await self.generate_code_with_error_feedback(task)
            if code:
                self._save_generated_code(code, iteration=0, instance_id=instance_id)
            return code, execution_history
        
        logger.info("="*60)
        logger.info("Starting iterative code generation and execution")
        logger.info("="*60)
        
        code = None
        for exec_iteration in range(max_exec_iterations):
            logger.info(f"\n--- Execution Iteration {exec_iteration + 1}/{max_exec_iterations} ---")
            
            error_feedback = None
            if exec_iteration > 0 and execution_history:
                last_result = execution_history[-1]
                error_feedback = extract_error_feedback(last_result)
                if error_feedback:
                    logger.info("Regenerating code with error feedback...")
                else:
                    logger.info("Previous execution succeeded, no regeneration needed")
                    break
            
            code = await self.generate_code_with_error_feedback(
                task, 
                error_feedback=error_feedback,
                max_iterations=10
            )
            
            if not code:
                logger.error("Failed to generate code")
                break
            
            self._save_generated_code(code, iteration=exec_iteration + 1, instance_id=instance_id)
            
            logger.info("Executing generated code...")
            executor = CodeExecutor(
                lib_path=str(self.lib_path),
                server_url=self.server_url,
                project_path=task.get("repo_path")
            )
            
            result = executor.execute(code, timeout=300)
            execution_history.append(result)
            
            logger.info(f"Execution result: {'SUCCESS' if result.success else 'FAILED'}")
            if result.success:
                logger.info(f"Output:\n{result.stdout}")
                logger.info("="*60)
                logger.info("Code executed successfully!")
                logger.info("="*60)
                break
            else:
                logger.warning(f"Execution failed (exit code: {result.exit_code})")
                logger.warning(f"Error output:\n{result.error_output}")
                if exec_iteration < max_exec_iterations - 1:
                    logger.info("Will regenerate code with error feedback...")
                else:
                    logger.warning("Max execution iterations reached")
        
        return code, execution_history


async def main():
    parser = argparse.ArgumentParser(
        description='Code Generation Agent for DPAIA tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--repo', type=str, help='Repository in format owner/repo')
    parser.add_argument('--issue-url', type=str, help='GitHub issue URL')
    parser.add_argument('--issue-title', type=str, help='Issue title')
    parser.add_argument('--issue-body', type=str, help='Issue description')
    parser.add_argument('--lib-path', type=str, required=True, 
                       help='Path to generated LIB.py (must be generated separately)')
    parser.add_argument('--mcp-url', type=str, required=True,
                       help='MCP server URL (required for code generation)')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-20250514', 
                       help='Claude model to use')
    parser.add_argument('--max-tokens', type=int, default=8000, help='Max tokens per request')
    parser.add_argument('--instance-id', type=str, help='Custom instance ID (default: timestamp)')
    parser.add_argument('--output', type=str, help='Output file for generated code')
    parser.add_argument('--execute', action='store_true',
                       help='Execute generated code and iteratively improve based on errors')
    parser.add_argument('--max-exec-iterations', type=int, default=5,
                       help='Maximum number of execution attempts (default: 5)')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    script_dir = Path(__file__).parent.resolve()
    
    instance_id = args.instance_id or datetime.now().strftime('%Y%m%d_%H%M%S')

    logs_dir = script_dir / "results_codegen"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"codegen_agent_{instance_id}.txt"
    
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)
    
    log_file = open(log_path, "a", encoding="utf-8")
    orig_stdout = sys.stdout

    class TeeLogger:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = TeeLogger(orig_stdout, log_file)
    
    if args.config:
        if not Path(args.config).exists():
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
        with open(args.config, 'r') as f:
            TASK_CONFIG = json.load(f)
    else:
        TASK_CONFIG = {
            "repo": "dpaia/spring-petclinic-rest",
            "issue_url": "https://github.com/dpaia/spring-petclinic-rest/issues/5",
            "issue_title": "Detect and resolve N+1 select problem",
            "issue_body": """
Analyze the Owner-Pet and Pet-Visit relationships in the Spring PetClinic REST application.
Detect N+1 SELECT queries that occur when fetching owners with their pets and visits.
Resolve these issues by adding appropriate JPA annotations like @EntityGraph or using fetch joins.
Test the fix to ensure all data is loaded correctly with minimal queries.
            """,
            "base_path": None
        }

    if args.repo:
        TASK_CONFIG["repo"] = args.repo
    if args.issue_url:
        TASK_CONFIG["issue_url"] = args.issue_url
    if args.issue_title:
        TASK_CONFIG["issue_title"] = args.issue_title
    if args.issue_body:
        TASK_CONFIG["issue_body"] = args.issue_body
    
    logger.info(f"Creating new repository instance: {instance_id}")
    
    repo_path = setup_repository(TASK_CONFIG["repo"], TASK_CONFIG.get("base_path"), instance_id=instance_id)
    TASK_CONFIG["repo_path"] = repo_path

    logger.info(f"Repository ready at: {repo_path}")

    lib_path = args.lib_path
    mcp_url = args.mcp_url

    generated_code_dir = script_dir / "generated_code_iterations"
    generated_code_dir.mkdir(exist_ok=True)

    agent = CodeGenAgent(
        model=args.model,
        max_tokens=args.max_tokens,
        lib_path=lib_path,
        server_url=mcp_url,
        save_code_dir=str(generated_code_dir)
    )

    try:
        logger.info("="*60)
        logger.info(f"TASK: {TASK_CONFIG['issue_title']}")
        logger.info(f"REPO: {TASK_CONFIG['repo']}")
        logger.info(f"PATH: {TASK_CONFIG['repo_path']}")
        logger.info(f"LIB: {lib_path}")
        logger.info(f"MCP URL: {mcp_url}")
        logger.info("="*60)

        code, execution_history = await agent.run_task(
            TASK_CONFIG,
            execute=args.execute,
            max_exec_iterations=args.max_exec_iterations,
            instance_id=instance_id
        )

        if code:
            logger.info("="*60)
            logger.info("Final generated code:")
            logger.info("="*60)
            print("\n" + "="*60)
            print("GENERATED CODE:")
            print("="*60)
            print(code)
            print("="*60 + "\n")

            if args.execute and execution_history:
                logger.info("="*60)
                logger.info("Execution History:")
                logger.info("="*60)
                for i, result in enumerate(execution_history, 1):
                    status = "SUCCESS" if result.success else "FAILED"
                    logger.info(f"Attempt {i}: {status} (exit code: {result.exit_code})")
                    if result.success:
                        logger.info(f"Output:\n{result.stdout}")
                    else:
                        logger.warning(f"Error:\n{result.error_output}")
                logger.info("="*60)

            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(code)
                logger.info(f"Code saved to: {output_path}")
        else:
            logger.warning("No code generated")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
    finally:
        log_file.close()
        sys.stdout = orig_stdout
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        logger.info(f"Log saved to: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())

