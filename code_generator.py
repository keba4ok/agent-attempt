import asyncio
import logging
from typing import Optional, Dict, List
from pathlib import Path

from anthropic import Anthropic
import os

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Handles LLM-based code generation."""
    
    def __init__(self, anthropic: Anthropic, model: str, max_tokens: int, 
                 lib_loader, phase_manager, server_url: str):
        self.anthropic = anthropic
        self.model = model
        self.max_tokens = max_tokens
        self.lib_loader = lib_loader
        self.phase_manager = phase_manager
        self.server_url = server_url
    
    def _get_server_url_code(self) -> str:
        return f'await LIB.hidden_fun_create_session_parts("{self.server_url}")'
    
    async def generate_phase_code(self, phase: str, task: Dict, error_feedback: Optional[str] = None,
                                  instance_id: str = None, max_iterations: int = 10) -> str:
        """Generate code for a specific phase."""
        repo = task["repo"]
        issue_url = task["issue_url"]
        issue_title = task["issue_title"]
        issue_body = task["issue_body"]
        repo_path = task["repo_path"]
        
        tool_docs = self.lib_loader.get_tool_documentation()
        server_url_code = self._get_server_url_code()
        project_path = task.get("repo_path", "")
        previous_context = self.phase_manager.get_previous_phases_context(phase, instance_id, project_path) if instance_id else ""
        
        phase_descriptions = {
            "exploration": """EXPLORATION: Find entities, repositories, packages. Save to `phase_results/exploration_result.json`.
Use: `LIB.get_project_modules()`, `LIB.find_files_by_name_keyword()`, `LIB.find_jpa_entity()`, `LIB.get_file_text_by_path()`
CRITICAL: Keep code SHORT (<150 lines). Do only essential discovery, no verbose helpers.""",
            
            "analysis": """ANALYSIS: Load exploration results, analyze files, extract patterns/package names. Save to `phase_results/analysis_result.json`.
Use: `LIB.get_jpa_entity_by_name()`, `LIB.find_spring_data_repositories()`, `LIB.get_file_text_by_path()`
CRITICAL: Keep code SHORT (<150 lines). Focus on extracting key info only.""",
            
            "implementation": """IMPLEMENTATION: Load previous phase results, create entities/repositories/migrations using discovered values.
Use: `LIB.create_jpa_entity()`, `LIB.create_spring_data_repository()`, `LIB.create_or_update_entity_attribute()`, `LIB.create_jpa_query()`, `LIB.generate_db_migration()`
CRITICAL: Keep code SHORT (<200 lines). Create only what's needed, minimal helpers.""",
            
            "verification": """VERIFICATION: Build project, check for errors, generate tests if needed.
Use: `LIB.build_project(rebuild=False, timeout=120000)`, `LIB.get_file_problems()`, `LIB.generate_integration_tests()`
CRITICAL: Keep code SHORT (<100 lines). Just build and check."""
        }
        
        phase_instructions = phase_descriptions.get(phase, "")

        project_path = task.get("repo_path", "")
        phase_results_dir = Path(project_path) / "phase_results" if project_path else None
        if phase_results_dir:
            phase_results_dir.mkdir(exist_ok=True)
        
        phase_results_path = str(phase_results_dir) if phase_results_dir else "phase_results"
        
        if phase in ["exploration", "analysis"]:
            phase_instructions += f"\nSave results to: {phase_results_path}/{phase}_result.json"
        elif phase in ["implementation", "verification"]:
            phase_instructions += f"\nLoad previous results from: {phase_results_path}/"
        
        phase_examples = {
            "exploration": f"""EXAMPLE EXPLORATION PHASE CODE (SHORT - ~80 lines):
```python
import LIB
import json
import re
import asyncio
from pathlib import Path

async def solve_task():
    project_path = "{repo_path}"
    {server_url_code}
    
    try:
        # Get modules
        modules = await LIB.get_project_modules(projectPath=project_path)
        
        # Find entities
        entity_files = await LIB.find_files_by_name_keyword(
            nameKeyword="Entity", fileCountLimit=20, timeout=30, projectPath=project_path
        )
        
        # Find repositories
        repo_files = await LIB.find_files_by_name_keyword(
            nameKeyword="Repository", fileCountLimit=20, timeout=30, projectPath=project_path
        )
        
        # Extract package from first entity
        package_names = set()
        if entity_files.get("files"):
            content = await LIB.get_file_text_by_path(
                pathInProject=entity_files["files"][0],
                truncateMode="NONE", maxLinesCount=50, projectPath=project_path
            )
            match = re.search(r'package\\s+([\\w.]+);', content)
            if match:
                pkg = match.group(1)
                package_names.add(pkg)
                package_names.add(pkg.rsplit('.', 1)[0])
        
        # Save results
        results_dir = Path(project_path) / "phase_results"
        results_dir.mkdir(exist_ok=True)
        with open(results_dir / "exploration_result.json", 'w') as f:
            json.dump({{
                "modules": modules,
                "entities": entity_files.get("files", []),
                "repositories": repo_files.get("files", []),
                "package_names": list(package_names),
                "base_package": list(package_names)[0].rsplit('.', 2)[0] if package_names else None
            }}, f, indent=2)
        
    finally:
        await LIB.hidden_fun_cleanup()

if __name__ == "__main__":
    asyncio.run(solve_task())
```""",
            
            "analysis": f"""EXAMPLE ANALYSIS PHASE CODE (SHORT - ~60 lines):
```python
import LIB
import json
import re
import asyncio
from pathlib import Path

async def solve_task():
    project_path = "{repo_path}"
    {server_url_code}
    
    try:
        # Load exploration
        with open(Path(project_path) / "phase_results" / "exploration_result.json", 'r') as f:
            exploration = json.load(f)
        
        # Analyze first entity
        entity_file = exploration.get("entities", [None])[0]
        if entity_file:
            content = await LIB.get_file_text_by_path(
                pathInProject=entity_file, truncateMode="NONE", maxLinesCount=100, projectPath=project_path
            )
            package_match = re.search(r'package\\s+([\\w.]+);', content)
            package = package_match.group(1) if package_match else None
            
            entity_match = re.search(r'public\\s+class\\s+(\\w+)', content)
            entity_name = entity_match.group(1) if entity_match else None
            
            if entity_name:
                entity_details = await LIB.get_jpa_entity_by_name(entityName=entity_name, projectPath=project_path)
                repos = await LIB.find_spring_data_repositories(
                    entityName=entity_name, moduleName="", searchInLibraries=False, projectPath=project_path
                )
        
        # Save results
        results_dir = Path(project_path) / "phase_results"
        with open(results_dir / "analysis_result.json", 'w') as f:
            json.dump({{
                "package_names": [package] if package else [],
                "base_package": exploration.get("project_structure", {{}}).get("base_package"),
                "entity_name": entity_name,
                "entity_details": entity_details if entity_name else None
            }}, f, indent=2)
        
    finally:
        await LIB.hidden_fun_cleanup()

if __name__ == "__main__":
    asyncio.run(solve_task())
```""",
            
            "implementation": f"""EXAMPLE IMPLEMENTATION PHASE CODE (SHORT - ~100 lines):
```python
import LIB
import json
import asyncio
from pathlib import Path

async def solve_task():
    project_path = "{repo_path}"
    {server_url_code}
    
    try:
        # Load previous results
        results_dir = Path(project_path) / "phase_results"
        with open(results_dir / "exploration_result.json", 'r') as f:
            exploration = json.load(f)
        with open(results_dir / "analysis_result.json", 'r') as f:
            analysis = json.load(f)
        
        base_package = analysis.get("base_package") or analysis.get("package_names", [None])[0]
        entity_package = f"{{base_package}}.domain.entities"
        
        # Create entity
        await LIB.create_jpa_entity(
            entityName="Tag", entityPackage=entity_package, language="JAVA", moduleName="",
            tableName=None, attributes=[], inheritanceType=None, discriminatorColumn=None,
            discriminatorType=None, discriminatorValue=None, superclass=None, mappedSuperclass=None,
            compositeId=None, indexes=None, namedQueries=None, projectPath=project_path
        )
        
        # Create repository
        await LIB.create_spring_data_repository(
            repositoryClassName="TagRepository", entityName="Tag",
            packageName=base_package, sourceRootType="MAIN", moduleName="",
            baseRepositoryClass="JpaRepository", projectPath=project_path
        )
        
        # Generate migration
        await LIB.generate_db_migration(
            service_name="feature-service",
            change_description="Create Tag entity and many-to-many relationship with Feature",
            projectPath=project_path
        )
        
    finally:
        await LIB.hidden_fun_cleanup()

if __name__ == "__main__":
    asyncio.run(solve_task())
```""",
            
            "verification": f"""EXAMPLE VERIFICATION PHASE CODE (SHORT - ~40 lines):
```python
import LIB
import asyncio

async def solve_task():
    project_path = "{repo_path}"
    {server_url_code}
    
    try:
        # Build project
        build_result = await LIB.build_project(
            rebuild=False, timeout=120000, projectPath=project_path
        )
        print(f"Build: {{build_result}}")
        
        # Check for errors in created files (if needed)
        # await LIB.get_file_problems(filePath="...", errorsOnly=True, timeout=10000, projectPath=project_path)
        
    finally:
        await LIB.hidden_fun_cleanup()

if __name__ == "__main__":
    asyncio.run(solve_task())
```"""
        }
        
        example_code = phase_examples.get(phase, "")
        
        if error_feedback:
            system_prompt = f"""Fix errors and generate COMPLETE, EXECUTABLE Python code for {phase} phase.

Task: {issue_title} | Repo: {repo} | Path: {repo_path}

{phase_instructions}

{previous_context}

{tool_docs}

ERRORS:
{error_feedback}

CRITICAL FIXING RULES:
- Make MINIMAL changes - only fix the specific error, don't rewrite the entire code
- Keep the same structure and logic flow
- Only modify the parts that are causing the error
- Don't add new features or refactor unnecessarily
- Preserve existing helper functions and their logic
- If error is about missing parameter, add ONLY that parameter
- If error is about wrong function name, fix ONLY that function call
- Keep code SHORT (<150 lines for exploration/analysis, <200 for implementation, <100 for verification)

REQUIREMENTS:
- `import LIB` + `import json`
- `async def solve_task():` with {server_url_code} at start, `await LIB.hidden_fun_cleanup()` in finally
- Minimal helpers, parse JSON results (may be string or dict)
- `asyncio.run(solve_task())` at end

Generate MINIMAL FIX - only change what's needed to fix the error."""
        else:
            system_prompt = f"""Generate COMPLETE, EXECUTABLE Python code for {phase} phase.

Task: {issue_title} | Repo: {repo} | Path: {repo_path}
Description: {issue_body}

{phase_instructions}

{previous_context}

{tool_docs}

REQUIREMENTS:
- `import LIB` + `import json`
- `async def solve_task():` with {server_url_code} at start, `await LIB.hidden_fun_cleanup()` in finally
- Keep code SHORT: <150 lines (exploration/analysis), <200 lines (implementation), <100 lines (verification)
- Minimal helpers, parse JSON (string or dict), no placeholders
- `asyncio.run(solve_task())` at end

{example_code}

Generate SHORT, focused {phase} phase code."""

        return await self._generate_with_iterations(system_prompt, phase, max_iterations, error_feedback is not None)
    
    async def generate_code_with_error_feedback(self, task: Dict, error_feedback: Optional[str] = None, 
                                                max_iterations: int = 10) -> str:
        """Generate code with optional error feedback from previous execution."""
        repo = task["repo"]
        issue_url = task["issue_url"]
        issue_title = task["issue_title"]
        issue_body = task["issue_body"]
        repo_path = task["repo_path"]

        tool_docs = self.lib_loader.get_tool_documentation()
        server_url_code = self._get_server_url_code()

        if error_feedback:
            system_prompt = f"""Fix errors and generate COMPLETE, EXECUTABLE code.

Task: {issue_title} | Repo: {repo} | Path: {repo_path}

{tool_docs}

ERRORS:
{error_feedback}

CRITICAL FIXING RULES:
- Make MINIMAL changes - only fix the specific error, don't rewrite the entire code
- Keep the same structure and logic flow
- Only modify the parts that are causing the error
- Don't add new features or refactor unnecessarily
- Preserve existing helper functions and their logic
- If error is about missing parameter, add ONLY that parameter
- If error is about wrong function name, fix ONLY that function call

REQUIREMENTS:
- `import LIB`, `async def solve_task():`, {server_url_code}, `await LIB.hidden_fun_cleanup()`, `asyncio.run(solve_task())`
- Check: syntax, imports, await keywords, function signatures

Generate MINIMAL FIX - only change what's needed to fix the error."""
        else:
            system_prompt = f"""Generate COMPLETE, EXECUTABLE Python code.

Task: {issue_title} | Repo: {repo} | Path: {repo_path}
Description: {issue_body}

{tool_docs}

REQUIREMENTS:
- `import LIB`, `async def solve_task():`, {server_url_code}, `await LIB.hidden_fun_cleanup()`, `asyncio.run(solve_task())`
- Workflow: EXPLORE → ANALYZE → IMPLEMENT → VERIFY
- Use `await LIB.function_name(...)`, parse JSON results, discover values (don't hardcode), use helper functions

Return executable Python code."""

        return await self._generate_with_iterations(system_prompt, "full", max_iterations, error_feedback is not None, 
                                                     require_function_calls=True)
    
    async def _generate_with_iterations(self, system_prompt: str, phase_or_type: str, max_iterations: int,
                                        has_error_feedback: bool, require_function_calls: bool = False) -> str:
        """Internal method to generate code with iterative refinement."""
        messages = [{"role": "user", "content": system_prompt}]
        logger.info(f"Generating {phase_or_type} code..." + (" (with error feedback)" if has_error_feedback else ""))
        
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

                code_length = len(generated_code)
                logger.info(f"Generated {phase_or_type} code (iteration {iteration + 1}, length: {code_length}):")
                logger.info(f"{generated_code[:500]}...")
                
                if phase_or_type == "exploration" or phase_or_type == "analysis":
                    if code_length > 150:
                        logger.warning(f"Code too long ({code_length} lines), should be <150 for {phase_or_type}")
                elif phase_or_type == "implementation":
                    if code_length > 200:
                        logger.warning(f"Code too long ({code_length} lines), should be <200 for {phase_or_type}")
                elif phase_or_type == "verification":
                    if code_length > 100:
                        logger.warning(f"Code too long ({code_length} lines), should be <100 for {phase_or_type}")

                has_lib_import = "import LIB" in generated_code
                has_async_structure = "async def" in generated_code and "asyncio.run" in generated_code
                has_no_placeholders = "..." not in generated_code
                has_cleanup = "hidden_fun_cleanup" in generated_code
                has_session_setup = "hidden_fun_create_session_parts" in generated_code
                
                has_any_lib_call = "LIB." in generated_code or "await LIB." in generated_code
                
                if require_function_calls:
                    has_function_calls = any(
                        f"LIB.{tool}" in generated_code or f"await LIB.{tool}" in generated_code
                        for tool in self.lib_loader.tools[:10]
                    )
                    is_complete = (has_function_calls and has_lib_import and has_async_structure 
                                  and has_no_placeholders and has_cleanup and has_session_setup
                                  and len(generated_code) > 200)
                else:

                    is_complete = (has_any_lib_call and has_lib_import and has_async_structure 
                                  and has_no_placeholders and has_cleanup and has_session_setup
                                  and len(generated_code) > 100)
                
                if has_error_feedback:
                    if is_complete and iteration >= 6:
                        logger.info(f"{phase_or_type} code appears complete after {iteration + 1} iteration(s) (with error feedback)")
                        return generated_code
                else:
                    if is_complete and iteration >= 2:
                        logger.info(f"{phase_or_type} code appears complete after {iteration + 1} iteration(s)")
                        return generated_code
                

                if iteration >= max_iterations - 1:
                    if not is_complete:
                        logger.warning(f"{phase_or_type} code may be incomplete, but returning due to iteration limit")
                    return generated_code

                server_url_code = self._get_server_url_code()
                improvement_prompt = self._create_improvement_prompt(generated_code, phase_or_type, server_url_code, require_function_calls)
                
                messages.append({"role": "assistant", "content": generated_code})
                messages.append({"role": "user", "content": improvement_prompt})

            except Exception as e:
                logger.error(f"Error generating {phase_or_type} code: {e}")
                import traceback
                traceback.print_exc()
                return ""

        return ""
    
    def _create_improvement_prompt(self, generated_code: str, phase_or_type: str, server_url_code: str, 
                                   require_function_calls: bool) -> str:
        """Create improvement prompt for code refinement."""
        if phase_or_type in ["exploration", "analysis", "implementation", "verification"]:
            return f"""Improve {phase_or_type} phase code. Current (may be incomplete):
```python
{generated_code[:800]}
```

FIXES NEEDED:
- Use `import LIB`, `await LIB.function_name(...)`
- `async def solve_task():` with {server_url_code} at start, `await LIB.hidden_fun_cleanup()` in finally
- Keep SHORT: <150 lines (exploration/analysis), <200 (implementation), <100 (verification)
- Minimal helpers, no placeholders, parse JSON (string or dict)
- `asyncio.run(solve_task())` at end
- Make MINIMAL changes - only fix what's missing/wrong, preserve existing structure

Generate SHORT, complete {phase_or_type} code with minimal changes."""
        else:
            return f"""Improve code. Current (may be incomplete):
```python
{generated_code[:800]}
```

FIXES NEEDED:
- Use `import LIB`, `await LIB.function_name(...)`
- `async def solve_task():` with {server_url_code} at start, `await LIB.hidden_fun_cleanup()` in finally
- Helper functions, workflow: EXPLORE → ANALYZE → IMPLEMENT → VERIFY
- Discover values, no placeholders, parse JSON results
- `asyncio.run(solve_task())` at end

Generate complete code."""

