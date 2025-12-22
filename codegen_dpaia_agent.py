import asyncio
import json
import os
import sys
import subprocess
import argparse
import logging
from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime

from mcp import ClientSession
from mcp.client.sse import sse_client
from anthropic import Anthropic
from dotenv import load_dotenv

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
    """Agent that generates Python code using MCP tools as functions."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 8000):
        self.session: Optional[ClientSession] = None
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.tools = []
        self.server_url = None
        self.proxy_url = None

    async def connect_to_mcp_server(self, server_url: str, timeout: int = 30):
        """Connect to IntelliJ MCP server to discover available tools."""
        logger.info(f"Connecting to MCP server: {server_url}")
        self.server_url = server_url

        try:
            self._streams_context = sse_client(
                url=server_url,
                timeout=timeout,
                sse_read_timeout=3600,
            )
            streams = await asyncio.wait_for(
                self._streams_context.__aenter__(),
                timeout=timeout
            )

            self._session_context = ClientSession(*streams)
            self.session: ClientSession = await self._session_context.__aenter__()

            await self.session.initialize()
            response = await self.session.list_tools()
            self.tools = response.tools

            logger.info(f"Connected. Available tools: {len(self.tools)}")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    async def cleanup(self):
        """Clean up MCP connections."""
        try:
            if hasattr(self, "_session_context"):
                await self._session_context.__aexit__(None, None, None)
            if hasattr(self, "_streams_context"):
                await self._streams_context.__aexit__(None, None, None)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def _generate_tool_wrappers_doc(self) -> str:
        """Generate documentation for available tools as Python functions."""
        doc_lines = [
            "AVAILABLE TOOLS (use as Python functions):",
            "",
            "All tools accept keyword arguments matching their MCP parameters.",
            "Example: find_jpa_entity(entity_name='User', project_path='/path/to/project')",
            "",
            "TOOLS:"
        ]
        
        for tool in self.tools:
            func_name = tool.name
            
            desc = tool.description or "No description"
            
            params = []
            if tool.inputSchema and 'properties' in tool.inputSchema:
                for param_name, param_info in tool.inputSchema['properties'].items():
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', '')
                    required = param_name in tool.inputSchema.get('required', [])
                    req_marker = " (required)" if required else " (optional)"
                    params.append(f"  - {param_name}: {param_type}{req_marker} - {param_desc}")
            
            doc_lines.append(f"\n{func_name}()")
            doc_lines.append(f"  {desc}")
            if params:
                doc_lines.append("  Parameters:")
                doc_lines.extend(params)
        
        return "\n".join(doc_lines)

    async def generate_code(self, task: Dict, max_iterations: int = 10) -> str:
        """Generate Python code to solve the task using MCP tools."""
        repo = task["repo"]
        issue_url = task["issue_url"]
        issue_title = task["issue_title"]
        issue_body = task["issue_body"]
        repo_path = task["repo_path"]

        tool_docs = self._generate_tool_wrappers_doc()

        system_prompt = f"""Generate COMPLETE, EXECUTABLE Python code to solve this Spring Boot task.

Repository: {repo} | Issue: {issue_url} | Title: {issue_title}
Description: {issue_body}
Project Path: {repo_path}

{tool_docs}

WORKFLOW: 1) EXPLORE (find entities/repos, read files) → 2) ANALYZE (extract patterns/package names) → 3) IMPLEMENT (use discovered values) → 4) VERIFY (build/check)

KEY RULES:
- CHAIN CALLS: Store results, parse (may be JSON), extract values, use in next call
- READ FIRST: Always read files with get_file_text_by_path before modifying
- DISCOVER VALUES: Extract package names from file content, don't hardcode
- PARSE RESULTS: Tool outputs may be JSON strings (use json.loads) or dicts

EXAMPLE:
```python
import json
import re

project_path = "{repo_path}"

# 1. Explore
entities = find_jpa_entities(projectPath=project_path)
feature_file = get_file_text_by_path(pathInProject="src/.../Feature.java", projectPath=project_path)

# 2. Extract package from file content
package_match = re.search(r'package\\s+([\\w.]+)', feature_file)
package_name = package_match.group(1) if package_match else "com.example"

# 3. Use discovered value
create_jpa_entity(entityPackage=package_name, projectPath=project_path, ...)

# 4. Read before modifying
old_content = get_file_text_by_path(pathInProject="...", projectPath=project_path)
new_content = modify_content(old_content)
replace_text_in_file(oldText=old_content, newText=new_content, ...)

# 5. Verify
build_project(projectPath=project_path)
```

NO placeholders, NO "...", use discovered values. Return ONLY executable Python code."""

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
                    tool.name in generated_code 
                    for tool in self.tools[:10]
                )
                has_no_placeholders = "..." not in generated_code and "..." not in generated_code
                
                if (has_function_calls and has_no_placeholders and len(generated_code) > 200) or iteration >= max_iterations - 1:
                    if not has_function_calls or not has_no_placeholders:
                        logger.warning("Code may be incomplete, but returning due to iteration limit")
                    return generated_code

                improvement_prompt = f"""The generated code needs improvement. Please generate COMPLETE, EXECUTABLE Python code following the workflow pattern.

Current code (may be incomplete):
```python
{generated_code[:1000]}
```

CRITICAL IMPROVEMENTS NEEDED:
1. **Follow exploration → analysis → implementation → verification workflow**
2. **Chain tool calls** - Use results from one call in the next (store in variables, parse results, use discovered values)
3. **Read files before modifying** - Always use get_file_text_by_path first, then replace_text_in_file
4. **Discover values** - Don't hardcode package names, entity names, etc. Discover them first
5. **Complete ALL steps** - Include exploration, reading existing files, making changes, verification
6. **Use actual discovered values** - Parse results from tool calls and use them in subsequent calls
7. **NO placeholders** - All function calls must have complete parameters

EXAMPLE OF GOOD WORKFLOW (CHAINING RESULTS):
```python
import json
import re

project_path = "{repo_path}"

# 1. Explore - discover what exists
entities_result = find_jpa_entities(projectPath=project_path)
print(f"Step 1 - Found entities: {{entities_result}}")

# 2. Parse result and extract information
# Tool results may be JSON strings or dicts - handle both
if isinstance(entities_result, str):
    try:
        entities_data = json.loads(entities_result)
    except:
        entities_data = entities_result
else:
    entities_data = entities_result

# 3. Use discovered information to read specific files
# Extract file path from entities_result or use known pattern
feature_content = get_file_text_by_path(
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
    result = create_jpa_entity(
        entityName="Tag",
        entityPackage=package_name,  # Using discovered value
        projectPath=project_path,
        # ... complete parameters
    )
    print(f"Step 4 - Created entity using package {{package_name}}")
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

    async def run_task(self, task: Dict, proxy_url: str) -> str:
        """Generate code for the task and return it."""
        self.proxy_url = proxy_url
        
        mcp_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:64343/sse")
        await self.connect_to_mcp_server(mcp_url)
        
        try:
            code = await self.generate_code(task)
            return code
        finally:
            await self.cleanup()


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
    parser.add_argument('--mcp-url', type=str, help='MCP server URL (overrides env var)')
    parser.add_argument('--proxy-url', type=str, default='http://127.0.0.1:8900/sse', 
                       help='Code proxy server URL')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-20250514', 
                       help='Claude model to use')
    parser.add_argument('--max-tokens', type=int, default=8000, help='Max tokens per request')
    parser.add_argument('--instance-id', type=str, help='Custom instance ID (default: timestamp)')
    parser.add_argument('--output', type=str, help='Output file for generated code')
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

    agent = CodeGenAgent(
        model=args.model,
        max_tokens=args.max_tokens
    )

    mcp_url = args.mcp_url or os.getenv("MCP_SERVER_URL", "http://127.0.0.1:64343/sse")
    proxy_url = args.proxy_url

    try:
        logger.info("="*60)
        logger.info(f"TASK: {TASK_CONFIG['issue_title']}")
        logger.info(f"REPO: {TASK_CONFIG['repo']}")
        logger.info(f"PATH: {TASK_CONFIG['repo_path']}")
        logger.info("="*60)

        code = await agent.run_task(TASK_CONFIG, proxy_url)

        if code:
            logger.info("="*60)
            logger.info("Generated code:")
            logger.info("="*60)
            print("\n" + "="*60)
            print("GENERATED CODE:")
            print("="*60)
            print(code)
            print("="*60 + "\n")

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

