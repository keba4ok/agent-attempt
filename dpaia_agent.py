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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_repository(repo: str, base_path: Optional[str] = None, instance_id: Optional[str] = None) -> str:
    """
    Clone repository into a unique instance folder.
    Each run gets its own fresh copy to avoid mixing results.
    
    Args:
        repo: Repository name in format 'owner/repo'
        base_path: Base path for repositories (defaults to script_dir/repos)
        instance_id: Unique identifier for this run (defaults to timestamp)
    
    Returns:
        Absolute path to the cloned repository instance
    """
    if base_path is None:
        script_dir = Path(__file__).parent.resolve()
        base_path = script_dir / "repos"
    else:
        base_path = Path(base_path).expanduser().resolve()
    
    # Generate unique instance ID if not provided
    if instance_id is None:
        from datetime import datetime
        instance_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    repo_name = repo.split('/')[-1]
    # Structure: repos/<repo_name>/<instance_id>/
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


class MCPAutonomousAgent:
    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 8000):
        self.session: Optional[ClientSession] = None
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.tools = []
        self.server_url = None
        self.connection_failures = 0
        self.max_connection_failures = 3

    async def connect_to_sse_server(self, server_url: str, timeout: int = 30, sse_read_timeout: float | None = 3600):
        logger.info(f"Connecting to MCP server: {server_url}")
        self.server_url = server_url

        try:
            self._streams_context = sse_client(
                url=server_url,
                timeout=timeout,
                sse_read_timeout=sse_read_timeout,
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
            self.connection_failures = 0
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    async def reconnect(self):
        if self.connection_failures >= self.max_connection_failures:
            return False

        logger.warning(f"Reconnecting (attempt {self.connection_failures + 1})...")
        self.connection_failures += 1

        try:
            await self.cleanup(silent=True)
            await asyncio.sleep(2)
            await self.connect_to_sse_server(self.server_url, timeout=30)
            return True
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False

    async def cleanup(self, silent=False):
        if not silent:
            logger.info("Cleaning up...")
        try:
            if hasattr(self, "_session_context"):
                await self._session_context.__aexit__(None, None, None)
            if hasattr(self, "_streams_context"):
                await self._streams_context.__aexit__(None, None, None)
        except Exception as e:
            if not silent:
                logger.error(f"Cleanup error: {e}")

    async def call_tool_with_retry(self, tool_name: str, tool_args: dict, timeout: int = 30) -> tuple[bool, str]:
        max_retries = 2

        for attempt in range(max_retries):
            try:
                result = await asyncio.wait_for(
                    self.session.call_tool(tool_name, tool_args),
                    timeout=timeout
                )
                return True, str(result.content)
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    if await self.reconnect():
                        continue
                return False, f"Tool {tool_name} timed out"
            except Exception as e:
                error_msg = str(e)
                if "Connection closed" in error_msg or "ReadTimeout" in error_msg:
                    if attempt < max_retries - 1:
                        if await self.reconnect():
                            continue
                return False, f"Tool {tool_name} failed: {error_msg}"

        return False, f"Tool {tool_name} failed after retries"

    async def run_task(self, task: Dict, max_steps: int = 128):
        repo = task["repo"]
        issue_url = task["issue_url"]
        issue_title = task["issue_title"]
        issue_body = task["issue_body"]
        repo_path = task["repo_path"]

        system_prompt = f"""
You are an autonomous software engineer working on a GitHub issue for a Spring Boot project.
You have access to IntelliJ MCP tools to inspect, edit, and test the repository.

Repository: {repo}
Issue: {issue_url}
Title: {issue_title}

Issue Description:
{issue_body}

Repository Path: {repo_path}

EXAMPLES OF AVAILABLE TOOLS - USE INTELLIGENTLY:
- find_jpa_entity / find_jpa_entities - Find JPA entities
- find_spring_data_repositories - Find Spring Data repositories
- get_file_text_by_path - Read file contents
- replace_text_in_file - Edit files (use old_str/new_str)
- create_new_file - Create new files
- list_directory_tree - Navigate project structure
- build_project - Compile and verify
- get_related_domain_items - Find related entities/services

- execute_terminal_command - Run shell commands (mvn test, etc.)
VERY IMPORTANT: Before using execute_terminal_command, ensure that there's no other tools
that can be used to achieve the same result. Use only when very necessary.

CRITICAL RULES:
1. ALWAYS read files with get_file_text_by_path - NEVER use cat/grep
2. Use IntelliJ tools instead of grep/find commands
3. Keep terminal commands short and specific
4. Run tests: "cd {repo_path} && mvn test -Dtest=TestClass#testMethod"
5. Use -Dtest flag for specific tests (faster)

WORKFLOW:
1. Understand requirements - break down the issue into subtasks
2. Explore project - use list_directory_tree, find_jpa_entity, find_spring_data_repositories
3. Read relevant files - use get_file_text_by_path to understand existing code
4. Implement changes - use specific tools or create_new_file, replace_text_in_file
5. Build - run build_project or get_file_problems to verify compilation
6. Test - run specific tests using tools to verify changes
7. Complete - when all requirements met and tests pass, respond with: TASK_COMPLETE

Think step-by-step and use appropriate tools.
"""

# EXTREMELY IMPORTANT: For solving a task you should absolutely use some of the following tools:
#     create_or_update_entity_attribute
#     find_jpa_entity
#     create_jpa_entity
#     create_spring_data_repository
#     generate_db_migration
#     generate_jpa_entity
#     generate_jpa_repository
#     get_jpa_entity_by_name
#     find_spring_data_repositories
#     create_jpa_query
#     reverse_engineering_jpa_entity_by_name
# These tools are most important for solving tasks from current benchmark.
# You should ABSOLUTELY use some of these tools to solve the task.

# EXTREMELY IMPORTANT: For solving a task you should absolutely use these tools:
#     [Critical]
#     find_jpa_entities
#     find_jpa_entity
#     get_jpa_entity_by_name
#     find_spring_data_repositories
#     create_jpa_query
#     create_spring_data_repository
#     generate_jpa_repository 
#     create_jpa_entity
#     generate_jpa_entity
#     create_or_update_entity_attribute
#     generate_db_migration
# IT IS EXTREMELY IMPORTANT TO USE THESE TOOLS TO SOLVE THE CURRENT TASK.
# YOU SHOULD PREFER THESE TOOLS OVER OTHER TOOLS WHENEVER POSSIBLE.

# IMPORTANT: You should also use the following tools:
# [Relevant] 
#     generate_aggregate_query
#     execute_run_configuration
#     get_run_configurations
#     find_files_by_name_keyword
#     get_file_text_by_path
#     replace_text_in_file
#     get_symbol_info
#     execute_terminal_command
#     get_related_domain_items
#     get_logical_or_architectural_map
#     find_liquibase_scripts
#     create_new_file
# Prefer these tools over other tools when solving the current task when possible.

        messages: List[Dict] = [{"role": "user", "content": system_prompt}]

        tool_schemas = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in self.tools
        ]

        logger.info(f"Starting agent (max {max_steps} steps)...")

        for step in range(max_steps):
            logger.info(f"{'='*60}")
            logger.info(f"Step {step + 1}/{max_steps}")
            logger.info('='*60)

            try:
                response = await asyncio.to_thread(
                    self.anthropic.messages.create,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    tools=tool_schemas,
                    messages=messages,
                )

                assistant_content = []
                task_complete = False

                for content in response.content:
                    if content.type == "text":
                        text = content.text.strip()
                        print(f"\nReasoning:")
                        print(f"{text[:800]}{'...' if len(text) > 800 else ''}\n")

                        assistant_content.append({"type": "text", "text": text})

                        if "TASK_COMPLETE" in text:
                            logger.info("Task marked complete.")
                            task_complete = True

                    elif content.type == "tool_use":
                        tool_name = content.name
                        tool_args = content.input
                        tool_id = content.id

                        print(f"\nTool: {tool_name}")
                        print(f"Args: {json.dumps(tool_args, indent=2)}")

                        assistant_content.append({
                            "type": "tool_use",
                            "id": tool_id,
                            "name": tool_name,
                            "input": tool_args
                        })

                        if tool_name == "execute_terminal_command":
                            cmd = tool_args.get("command", "")
                            timeout = 120 if "test" in cmd else 30
                        elif tool_name == "build_project":
                            timeout = 120
                        else:
                            timeout = 30

                        success, result_text = await self.call_tool_with_retry(
                            tool_name, tool_args, timeout=timeout
                        )

                        if not success:
                            logger.error(f"Tool {tool_name} failed: {result_text}")
                            if not hasattr(self, '_pending_tool_results'):
                                self._pending_tool_results = []
                            self._pending_tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result_text,
                                "is_error": True
                            })
                            continue

                        if tool_name == "execute_terminal_command" and '[TextContent' in result_text:
                            try:
                                import re
                                json_match = re.search(r"text='({.*?})'", result_text, re.DOTALL)
                                if json_match:
                                    cmd_result = json.loads(json_match.group(1))
                                    exit_code = cmd_result.get('command_exit_code', '?')
                                    output = cmd_result.get('command_output', '')
                                    print(f"\nExit code: {exit_code}")
                                    print(f"Output:\n{output[:2000]}")
                                    result_text = json.dumps(cmd_result)
                            except:
                                pass

                        if tool_name != "execute_terminal_command":
                            truncated = len(result_text) > 500
                            print(f"\nResult: {result_text[:500]}{'...' if truncated else ''}")

                        if not hasattr(self, '_pending_tool_results'):
                            self._pending_tool_results = []

                        self._pending_tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result_text[:8000]
                        })

                messages.append({"role": "assistant", "content": assistant_content})

                if hasattr(self, '_pending_tool_results') and self._pending_tool_results:
                    messages.append({"role": "user", "content": self._pending_tool_results})
                    self._pending_tool_results = []

                if task_complete:
                    return True

            except Exception as e:
                print(f"\n[ERROR] Step {step + 1}: {e}")
                import traceback
                traceback.print_exc()
                messages.append({
                    "role": "user",
                    "content": f"Error: {str(e)}. Try a different approach."
                })

        logger.warning("Reached max steps without completion.")
        return False


async def main():
    parser = argparse.ArgumentParser(
        description='DPAIA Autonomous Agent for Spring Boot projects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configuration
  python dpaia_agent.py

  # Specify task via JSON file
  python dpaia_agent.py --config task.json

  # Override specific parameters
  python dpaia_agent.py --repo owner/repo --issue-url https://github.com/owner/repo/issues/1

  # Custom MCP server and steps
  python dpaia_agent.py --mcp-url http://localhost:8080/sse --max-steps 64
        """
    )
    
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--repo', type=str, help='Repository in format owner/repo')
    parser.add_argument('--issue-url', type=str, help='GitHub issue URL')
    parser.add_argument('--issue-title', type=str, help='Issue title')
    parser.add_argument('--issue-body', type=str, help='Issue description')
    parser.add_argument('--mcp-url', type=str, help='MCP server URL (overrides env var)')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-20250514', help='Claude model to use')
    parser.add_argument('--max-tokens', type=int, default=8000, help='Max tokens per request')
    parser.add_argument('--max-steps', type=int, default=128, help='Maximum agent steps')
    parser.add_argument('--instance-id', type=str, help='Custom instance ID (default: timestamp)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    script_dir = Path(__file__).parent.resolve()
    
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
    
    required_fields = ["repo", "issue_url", "issue_title", "issue_body"]
    for field in required_fields:
        if not TASK_CONFIG.get(field):
            logger.error(f"Missing required field: {field}")
            sys.exit(1)
    
    logger.info("="*60)
    logger.info("REPOSITORY SETUP")
    logger.info("="*60)


    instance_id = args.instance_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info(f"Creating new repository instance: {instance_id}")
    
    repo_path = setup_repository(TASK_CONFIG["repo"], TASK_CONFIG.get("base_path"), instance_id=instance_id)
    TASK_CONFIG["repo_path"] = repo_path

    logger.info(f"Repository ready at: {repo_path}")

    agent = MCPAutonomousAgent(
        model=args.model,
        max_tokens=args.max_tokens
    )

    logs_dir = script_dir / "results_dpaia"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"dpaia_agent_{instance_id}.txt"
    

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

    try:
        server_url = args.mcp_url or os.getenv("MCP_SERVER_URL", "http://127.0.0.1:64343/sse")
        await agent.connect_to_sse_server(server_url, timeout=30)

        logger.info("="*60)
        logger.info(f"TASK: {TASK_CONFIG['issue_title']}")
        logger.info(f"REPO: {TASK_CONFIG['repo']}")
        logger.info(f"PATH: {TASK_CONFIG['repo_path']}")
        logger.info("="*60)

        success = await agent.run_task(TASK_CONFIG, max_steps=args.max_steps)

        if success:
            logger.info("="*60)
            logger.info("✓ Task completed successfully!")
            logger.info("="*60)
        else:
            logger.warning("="*60)
            logger.warning("✗ Task did not complete")
            logger.warning("="*60)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
    finally:
        await agent.cleanup()
        log_file.close()
        sys.stdout = orig_stdout
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        logger.info(f"Log saved to: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())