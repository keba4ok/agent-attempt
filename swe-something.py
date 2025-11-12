import asyncio
import json
import os
import sys
from typing import Optional, List, Dict
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

def load_swebench_instance(instance_id: str, dataset_path: str = "~/swe-mcp-demo/swebench_verified.jsonl") -> dict:
    dataset_path = Path(dataset_path).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"SWE-bench dataset not found at {dataset_path}")

    if dataset_path.suffix == ".jsonl":
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data["instance_id"] == instance_id:
                    return data
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
            for data in data_list:
                if data["instance_id"] == instance_id:
                    return data

    raise ValueError(f"Instance {instance_id} not found in {dataset_path}")


class MCPAutonomousAgent:
    def __init__(self, model: str = "claude-3-5-haiku-20241022", max_tokens: int = 4096):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.tools = []
        self.environment_ready = False
        self.server_url = None
        self.connection_failures = 0
        self.max_connection_failures = 3

    async def connect_to_sse_server(self, server_url: str, timeout: int = 30):
        print(f"Connecting to MCP runtime: {server_url}")
        self.server_url = server_url

        try:
            self._streams_context = sse_client(url=server_url)
            streams = await asyncio.wait_for(
                self._streams_context.__aenter__(),
                timeout=timeout
            )

            self._session_context = ClientSession(*streams)
            self.session: ClientSession = await self._session_context.__aenter__()

            await self.session.initialize()
            response = await self.session.list_tools()
            self.tools = response.tools

            print("Connected. Available tools:", [tool.name for tool in self.tools])
            self.connection_failures = 0
        except asyncio.TimeoutError:
            print(f"Connection timeout after {timeout}s")
            raise
        except Exception as e:
            print(f"Connection failed: {e}")
            raise

    async def reconnect(self):
        if self.connection_failures >= self.max_connection_failures:
            print(f"Max reconnection attempts ({self.max_connection_failures}) reached")
            return False

        print(f"Attempting to reconnect (attempt {self.connection_failures + 1}/{self.max_connection_failures})...")
        self.connection_failures += 1

        try:
            await self.cleanup(silent=True)
            await asyncio.sleep(2)
            await self.connect_to_sse_server(self.server_url, timeout=30)
            print("Reconnection successful!")
            return True
        except Exception as e:
            print(f"Reconnection failed: {e}")
            return False

    async def cleanup(self, silent=False):
        if not silent:
            print("Cleaning up session...")
        try:
            if hasattr(self, "_session_context"):
                await self._session_context.__aexit__(None, None, None)
            if hasattr(self, "_streams_context"):
                await self._streams_context.__aexit__(None, None, None)
        except Exception as e:
            if not silent:
                print(f"Cleanup error: {e}")

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
                print(f"Tool call timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    reconnected = await self.reconnect()
                    if not reconnected:
                        return False, f"Connection lost and reconnection failed"
                else:
                    return False, f"Tool {tool_name} timed out after {timeout}s"

            except Exception as e:
                error_msg = str(e)
                if "Connection closed" in error_msg or "ReadTimeout" in error_msg:
                    print(f"Connection error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                    if attempt < max_retries - 1:
                        reconnected = await self.reconnect()
                        if not reconnected:
                            return False, f"Connection lost: {error_msg}"
                    else:
                        return False, f"Connection failed: {error_msg}"
                else:
                    return False, f"Tool {tool_name} failed: {error_msg}"

        return False, f"Tool {tool_name} failed after {max_retries} attempts"

    async def setup_environment(self, repo_path: str = "."):
        if self.environment_ready:
            return True

        print("Setting up environment...")

        try:
            success, result = await self.call_tool_with_retry(
                "execute_terminal_command",
                {"command": f"cd {repo_path} && pip install -e .[test] --quiet"},
                timeout=120
            )

            if success:
                print(f"Environment setup complete")
                self.environment_ready = True
                return True
            else:
                print(f"Environment setup failed: {result}")
                return False

        except Exception as e:
            print(f"Environment setup failed: {e}")
            return False

    async def run_task(self, task: Dict, max_steps: int = 20):
        repo = task["repo"]
        commit = task["commit"]
        problem = task["problem"]
        repo_path = task.get("repo_path", ".")

        env_ok = await self.setup_environment(repo_path)
        if not env_ok:
            print("Proceeding without full environment setup...")

        system_prompt = f"""
You are an autonomous software engineer working on a bug fix.
You have access to tools through the MCP runtime to inspect, edit, and test a repository.

Repository: {repo}
Commit: {commit}

Bug to fix:
{problem}

Examples of available tools:
- find_files_by_name_keyword(nameKeyword) - Find files by keyword
- get_file_text_by_path(pathInProject) - Read file contents (PREFERRED over cat/grep)
- replace_text_in_file(pathInProject, oldText, newText) - Edit files
- search_in_files_by_text(searchText, fileMask) - Search for text in files
- execute_terminal_command(command) - Run shell commands
- create_new_file(pathInProject, text) - Create new files (NOTE: parent directories must exist)

FILE CREATION BEST PRACTICES:
- Before using create_new_file, ensure parent directory exists
- Check if file already exists using find_files_by_name_keyword
- If create_new_file fails, use execute_terminal_command with "mkdir -p dir && cat > file" as fallback
- Prefer execute_terminal_command for complex file operations

EXAMPLES OF GOOD COMMANDS:
- Check if file exists: "test -f path/to/file && echo 'EXISTS' || echo 'NOT FOUND'"
- List directory: "ls -la path/"
- Run specific test: "cd {repo_path} && python -m pytest path/to/test.py::TestClass::test_method -xvs"
- Check Python syntax: "python -m py_compile path/to/file.py && echo 'OK'"
- Create file (if create_new_file fails): "cat > path/to/file.py << 'EOF'
[file content]
EOF"
- Create file with directory: "mkdir -p path/to && cat > path/to/file.py << 'EOF'
[file content]  
EOF"

CRITICAL RULES TO AVOID TIMEOUTS:
1. ALWAYS use get_file_text_by_path() to read files - NEVER use cat/grep commands
2. AVOID long-running terminal commands (cat, grep on large files, etc)
3. Keep terminal commands short and specific
4. Use MCP tools instead of shell commands whenever possible
5. If a tool fails, try a different approach immediately
6. To create files, use create_new_file tool - NOT echo commands with complex content

TESTING INSTRUCTIONS:
- To run tests: execute_terminal_command with "cd {repo_path} && python -m pytest <test_file> -xvs -k test_name"
- Use -xvs flags: -x (stop on first failure), -v (verbose), -s (show output)
- Use -k to run specific tests only (faster)
- Run tests from repository root
- Always check test output - empty output may mean no tests were found or file doesn't exist

Workflow:
1. Use find_files_by_name_keyword or search_in_files_by_text to locate files
2. Use get_file_text_by_path to read file contents (NOT cat or grep!)
3. Make code changes with replace_text_in_file
4. Run specific tests to verify the fix
5. When tests pass, respond with: TASK_COMPLETE

Always reason step-by-step and use the most efficient tool for each task.
"""

        messages: List[Dict] = [{"role": "user", "content": system_prompt}]

        tool_schemas = []
        for tool in self.tools:
            tool_schemas.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            })

        print(f"Starting autonomous agent loop (max {max_steps} steps)...\n")

        for step in range(max_steps):
            print(f"\n{'='*60}")
            print(f"Step {step + 1}/{max_steps}")
            print('='*60)

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
                        print(f"\nClaude's reasoning:")
                        print(f"{text[:800]}{'...' if len(text) > 800 else ''}\n")

                        assistant_content.append({
                            "type": "text",
                            "text": text
                        })

                        if "TASK_COMPLETE" in text:
                            print("Task marked as complete by Claude.")
                            task_complete = True

                    elif content.type == "tool_use":
                        tool_name = content.name
                        tool_args = content.input
                        tool_id = content.id

                        print(f"\nTool call: {tool_name}")
                        print(f"   Args: {json.dumps(tool_args, indent=6)}")

                        assistant_content.append({
                            "type": "tool_use",
                            "id": tool_id,
                            "name": tool_name,
                            "input": tool_args
                        })

                        if tool_name == "execute_terminal_command":
                            cmd = tool_args.get("command", "")
                            if "pytest" in cmd:
                                timeout = 60
                            else:
                                timeout = 20
                        else:
                            timeout = 30

                        success, result_text = await self.call_tool_with_retry(
                            tool_name, tool_args, timeout=timeout
                        )

                        if not success:
                            error_msg = result_text

                            if tool_name == "create_new_file":
                                error_msg += "\n\nSUGGESTION: create_new_file often fails if parent directory doesn't exist. Try using execute_terminal_command instead:\n"
                                file_path = tool_args.get('pathInProject', '')
                                error_msg += f"  mkdir -p $(dirname {file_path}) && cat > {file_path} << 'EOF'\n  [your content]\n  EOF"

                            print(f"[ERROR] {error_msg}")

                            if not hasattr(self, '_pending_tool_results'):
                                self._pending_tool_results = []

                            self._pending_tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": error_msg,
                                "is_error": True
                            })
                            continue

                        if tool_name == "execute_terminal_command" and result_text.startswith('[TextContent'):
                            try:
                                import re
                                import ast

                                json_match = re.search(r"text='({.*?})'(?:,|\))", result_text, re.DOTALL)
                                if not json_match:
                                    json_match = re.search(r'text="({.*?})"(?:,|\))', result_text, re.DOTALL)

                                if json_match:
                                    json_str = json_match.group(1)

                                    cmd_result = None

                                    try:
                                        cmd_result = json.loads(json_str)
                                    except json.JSONDecodeError:
                                        pass

                                    if not cmd_result:
                                        try:
                                            decoded_str = ast.literal_eval(f'"{json_str}"')
                                            cmd_result = json.loads(decoded_str)
                                        except:
                                            pass

                                    if not cmd_result:
                                        try:
                                            fixed_str = json_str.replace('\\n', '\n')
                                            fixed_str = fixed_str.replace('\\t', '\t')
                                            fixed_str = fixed_str.replace('\\r', '\r')
                                            fixed_str = fixed_str.replace('\\"', '"')
                                            fixed_str = fixed_str.replace("\\'", "'")
                                            cmd_result = json.loads(fixed_str)
                                        except:
                                            pass

                                    if cmd_result:
                                        exit_code = cmd_result.get('command_exit_code', 'unknown')
                                        output = cmd_result.get('command_output', result_text)

                                        print(f"\nCommand exit code: {exit_code}")

                                        if not output or output.strip() == "":
                                            print(f"   [WARNING] Command produced no output (this may indicate no tests were found or command was silent)")
                                            output = "[No output - command may have been silent or produced no results]"

                                        truncated = len(output) > 2000
                                        display_output = output[:2000] + ("\n...[truncated]" if truncated else "")
                                        print(f"   Output:\n{display_output}")

                                        result_text = json.dumps(cmd_result)
                                    else:
                                        print(f"\nCommand output (raw):\n{result_text[:500]}...")

                            except Exception as e:
                                print(f"Could not parse command output: {e}")

                        truncated = len(result_text) > 5000
                        display_text = result_text[:5000] + ("...[truncated]" if truncated else "")

                        if tool_name != "execute_terminal_command":
                            print(f"\nTool result ({len(result_text)} chars):")
                            print(f"{display_text[:500]}...")

                        if not hasattr(self, '_pending_tool_results'):
                            self._pending_tool_results = []

                        result_context = result_text[:8000]
                        if tool_name == "execute_terminal_command":
                            try:
                                import re
                                json_match = re.search(r"text='({.*?})'", result_text, re.DOTALL)
                                if json_match:
                                    cmd_data = json.loads(json_match.group(1).replace('\\n', '\n'))
                                    output = cmd_data.get('command_output', '')
                                    exit_code = cmd_data.get('command_exit_code', 0)

                                    if not output or output.strip() == "":
                                        result_context = json.dumps({
                                            "command_exit_code": exit_code,
                                            "command_output": "[No output produced - command was silent or produced no results. For pytest, this often means no tests were found or file doesn't exist. Consider checking if the file exists first or using more specific test paths.]"
                                        })
                            except:
                                pass

                        self._pending_tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result_context
                        })

                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })

                if hasattr(self, '_pending_tool_results') and self._pending_tool_results:
                    messages.append({
                        "role": "user",
                        "content": self._pending_tool_results
                    })
                    self._pending_tool_results = []

                if task_complete:
                    return True

            except Exception as e:
                print(f"\n[ERROR] Error in step {step + 1}: {e}")
                import traceback
                traceback.print_exc()

                messages.append({
                    "role": "user",
                    "content": f"An error occurred: {str(e)}. Please try a different approach."
                })

        print(f"\nReached max steps ({max_steps}) without TASK_COMPLETE.")
        return False


async def main():
    # Example task
    instance_id = "astropy__astropy-13579"
    raw = load_swebench_instance(instance_id)

    TASK = {
        "repo": raw["repo"],
        "commit": raw["base_commit"],
        "repo_path": f"/Users/ivan.kabashnyi/swe-mcp-demo/repos/{raw['repo'].split('/')[-1]}",
        "problem": raw["problem_statement"],
    }

    agent = MCPAutonomousAgent(
        model="claude-3-5-haiku-20241022",
        max_tokens=4096
    )

    from datetime import datetime
    log_path = f"agent_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_path, "w", encoding="utf-8")
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
        server_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:64342/sse")
        await agent.connect_to_sse_server(server_url, timeout=30)

        success = await agent.run_task(TASK, max_steps=50)

        if success:
            print("\n" + "="*60)
            print("Task completed successfully!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("Task did not complete within step limit.")
            print("="*60)

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        await agent.cleanup()
        log_file.close()
        sys.stdout = orig_stdout


if __name__ == "__main__":
    asyncio.run(main())