import asyncio
import json
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

from codegen_agent import CodeGenAgent
from repository_utils import setup_repository

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H%M%S'
)
logger = logging.getLogger(__name__)


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
    parser.add_argument('--max-exec-iterations', type=int, default=10,
                       help='Maximum number of execution attempts per phase (default: 10)')
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
