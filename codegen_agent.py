import asyncio
import json
import logging
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from datetime import datetime

import os
from anthropic import Anthropic

from code_executor import CodeExecutor, ExecutionResult
from error_parser import extract_error_feedback
from lib_loader import LibLoader
from phase_manager import PhaseManager, PHASES
from code_generator import CodeGenerator

logger = logging.getLogger(__name__)


class CodeGenAgent:
    """Agent that generates Python code using generated LIB functions from MCP tools."""
    
    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 8000, 
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
        if save_code_dir:
            self.save_code_dir = Path(save_code_dir).expanduser().resolve()
            self.save_code_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_code_dir = None
        
        self.lib_loader = LibLoader(self.lib_path)
        self.phase_manager = PhaseManager(self.save_code_dir)
        self.code_generator = CodeGenerator(
            self.anthropic, self.model, self.max_tokens,
            self.lib_loader, self.phase_manager, self.server_url
        )

    def load_lib(self):
        if not self.lib_path.exists():
            raise FileNotFoundError(f"LIB file not found: {self.lib_path}. Please generate LIB.py first.")
        
        logger.info(f"Loading LIB from: {self.lib_path}")
        self.lib_loader.load()

    def _save_generated_code(self, code: str, iteration: int, instance_id: str = None, phase: str = None):
        """Save generated code to file for later review, organized by instance_id."""
        if not self.save_code_dir or not code:
            return
        
        if instance_id is None:
            instance_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        instance_dir = self.save_code_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        
        if phase:
            filename = f"phase_{phase}_iter_{iteration:02d}.py"
        else:
            filename = f"generated_code_iter_{iteration:02d}.py"
        file_path = instance_dir / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            logger.info(f"Saved generated code to: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to save generated code: {e}")

    async def run_task(self, task: Dict, execute: bool = False, 
                      max_exec_iterations: int = 10, instance_id: str = None) -> Tuple[str, List[ExecutionResult]]:
        """
        Generate code for the task and optionally execute it with phase-based iterative improvement.
        
        Args:
            task: Task configuration dictionary
            execute: Whether to execute the code and iteratively improve
            max_exec_iterations: Maximum number of execution attempts per phase
            instance_id: Instance ID for saving code files
        """
        self.load_lib()
        
        execution_history = []
        
        if not execute:
            code = await self.code_generator.generate_code_with_error_feedback(task)
            if code:
                self._save_generated_code(code, iteration=0, instance_id=instance_id)
            return code, execution_history
        
        logger.info("="*60)
        logger.info("Starting phase-based code generation and execution")
        logger.info("="*60)
        
        executor = CodeExecutor(
            lib_path=str(self.lib_path),
            server_url=self.server_url,
            project_path=task.get("repo_path")
        )
        
        phase_results = {}
        all_code = []
        max_full_iterations = 3
        
        for full_iteration in range(max_full_iterations):
            logger.info(f"\n{'='*80}")
            logger.info(f"FULL ITERATION {full_iteration + 1}/{max_full_iterations}")
            logger.info(f"{'='*80}")
            
            iteration_successful = True
            
            for phase_idx, phase in enumerate(PHASES):
                logger.info(f"\n{'='*60}")
                logger.info(f"PHASE {phase_idx + 1}/{len(PHASES)}: {phase.upper()}")
                logger.info(f"{'='*60}")
                
                project_path = task.get("repo_path", "")
                
                existing_result = self.phase_manager.load_phase_result(phase, instance_id, project_path)
                if full_iteration == 0 and existing_result and existing_result.get("success"):
                    logger.info(f"Phase {phase} already completed successfully, skipping...")
                    phase_results[phase] = existing_result
                    continue
                
                # Context from previous phases in order:
                # 1. Previous phases from current iteration (in order)
                # 2. Same phase from previous iteration
                # 3. Verification from previous iteration
                previous_phases_context = []
                current_idx = PHASES.index(phase)
                
                for prev_phase in PHASES[:current_idx]:
                    if prev_phase in phase_results and phase_results[prev_phase].get("data"):
                        prev_data = phase_results[prev_phase].get("data", {})
                        previous_phases_context.append(f"{prev_phase.upper()} (iteration {full_iteration + 1}) results:\n{json.dumps(prev_data, indent=2)}")
                
                if full_iteration > 0:
                    same_phase_prev = self.phase_manager.load_phase_result(phase, instance_id, project_path)
                    if same_phase_prev and same_phase_prev.get("data"):
                        prev_data = same_phase_prev.get("data", {})
                        previous_phases_context.append(f"{phase.upper()} (iteration {full_iteration}) results:\n{json.dumps(prev_data, indent=2)}")
                    
                    prev_verification = self.phase_manager.load_phase_result("verification", instance_id, project_path)
                    if prev_verification:
                        if prev_verification.get("data"):
                            prev_data = prev_verification.get("data", {})
                            previous_phases_context.append(f"VERIFICATION (iteration {full_iteration}) results:\n{json.dumps(prev_data, indent=2)}")
                        if not prev_verification.get("success"):
                            verification_feedback = extract_error_feedback(
                                ExecutionResult(
                                    success=False,
                                    stdout=prev_verification.get("stdout", ""),
                                    stderr=prev_verification.get("stderr", ""),
                                    exit_code=prev_verification.get("exit_code", 1),
                                    error_output=prev_verification.get("error", "")
                                )
                            )
                            if verification_feedback:
                                previous_phases_context.append(f"VERIFICATION (iteration {full_iteration}) FAILED:\n{verification_feedback}")
                
                previous_iteration_context = "\n\n".join(previous_phases_context) if previous_phases_context else None
                if previous_iteration_context:
                    logger.info(f"Including context from previous phases for {phase} phase (iteration {full_iteration + 1})")
            
                phase_code = None
                phase_execution_history = []
                
                for exec_iteration in range(max_exec_iterations):
                    logger.info(f"\n--- {phase.upper()} Phase - Execution Attempt {exec_iteration + 1}/{max_exec_iterations} (Iteration {full_iteration + 1}/{max_full_iterations}) ---")
                    logger.info(f"Each attempt includes up to 10 internal code refinement iterations before execution.")
                    
                    error_feedback = None
                    if exec_iteration > 0 and phase_execution_history:
                        last_result = phase_execution_history[-1]
                        exec_error_feedback = extract_error_feedback(last_result)
                        if exec_error_feedback:
                            error_feedback = exec_error_feedback
                            if previous_iteration_context:
                                error_feedback = f"{previous_iteration_context}\n\n{error_feedback}"
                    elif previous_iteration_context:
                        error_feedback = previous_iteration_context
                    
                    if error_feedback:
                        logger.info(f"Regenerating {phase} phase code with error feedback...")
                    elif exec_iteration > 0:
                        logger.info(f"{phase} phase completed successfully")
                        break
                
                    phase_code = await self.code_generator.generate_phase_code(
                        phase=phase,
                        task=task,
                        error_feedback=error_feedback,
                        instance_id=instance_id,
                        max_iterations=10
                    )
                    
                    if not phase_code:
                        logger.error(f"Failed to generate {phase} phase code")
                        break
                    
                    self._save_generated_code(
                        code=phase_code, 
                        iteration=exec_iteration + 1, 
                        instance_id=instance_id,
                        phase=phase
                    )
                    
                    logger.info(f"Executing {phase} phase code...")
                    result = executor.execute(phase_code, timeout=300, instance_id=instance_id, phase=phase)
                    phase_execution_history.append(result)
                    execution_history.append(result)
                    
                    logger.info(f"{phase.upper()} execution result: {'SUCCESS' if result.success else 'FAILED'}")
                    logger.info(f"Tool calls made: {result.tool_calls_count}")
                    if result.tool_calls_stats:
                        sorted_stats = sorted(result.tool_calls_stats.items(), key=lambda item: item[1], reverse=True)
                        stats_str = ", ".join([f"{name}: {count}" for name, count in sorted_stats])
                        logger.info(f"Tool calls breakdown: {stats_str}")
                    if result.tool_calls_log_path:
                        logger.info(f"Tool calls log: {result.tool_calls_log_path}")
                    
                    if result.success:
                        phase_result_data = self.phase_manager.load_phase_result_from_project(phase, project_path)
                        if not phase_result_data:
                            phase_result_data = self.phase_manager.extract_phase_result(result, phase)
                        
                        phase_result = {
                            "success": True,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "exit_code": result.exit_code,
                            "data": phase_result_data
                        }
                        self.phase_manager.save_phase_result(phase, phase_result, instance_id, project_path)
                        phase_results[phase] = phase_result
                        
                        logger.info(f"{phase.upper()} phase completed successfully!")
                        logger.info(f"Output:\n{result.stdout[:500]}...")
                        break
                    else:
                        logger.warning(f"{phase.upper()} phase failed (exit code: {result.exit_code})")
                        logger.warning(f"Error output:\n{result.error_output[:500]}...")
                        if exec_iteration < max_exec_iterations - 1:
                            logger.info(f"Will regenerate {phase} phase code with error feedback...")
                        else:
                            logger.warning(f"Max execution iterations reached for {phase} phase")
                            phase_result = {
                                "success": False,
                                "stdout": result.stdout,
                                "stderr": result.stderr,
                                "exit_code": result.exit_code,
                                "error": result.error_output
                            }
                            self.phase_manager.save_phase_result(phase, phase_result, instance_id, project_path)
                            phase_results[phase] = phase_result
                
                if phase_code:
                    all_code.append(f"# === {phase.upper()} PHASE (Iteration {full_iteration + 1}) ===\n{phase_code}\n")
                
                if phase in phase_results and not phase_results[phase].get("success"):
                    if phase in ["exploration", "analysis"]:
                        logger.error(f"Phase {phase} failed after {max_exec_iterations} attempts. STOPPING execution.")
                        logger.error("Cannot proceed to next phases without successful completion of this phase.")
                        final_code = "\n".join(all_code) if all_code else None
                        return final_code, execution_history
                    else:
                        iteration_successful = False
                        logger.warning(f"Phase {phase} failed. Will retry in next full iteration if available.")
            
            if phase_results.get("verification", {}).get("success"):
                logger.info("="*80)
                logger.info(f"VERIFICATION SUCCEEDED after {full_iteration + 1} iteration(s)!")
                logger.info("="*80)
                break
            elif full_iteration < max_full_iterations - 1:
                logger.warning("="*80)
                logger.warning(f"Verification failed or incomplete. Starting iteration {full_iteration + 2}/{max_full_iterations}...")
                logger.warning("="*80)
                if "verification" in phase_results:
                    del phase_results["verification"]
                if "implementation" in phase_results:
                    del phase_results["implementation"]
            else:
                logger.warning("="*80)
                logger.warning(f"Reached max full iterations ({max_full_iterations}). Verification may not have succeeded.")
                logger.warning("="*80)
        
        final_code = "\n".join(all_code) if all_code else None
        
        total_tool_calls = sum(result.tool_calls_count for result in execution_history)
        combined_stats = {}
        for result in execution_history:
            if result.tool_calls_stats:
                for tool_name, count in result.tool_calls_stats.items():
                    combined_stats[tool_name] = combined_stats.get(tool_name, 0) + count
        
        logger.info("="*80)
        logger.info("Phase-based execution completed")
        logger.info(f"Total tool calls across all phases and iterations: {total_tool_calls}")
        if combined_stats:
            sorted_combined_stats = sorted(combined_stats.items(), key=lambda item: item[1], reverse=True)
            logger.info("Tool usage statistics (sorted by frequency):")
            for tool_name, count in sorted_combined_stats:
                logger.info(f"  {tool_name}: {count} calls")
        logger.info("="*80)
        
        return final_code, execution_history

