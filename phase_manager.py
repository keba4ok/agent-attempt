import json
import logging
from typing import Optional, Dict, List
from pathlib import Path

from code_executor import ExecutionResult

logger = logging.getLogger(__name__)

PHASES = ["exploration", "analysis", "implementation", "verification"]


class PhaseManager:
    """Manages phase execution, result storage, and context passing."""
    
    def __init__(self, save_code_dir: Optional[Path] = None):
        self.save_code_dir = save_code_dir
    
    def get_phase_results_dir(self, instance_id: str, project_path: str = None) -> Optional[Path]:
        """Get directory for storing phase results."""
        if project_path:
            phase_dir = Path(project_path) / "phase_results"
            phase_dir.mkdir(parents=True, exist_ok=True)
            return phase_dir
        elif self.save_code_dir:
            phase_dir = self.save_code_dir.parent / "phase_results" / instance_id
            phase_dir.mkdir(parents=True, exist_ok=True)
            return phase_dir
        return None
    
    def save_phase_result(self, phase: str, result: Dict, instance_id: str, project_path: str = None):
        """Save phase execution result to JSON file."""
        phase_dir = self.get_phase_results_dir(instance_id, project_path)
        if not phase_dir:
            return
        
        result_file = phase_dir / f"{phase}_result.json"
        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {phase} result to: {result_file}")
        except Exception as e:
            logger.warning(f"Failed to save {phase} result: {e}")
    
    def load_phase_result(self, phase: str, instance_id: str, project_path: str = None) -> Optional[Dict]:
        """Load phase execution result from JSON file."""
        phase_dir = self.get_phase_results_dir(instance_id, project_path)
        if not phase_dir:
            return None
        
        result_file = phase_dir / f"{phase}_result.json"
        if not result_file.exists():
            return None
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {phase} result: {e}")
            return None
    
    def load_phase_result_from_project(self, phase: str, project_path: str) -> Optional[Dict]:
        """Load phase result JSON file saved by the generated code in the project directory."""
        if not project_path:
            return None
        
        result_file = Path(project_path) / "phase_results" / f"{phase}_result.json"
        if not result_file.exists():
            return None
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {phase} result from project: {e}")
            return None
    
    def get_previous_phases_context(self, current_phase: str, instance_id: str, project_path: str = None) -> str:
        """Get context from all previous phases as a string for codegen."""
        current_idx = PHASES.index(current_phase)
        context_parts = []
        
        for phase in PHASES[:current_idx]:
            result = self.load_phase_result(phase, instance_id, project_path)
            if result:
                context_parts.append(f"\n{phase.upper()} PHASE RESULTS:\n{json.dumps(result, indent=2)}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def extract_phase_result(self, execution_result: ExecutionResult, phase: str) -> Dict:
        """Extract phase-specific results from execution output (fallback)."""
        result_data = {}
        
        if execution_result.success:
            stdout = execution_result.stdout
            try:
                import re
                json_match = re.search(r'\{.*\}', stdout, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group(0))
            except:
                pass
        
        return result_data

