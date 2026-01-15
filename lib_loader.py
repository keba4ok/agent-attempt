import importlib.util
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class LibLoader:
    """Handles loading and inspecting the generated LIB module."""
    
    def __init__(self, lib_path: Path):
        self.lib_path = lib_path
        self.lib_module = None
        self.tools = []
    
    def load(self):
        """Load the LIB module and extract available tools."""
        if not self.lib_path.exists():
            raise FileNotFoundError(f"LIB file not found: {self.lib_path}. Please generate LIB.py first.")
        
        logger.info(f"Loading LIB from: {self.lib_path}")
        
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
    
    def get_tool_documentation(self) -> str:
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
    
    def _generate_tool_wrappers_doc_fallback(self) -> str:
        """Generate documentation by inspecting the LIB module."""
        if not self.lib_module:
            return "LIB module not loaded."
        
        doc_lines = ["AVAILABLE TOOLS (use as LIB.function_name() with await):\n"]
        doc_lines.append("All tools are async functions available in the LIB module.\n")
        doc_lines.append("Example: result = await LIB.function_name(...)\n\n")
        doc_lines.append("IMPORTANT: All LIB function calls must be prefixed with 'await' since they are async.\n\n")
        
        for tool_name in sorted(self.tools):
            doc_lines.append(f"- {tool_name}()\n")
        
        return "".join(doc_lines)

