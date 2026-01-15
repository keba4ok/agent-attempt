import subprocess
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime

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

