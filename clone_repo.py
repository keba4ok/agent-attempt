import subprocess
import sys
from pathlib import Path
from typing import Optional

def clone_and_checkout(repo: str, commit: str, base_dir: Optional[Path] = None) -> Path:
    """
    Clone a GitHub repository and checkout a specific commit.
    Returns the absolute path to the cloned repository.
    """
    if base_dir is None:

        script_dir = Path(__file__).parent.resolve()
        base_dir = script_dir / "repos"
    
    repo_name = repo.split("/")[1]
    repo_dir = base_dir / repo_name
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Setting up {repo}@{commit[:8]}...")

    if not repo_dir.exists():
        print(f"Cloning https://github.com/{repo}.git ...")
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo}.git", str(repo_dir)],
            check=True,
        )
    else:
        print(f"Repository already exists at {repo_dir}")

    print(f"Checking out {commit[:8]}...")
    subprocess.run(
        ["git", "fetch", "--all"],
        cwd=str(repo_dir),
        check=True,
    )
    subprocess.run(
        ["git", "checkout", commit],
        cwd=str(repo_dir),
        check=True,
    )

    print(f"Repository ready at {repo_dir.absolute()}")
    return repo_dir.absolute()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clone_repo.py <user/repo> <commit_hash>")
        sys.exit(1)

    repo = sys.argv[1]
    commit = sys.argv[2]
    clone_and_checkout(repo, commit)