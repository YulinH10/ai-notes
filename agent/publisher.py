import subprocess
from pathlib import Path

from .config import PROJECT_ROOT


def save_post(filename: str, content: str) -> Path:
    """Write the post file to content/posts/."""
    posts_dir = PROJECT_ROOT / "content" / "posts"
    posts_dir.mkdir(parents=True, exist_ok=True)
    filepath = posts_dir / filename
    filepath.write_text(content, encoding="utf-8")
    return filepath


def git_push(filepath: Path, topic: str):
    """Stage, commit, and push the new post."""
    _run_git("add", str(filepath))
    _run_git("commit", "-m", f"feat(post): {topic}")
    _run_git("push")


def _run_git(*args: str):
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {result.stderr}")
