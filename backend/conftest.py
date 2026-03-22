import os
import sys
from pathlib import Path

# Ensure the backend directory is on sys.path so that 'app' package can be imported
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, os.pardir))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Change working directory to BACKEND_DIR for relative imports
os.chdir(BACKEND_DIR)

# Allow importing from current directory
if "." not in sys.path:
    sys.path.insert(0, ".")


def _is_inside_repo(path: str) -> bool:
    """Return True when a path resolves inside the repository root."""
    try:
        return os.path.commonpath([os.path.abspath(path), REPO_ROOT]) == REPO_ROOT
    except ValueError:
        return False


def _default_temp_dir() -> str:
    """Prefer the user's normal temp directory outside the repository."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return os.path.join(local_app_data, "Temp")

    return str(Path.home() / "AppData" / "Local" / "Temp")


_safe_temp_dir = _default_temp_dir()
os.makedirs(_safe_temp_dir, exist_ok=True)

# Keep test collection stable even if the shell exports a non-boolean DEBUG value.
os.environ["DEBUG"] = "false"

for _temp_var in ("TEMP", "TMP", "TMPDIR"):
    _current_temp = os.environ.get(_temp_var)
    if not _current_temp or _is_inside_repo(_current_temp):
        os.environ[_temp_var] = _safe_temp_dir



