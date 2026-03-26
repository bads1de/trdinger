import os
import sys
from pathlib import Path

# 'app' パッケージをインポートできるように、backend ディレクトリを sys.path に追加する
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, os.pardir))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# 相対インポートのために作業ディレクトリを BACKEND_DIR に変更する
os.chdir(BACKEND_DIR)

# カレントディレクトリからのインポートを許可する
if "." not in sys.path:
    sys.path.insert(0, ".")


def _is_inside_repo(path: str) -> bool:
    """指定されたパスがリポジトリルート内に解決される場合に True を返す。"""
    try:
        return os.path.commonpath([os.path.abspath(path), REPO_ROOT]) == REPO_ROOT
    except ValueError:
        return False


def _default_temp_dir() -> str:
    """リポジトリ外の通常の一時ディレクトリを優先的に使用する。"""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return os.path.join(local_app_data, "Temp")

    return str(Path.home() / "AppData" / "Local" / "Temp")


_safe_temp_dir = _default_temp_dir()
os.makedirs(_safe_temp_dir, exist_ok=True)

# シェルがブール値以外の DEBUG 値を export していても、テスト収集を安定させる。
os.environ["DEBUG"] = "false"

for _temp_var in ("TEMP", "TMP", "TMPDIR"):
    _current_temp = os.environ.get(_temp_var)
    if not _current_temp or _is_inside_repo(_current_temp):
        os.environ[_temp_var] = _safe_temp_dir



