import os
import sys
import shutil
import tempfile
import uuid
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
    """テスト用の一時ディレクトリをリポジトリ内に固定する。"""
    # サンドボックスではワークスペース外の TEMP が書き込み不可になるため、
    # pytest 用の一時領域をリポジトリ配下の ignore 対象に寄せる。
    return str(Path(REPO_ROOT) / "tmp" / "pytest-backend")


_safe_temp_dir = _default_temp_dir()
os.makedirs(_safe_temp_dir, exist_ok=True)

# シェルがブール値以外の DEBUG 値を export していても、テスト収集を安定させる。
os.environ["DEBUG"] = "false"

for _temp_var in ("TEMP", "TMP", "TMPDIR"):
    os.environ[_temp_var] = _safe_temp_dir

# tempfile は初回参照時に既定値をキャッシュするので、明示的に固定する。
tempfile.tempdir = _safe_temp_dir


class _WorkspaceTemporaryDirectory:
    """ワークスペース内で確実に書き込める一時ディレクトリ実装。"""

    def __init__(
        self,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | os.PathLike[str] | None = None,
        ignore_cleanup_errors: bool = False,
    ) -> None:
        base_dir = Path(dir) if dir is not None else Path(_safe_temp_dir)
        if not _is_inside_repo(str(base_dir)):
            base_dir = Path(_safe_temp_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        prefix = prefix or "tmp"
        suffix = suffix or ""
        while True:
            candidate = base_dir / f"{prefix}{uuid.uuid4().hex}{suffix}"
            try:
                candidate.mkdir()
                break
            except FileExistsError:
                continue

        self.name = str(candidate)
        self._ignore_cleanup_errors = ignore_cleanup_errors
        self._closed = False

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        shutil.rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass


tempfile.TemporaryDirectory = _WorkspaceTemporaryDirectory
