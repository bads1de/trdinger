"""共有ロギング設定ユーティリティ。

アプリケーション全体（FastAPI サーバー・CLI・各種スクリプト）で共通の
ログ設定を提供します。コンソール出力に加え、`backend/logs/` 配下への
ローテーション式ファイル出力を行い、実行時のエラーを後から確認できるようにします。

ログ設定は `app/config/unified_config.py` の `LoggingConfig`
（環境変数 `LOG_LEVEL`, `LOG_FORMAT`, `LOG_FILE`, `LOG_MAX_BYTES`）に集約します。
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config.unified_config import unified_config

# オートストラテジー系ロガーの名前空間
AUTO_STRATEGY_LOGGER_NAME = "app.services.auto_strategy"


def get_log_dir() -> Path:
    """ログファイルの出力ディレクトリ（backend/logs）を返します。

    Returns:
        Path: ログディレクトリのパス
    """
    backend_root = Path(__file__).resolve().parents[2]
    log_dir = backend_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _resolve_level(level: str | None) -> int:
    """ログレベル文字列を logging のレベル値に変換します。

    Args:
        level: ログレベル文字列（例: "INFO"）。None の場合は設定値を使用。

    Returns:
        int: logging モジュールのレベル値
    """
    level_str = (level or unified_config.logging.level).upper()
    resolved = getattr(logging, level_str, None)
    if not isinstance(resolved, int):
        resolved = logging.INFO
    return resolved


def configure_logging(
    level: str | None = None,
    log_file: str | None = None,
    use_console: bool = True,
    use_file: bool = True,
    backup_count: int = 5,
) -> Path | None:
    """ルートロガーを設定し、コンソールとファイルの両方にログを出力します。

    既存のハンドラーを一旦クリアしてから、コンソールハンドラーと
    `RotatingFileHandler` を追加します。

    Args:
        level: ログレベル文字列。None の場合は `LoggingConfig.level` を使用。
        log_file: 出力ファイル名。None の場合は `LoggingConfig.file` を使用。
        use_console: コンソールへ出力するか。
        use_file: ファイルへ出力するか。
        backup_count: ローテーション時のバックアップファイル数。

    Returns:
        Path | None: ファイル出力時にログファイルのパス、それ以外は None。
    """
    level_int = _resolve_level(level)
    formatter = logging.Formatter(unified_config.logging.format)

    root_logger = logging.getLogger()
    root_logger.setLevel(level_int)

    # 既存のハンドラーをクリア（多重設定・重複出力を防ぐ）
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    if use_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level_int)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    log_path: Path | None = None
    if use_file:
        log_path = get_log_dir() / (log_file or unified_config.logging.file)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=unified_config.logging.max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level_int)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # オートストラテジー専用ロガーのレベルをルートに合わせる
    auto_strategy_logger = logging.getLogger(AUTO_STRATEGY_LOGGER_NAME)
    auto_strategy_logger.setLevel(level_int)

    return log_path
