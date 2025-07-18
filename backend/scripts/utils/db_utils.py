"""
スクリプト用のデータベースユーティリティ
"""

import sys
import os
from contextlib import contextmanager

# プロジェクトルートをパスに追加
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from database.connection import SessionLocal
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)
from database.repositories.backtest_result_repository import BacktestResultRepository


@contextmanager
def get_db_session():
    """データベースセッションを提供するコンテキストマネージャ"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_repositories(db):
    """
    必要なリポジトリをインスタンス化して辞書として返す

    Args:
        db: SQLAlchemyのセッションオブジェクト

    Returns:
        リポジトリのインスタンスを含む辞書
    """
    return {
        "ga_experiment": GAExperimentRepository(db),
        "generated_strategy": GeneratedStrategyRepository(db),
        "backtest_result": BacktestResultRepository(db),
    }
