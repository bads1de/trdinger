"""
並列評価ワーカーモジュール

Windowsのmultiprocessing (spawn) に対応するため、
トップレベル関数として評価ロジックを定義します。
"""

import logging
from typing import Any

import pandas as pd

# 遅延インポートで循環参照と起動コストを避ける。
# GA ワーカープロセスは Windows の spawn で生成されるため、
# モジュール import 時間がそのままワーカー起動時間になる。
# 重い依存（backtesting._plotting 等を連鎖 import する
# individual_evaluator / fitness_sharing_vectorizer）は
# ワーカー初期化時まで遅延させる。
from .parallel_evaluator import ParallelEvaluationResult

logger = logging.getLogger(__name__)

# ワーカープロセスごとのグローバル変数
_WORKER_EVALUATOR: Any | None = None
_WORKER_CONFIG: Any | None = None


def initialize_worker_process(
    backtest_config: dict[str, Any],
    ga_config: Any,
    shared_data: dict[str, Any] | None = None,
) -> None:
    """
    ワーカープロセスの初期化関数

    Args:
        backtest_config: バックテスト設定
        ga_config: GA設定
        shared_data: 親プロセスから共有されるデータ（OHLCVなど）
    """
    global _WORKER_EVALUATOR, _WORKER_CONFIG

    try:
        from app.services.backtest.services.backtest_service import BacktestService

        from .individual_evaluator import IndividualEvaluator

        # 1. BacktestServiceの初期化（これによりDB接続も各プロセスで確立）
        # 引数なしで初期化すると、デフォルトのリポジトリとDBセッションが作成される
        backtest_service = BacktestService()

        # 2. Evaluatorの初期化
        evaluator = IndividualEvaluator(backtest_service)

        # 3. 設定の適用
        evaluator.set_backtest_config(backtest_config)

        # 4. 共有データの適用（DBアクセス削減のため）
        if shared_data:
            # ワーカーコンテキストに共有データを登録
            from .parallel_evaluator import initialize_worker

            initialize_worker(shared_data)

            # 親プロセスでプリフェッチ済みの全期間データをローカルキャッシュへ
            # 直接投入する。個体ごとの warmup シフトにより要求キーがズレても、
            # BacktestDataProvider の内包スライス/前方拡張がこのキャッシュを
            # 再利用し、DB アクセスを不足分のみに抑えられる。
            for payload in shared_data.values():
                if not isinstance(payload, dict):
                    continue
                cache_key = payload.get("key")
                cache_data = payload.get("data")
                if isinstance(cache_key, tuple) and isinstance(
                    cache_data, pd.DataFrame
                ):
                    with evaluator._lock:
                        evaluator._data_cache[cache_key] = cache_data

        _WORKER_EVALUATOR = evaluator
        _WORKER_CONFIG = ga_config

    except Exception as e:
        logger.error(f"Worker initialization failed: {e}")
        raise


def worker_evaluate_individual(individual: Any) -> ParallelEvaluationResult:
    """
    個体評価関数（ワーカープロセス内で実行）

    Args:
        individual: 評価対象の個体

    Returns:
        フィットネス値のタプル
    """
    if _WORKER_EVALUATOR is None or _WORKER_CONFIG is None:
        logger.error("Worker evaluator or config not initialized!")
        return ParallelEvaluationResult(fitness=(0.0,))

    try:
        fitness = _WORKER_EVALUATOR.evaluate(individual, _WORKER_CONFIG)
        report = _WORKER_EVALUATOR.get_last_evaluation_report()
        # 重い import 連鎖を避けるため必要時のみロード
        from app.services.auto_strategy.core.fitness.fitness_sharing_vectorizer import (
            build_behavior_profile,
        )

        behavior_summary = build_behavior_profile(
            fitness_values=fitness,
            evaluation_report=report,
        )
        return ParallelEvaluationResult(
            fitness=fitness,
            behavior_summary=behavior_summary,
        )
    except Exception as e:
        logger.error(f"Evaluation error in worker: {e}")
        return ParallelEvaluationResult(fitness=(0.0,))
