"""
並列評価ワーカーモジュール

Windowsのmultiprocessing (spawn) に対応するため、
トップレベル関数として評価ロジックを定義します。
"""

import logging
from typing import Any

import pandas as pd

from app.services.auto_strategy.config import objective_registry
from app.services.auto_strategy.config.constants import PENALTY_FITNESS_MAGNITUDE

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


def _default_fitness_for_config(config: Any) -> tuple[float, ...]:
    """評価失敗時のデフォルトフィットネス（方向を考慮したペナルティ値）を生成する。

    マルチ目的構成（objectives が2個以上）でも長さ不一致で
    クラッシュしないようにする。0.0 を返すと最小化目的で最良スコアに
    なり失敗個体が選択を勝ち抜けるため、目的ごとに方向を考慮した
    ペナルティ値（objective_registry.build_penalty_values 準拠）を返す。
    """
    try:
        objectives = list(getattr(config, "objectives", None) or [])
    except Exception:
        objectives = []
    if not objectives:
        # 設定が壊れて目的が取得できない場合も、失敗個体が有利に
        # 扱われないよう最大化方向のペナルティを返す。
        return (-PENALTY_FITNESS_MAGNITUDE,)
    return objective_registry.build_penalty_values(objectives)


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


def worker_evaluate_individual(
    individual: Any,
    dynamic_scalars: dict[str, float] | None = None,
) -> ParallelEvaluationResult:
    """
    個体評価関数（ワーカープロセス内で実行）

    Args:
        individual: 評価対象の個体
        dynamic_scalars: メインプロセスで世代ごとに更新される
            ``objective_dynamic_scalars``。プール起動時に pickle された
            設定は古いままなので、評価のたびに最新値を受け取って反映する。

    Returns:
        フィットネス値のタプル
    """
    if _WORKER_EVALUATOR is None or _WORKER_CONFIG is None:
        logger.error("Worker evaluator or config not initialized!")
        return ParallelEvaluationResult(
            fitness=_default_fitness_for_config(_WORKER_CONFIG)
        )

    if dynamic_scalars:
        try:
            _WORKER_CONFIG.objective_dynamic_scalars = dict(dynamic_scalars)
        except Exception as e:
            logger.debug("dynamic_scalars の反映に失敗しました: %s", e)

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
        return ParallelEvaluationResult(
            fitness=_default_fitness_for_config(_WORKER_CONFIG)
        )
