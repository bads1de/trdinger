"""
並列評価ワーカーモジュール

Windowsのmultiprocessing (spawn) に対応するため、
トップレベル関数として評価ロジックを定義します。
"""

import logging
from typing import Any, Tuple, Optional, Dict

# 遅延インポートなどで循環参照を避ける
from app.services.backtest.backtest_service import BacktestService
from .individual_evaluator import IndividualEvaluator

logger = logging.getLogger(__name__)

# ワーカープロセスごとのグローバル変数
_WORKER_EVALUATOR: Optional[IndividualEvaluator] = None
_WORKER_CONFIG: Optional[Any] = None

def initialize_worker_process(
    backtest_config: Dict[str, Any], 
    ga_config: Any,
    shared_data: Optional[Dict[str, Any]] = None
):
    """
    ワーカープロセスの初期化関数
    
    Args:
        backtest_config: バックテスト設定
        ga_config: GA設定
        shared_data: 親プロセスから共有されるデータ（OHLCVなど）
    """
    global _WORKER_EVALUATOR, _WORKER_CONFIG
    
    try:
        # 1. BacktestServiceの初期化（これによりDB接続も各プロセスで確立）
        # 引数なしで初期化すると、デフォルトのリポジトリとDBセッションが作成される
        backtest_service = BacktestService()
        
        # 2. Evaluatorの初期化
        evaluator = IndividualEvaluator(backtest_service)
        
        # 3. 設定の適用
        evaluator.set_backtest_config(backtest_config)
        
        # 4. 共有データの適用（DBアクセス削減のため）
        if shared_data:
            if "main_data" in shared_data:
                evaluator._cached_data = shared_data["main_data"]
            if "minute_data" in shared_data:
                evaluator._cached_minute_data = shared_data["minute_data"]
        
        _WORKER_EVALUATOR = evaluator
        _WORKER_CONFIG = ga_config
        
        # logger.info(f"Worker process initialized. PID: {os.getpid()}")
        
    except Exception as e:
        logger.error(f"Worker initialization failed: {e}")
        raise

def worker_evaluate_individual(individual: Any) -> Tuple[float, ...]:
    """
    個体評価関数（ワーカープロセス内で実行）
    
    Args:
        individual: 評価対象の個体
    
    Returns:
        フィットネス値のタプル
    """
    global _WORKER_EVALUATOR, _WORKER_CONFIG
    
    if _WORKER_EVALUATOR is None:
        logger.error("Worker evaluator not initialized!")
        return (0.0,)
        
    try:
        return _WORKER_EVALUATOR.evaluate(individual, _WORKER_CONFIG)
    except Exception as e:
        logger.error(f"Evaluation error in worker: {e}")
        return (0.0,)
