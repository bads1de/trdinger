"""
IndividualEvaluator 用のバックテスト実行設定ビルダー。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.services.auto_strategy.config.ml_filter_settings import (
    resolve_ml_gate_settings,
)
from app.services.backtest.config.builders import build_execution_config

logger = logging.getLogger(__name__)


class RunConfigBuilder:
    """戦略評価用の backtest 実行設定を構築する。"""

    def build_run_config(
        self,
        gene: Any,
        backtest_config: Dict[str, Any],
        config: Any,
    ) -> Optional[Dict[str, Any]]:
        """バックテスト実行用の設定辞書を構築する。"""
        try:
            ml_gate_settings = resolve_ml_gate_settings(config)
            strategy_parameters = {
                "strategy_gene": gene,
                "volatility_gate_enabled": ml_gate_settings.enabled,
                "volatility_model_path": ml_gate_settings.model_path,
                "ml_filter_enabled": ml_gate_settings.enabled,
                "ml_model_path": ml_gate_settings.model_path,
                "enable_early_termination": bool(
                    getattr(config, "enable_early_termination", False)
                ),
                "early_termination_max_drawdown": getattr(
                    config, "early_termination_max_drawdown", None
                ),
                "early_termination_min_trades": getattr(
                    config, "early_termination_min_trades", None
                ),
                "early_termination_min_trade_check_progress": getattr(
                    config,
                    "early_termination_min_trade_check_progress",
                    0.5,
                ),
                "early_termination_trade_pace_tolerance": getattr(
                    config,
                    "early_termination_trade_pace_tolerance",
                    0.5,
                ),
                "early_termination_min_expectancy": getattr(
                    config, "early_termination_min_expectancy", None
                ),
                "early_termination_expectancy_min_trades": getattr(
                    config,
                    "early_termination_expectancy_min_trades",
                    5,
                ),
                "early_termination_expectancy_progress": getattr(
                    config,
                    "early_termination_expectancy_progress",
                    0.6,
                ),
            }
            evaluation_start = backtest_config.get("_evaluation_start")
            if evaluation_start is not None:
                strategy_parameters["evaluation_start"] = evaluation_start

            config_dict = build_execution_config(
                backtest_config,
                strategy_name=f"GA_Individual_{str(getattr(gene, 'id', 'unknown'))[:8]}",
                strategy_config={
                    "strategy_type": "GENERATED_GA",
                    "parameters": strategy_parameters,
                },
            )
            config_dict["_skip_validation"] = True
            return config_dict
        except Exception as e:
            logger.error(f"バックテスト設定生成エラー: {e}")
            return None

    @staticmethod
    def inject_external_objects(
        run_config: Dict[str, Any],
        *,
        minute_data: Any = None,
    ) -> None:
        """実行設定へ外部オブジェクトを注入する。"""
        if minute_data is not None:
            run_config["strategy_config"]["parameters"]["minute_data"] = minute_data
