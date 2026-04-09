"""
IndividualEvaluator 用のバックテスト実行設定ビルダー。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

from app.services.auto_strategy.config.ga.nested_configs import (
    resolve_early_termination_settings,
)
from app.services.auto_strategy.config.helpers import (
    normalize_ml_gate_fields,
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
        defaults: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """バックテスト実行用の設定辞書を構築する。"""
        try:
            early_termination_settings = resolve_early_termination_settings(config)
            strategy_parameters = {
                "strategy_gene": gene,
            }
            strategy_parameters.update(normalize_ml_gate_fields(config))
            strategy_parameters.update(early_termination_settings.to_strategy_params())
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
                defaults=defaults,
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
