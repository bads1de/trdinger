"""
IndividualEvaluator 用のバックテスト実行設定ビルダー。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

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
            config_dict = backtest_config.copy()
            strategy_parameters = {
                "strategy_gene": gene,
                "ml_filter_enabled": config.ml_filter_enabled,
                "ml_model_path": config.ml_model_path,
            }

            config_dict["strategy_config"] = {
                "strategy_type": "GENERATED_GA",
                "parameters": strategy_parameters,
            }
            gene_id = getattr(gene, "id", "unknown")[:8]
            config_dict["strategy_name"] = f"GA_Individual_{gene_id}"
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
