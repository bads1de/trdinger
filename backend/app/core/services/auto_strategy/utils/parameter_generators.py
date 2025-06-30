"""
パラメータ生成ユーティリティ

IndicatorParameterManagerを使用した統一されたパラメータ生成システム。
旧システムのParameterGeneratorクラスは廃止され、全てParameterManagerに統合されました。
"""

import random
import logging
from typing import Dict, Any

from app.core.services.indicators.parameter_manager import IndicatorParameterManager
from app.core.services.indicators.config.indicator_config import indicator_registry

logger = logging.getLogger(__name__)


class ThresholdGenerator:
    """閾値生成器（戦略条件用）"""

    @staticmethod
    def generate_percentage_threshold(condition_type: str = "entry") -> float:
        """パーセンテージ閾値を生成（0-100）"""
        if condition_type == "entry":
            return random.uniform(20, 80)
        else:
            return random.uniform(30, 70)

    @staticmethod
    def generate_oscillator_threshold(condition_type: str = "entry") -> float:
        """オシレーター閾値を生成（-100 to 100）"""
        if condition_type == "entry":
            return random.uniform(-80, 80)
        else:
            return random.uniform(-60, 60)

    @staticmethod
    def generate_price_ratio_threshold(condition_type: str = "entry") -> float:
        """価格比率閾値を生成（0.9-1.1）"""
        if condition_type == "entry":
            return random.uniform(0.95, 1.05)
        else:
            return random.uniform(0.98, 1.02)

    @staticmethod
    def generate_momentum_threshold(condition_type: str = "entry") -> float:
        """モメンタム閾値を生成"""
        if condition_type == "entry":
            return random.uniform(-10, 10)
        else:
            return random.uniform(-5, 5)

    @staticmethod
    def generate_volume_threshold(condition_type: str = "entry") -> float:
        """出来高閾値を生成"""
        if condition_type == "entry":
            return random.uniform(-1000, 1000)
        else:
            return random.uniform(-500, 500)


def generate_indicator_parameters(indicator_type: str) -> Dict[str, Any]:
    """
    指標タイプに応じたパラメータを生成

    IndicatorParameterManagerシステムを使用した統一されたパラメータ生成。
    """
    try:
        config = indicator_registry.get_indicator_config(indicator_type)
        if config:
            manager = IndicatorParameterManager()
            return manager.generate_parameters(indicator_type, config)
        else:
            logger.warning(f"指標 {indicator_type} の設定が見つかりません")
            return {}
    except Exception as e:
        logger.error(f"指標 {indicator_type} のパラメータ生成に失敗: {e}")
        return {}
