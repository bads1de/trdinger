"""
パラメータ生成ユーティリティ

指標のパラメータ生成ロジックを共通化し、重複を削除します。
新しいIndicatorParameterManagerシステムへの移行中です。
"""

import random
import logging
from typing import Dict, Any

from app.core.services.indicators.parameter_manager import IndicatorParameterManager
from app.core.services.indicators.config.indicator_config import indicator_registry

logger = logging.getLogger(__name__)


class ParameterGenerator:
    """パラメータ生成器"""

    @staticmethod
    def generate_period_parameter(
        min_period: int = 5, max_period: int = 50
    ) -> Dict[str, Any]:
        """期間パラメータを生成"""
        return {"period": random.randint(min_period, max_period)}

    @staticmethod
    def generate_fast_slow_periods(
        fast_min: int = 5, fast_max: int = 20, slow_min: int = 20, slow_max: int = 50
    ) -> Dict[str, Any]:
        """高速・低速期間パラメータを生成"""
        fast_period = random.randint(fast_min, fast_max)
        slow_period = random.randint(slow_min, slow_max)
        # 高速期間が低速期間より小さくなるように調整
        if fast_period >= slow_period:
            slow_period = fast_period + random.randint(5, 15)
        return {"fast_period": fast_period, "slow_period": slow_period}

    @staticmethod
    def generate_macd_parameters() -> Dict[str, Any]:
        """MACDパラメータを生成"""
        fast_period = random.randint(5, 20)
        slow_period = random.randint(20, 50)
        signal_period = random.randint(5, 15)
        return {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "signal_period": signal_period,
        }

    @staticmethod
    def generate_bollinger_bands_parameters() -> Dict[str, Any]:
        """ボリンジャーバンドパラメータを生成"""
        return {"period": random.randint(15, 25), "std_dev": random.uniform(1.5, 2.5)}

    @staticmethod
    def generate_stochastic_parameters() -> Dict[str, Any]:
        """ストキャスティクスパラメータを生成"""
        return {
            "k_period": random.randint(10, 20),
            "d_period": random.randint(3, 7),
            "slow_k_period": random.randint(3, 5),
        }


class ThresholdGenerator:
    """閾値生成器"""

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


# 指標タイプ別のパラメータ生成マッピング（オートストラテジー用10個の指標のみ）
PARAMETER_GENERATORS = {
    # 期間のみのパラメータ
    "period_only": [
        "SMA",  # Simple Moving Average
        "EMA",  # Exponential Moving Average
        "RSI",  # Relative Strength Index
        "CCI",  # Commodity Channel Index
        "ADX",  # Average Directional Movement Index
        "ATR",  # Average True Range
    ],
    # 特別なパラメータ生成が必要な指標
    "special": {
        "MACD": ParameterGenerator.generate_macd_parameters,
        "BB": ParameterGenerator.generate_bollinger_bands_parameters,
        "STOCH": ParameterGenerator.generate_stochastic_parameters,
    },
    # パラメータが不要な指標
    "no_params": [
        "OBV",  # On Balance Volume
    ],
}


def generate_indicator_parameters(indicator_type: str) -> Dict[str, Any]:
    """
    指標タイプに応じたパラメータを生成

    新しいIndicatorParameterManagerシステムを使用します。
    レジストリに登録されていない指標は従来のロジックにフォールバックします。
    """
    try:
        # 新しいシステムを試行
        config = indicator_registry.get_config(indicator_type)
        if config:
            manager = IndicatorParameterManager()
            return manager.generate_parameters(indicator_type, config)
    except Exception as e:
        logger.debug(
            f"新システムでの生成に失敗、フォールバックを使用: {indicator_type}, {e}"
        )

    # フォールバック: 従来のロジック
    if indicator_type in PARAMETER_GENERATORS["no_params"]:
        return {}
    elif indicator_type in PARAMETER_GENERATORS["special"]:
        return PARAMETER_GENERATORS["special"][indicator_type]()
    elif indicator_type in PARAMETER_GENERATORS["period_only"]:
        return ParameterGenerator.generate_period_parameter()
    else:
        # デフォルトは期間パラメータ
        return ParameterGenerator.generate_period_parameter()
