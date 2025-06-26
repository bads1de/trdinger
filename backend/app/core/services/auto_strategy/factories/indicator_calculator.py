"""
指標計算機

指標の計算ロジックを専門に担当するモジュール。
IndicatorInitializerから計算機能を分離し、責務を明確化します。
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional

from app.core.services.indicators.adapters.trend_adapter import TrendAdapter
from app.core.services.indicators.adapters.momentum_adapter import MomentumAdapter
from app.core.services.indicators.adapters.volatility_adapter import (
    VolatilityAdapter,
)
from app.core.services.indicators.adapters.volume_adapter import VolumeAdapter

# PriceTransformAdapterは使用しない（オートストラテジー用10個の指標のみ）

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """
    指標計算を担当するクラス
    """

    def __init__(self):
        """初期化"""
        self.indicator_config = self._setup_indicator_config()
        self.indicator_adapters = self._setup_indicator_adapters()
        self._current_data = None

    def _setup_indicator_adapters(self) -> Dict[str, Any]:
        """指標アダプターのマッピングを設定"""
        adapters = {}

        if TrendAdapter:
            adapters.update(
                {
                    "SMA": TrendAdapter.sma,
                    "EMA": TrendAdapter.ema,
                }
            )

        if MomentumAdapter:
            adapters.update(
                {
                    "RSI": MomentumAdapter.rsi,
                    "STOCH": MomentumAdapter.stochastic,
                    "CCI": MomentumAdapter.cci,
                    "ADX": MomentumAdapter.adx,
                    "MACD": MomentumAdapter.macd,
                }
            )

        if VolatilityAdapter:
            adapters.update(
                {
                    "ATR": VolatilityAdapter.atr,
                    "BB": VolatilityAdapter.bollinger_bands,
                }
            )

        if VolumeAdapter:
            adapters.update(
                {
                    "OBV": VolumeAdapter.obv,
                }
            )

        # PriceTransformAdapterは使用しない（オートストラテジー用10個の指標のみ）

        return adapters

    def _setup_indicator_config(self) -> Dict[str, Dict[str, Any]]:
        """指標設定マッピングを設定（オートストラテジー用10個の指標のみ）"""
        return {
            # トレンド系指標（4個）
            "SMA": {
                "adapter_function": TrendAdapter.sma,
                "required_data": ["close"],
                "parameters": {"period": {"default": 14, "param_key": "period"}},
                "result_type": "single",
                "name_format": "{indicator}_{period}",
            },
            "EMA": {
                "adapter_function": TrendAdapter.ema,
                "required_data": ["close"],
                "parameters": {"period": {"default": 14, "param_key": "period"}},
                "result_type": "single",
                "name_format": "{indicator}_{period}",
            },
            "MACD": {
                "adapter_function": MomentumAdapter.macd,
                "required_data": ["close"],
                "parameters": {
                    "fast_period": {"default": 12, "param_key": "fast"},
                    "slow_period": {"default": 26, "param_key": "slow"},
                    "signal_period": {"default": 9, "param_key": "signal"},
                },
                "result_type": "complex",
                "result_handler": "macd_handler",
                "name_format": "{indicator}_{fast_period}",
            },
            "BB": {
                "adapter_function": VolatilityAdapter.bollinger_bands,
                "required_data": ["close"],
                "parameters": {"period": {"default": 20, "param_key": "period"}},
                "result_type": "complex",
                "result_handler": "bb_handler",
                "name_format": "BB_MIDDLE_{period}",
            },
            # モメンタム系指標（4個）
            "RSI": {
                "adapter_function": MomentumAdapter.rsi,
                "required_data": ["close"],
                "parameters": {"period": {"default": 14, "param_key": "period"}},
                "result_type": "single",
                "name_format": "{indicator}_{period}",
            },
            "STOCH": {
                "adapter_function": MomentumAdapter.stochastic,
                "required_data": ["high", "low", "close"],
                "parameters": {"period": {"default": 14, "param_key": "period"}},
                "result_type": "complex",
                "result_handler": "stoch_handler",
                "name_format": "{indicator}_{period}",
            },
            "CCI": {
                "adapter_function": MomentumAdapter.cci,
                "required_data": ["high", "low", "close"],
                "parameters": {"period": {"default": 14, "param_key": "period"}},
                "result_type": "single",
                "name_format": "{indicator}_{period}",
            },
            "ADX": {
                "adapter_function": MomentumAdapter.adx,
                "required_data": ["high", "low", "close"],
                "parameters": {"period": {"default": 14, "param_key": "period"}},
                "result_type": "single",
                "name_format": "{indicator}_{period}",
            },
            # ボラティリティ系指標（1個）
            "ATR": {
                "adapter_function": VolatilityAdapter.atr,
                "required_data": ["high", "low", "close"],
                "parameters": {"period": {"default": 14, "param_key": "period"}},
                "result_type": "single",
                "name_format": "{indicator}_{period}",
            },
            # 出来高系指標（1個）
            "OBV": {
                "adapter_function": VolumeAdapter.obv,
                "required_data": ["close", "volume"],
                "parameters": {},
                "result_type": "single",
                "name_format": "{indicator}",
            },
        }

    def calculate_indicator(
        self,
        indicator_type: str,
        parameters: Dict[str, Any],
        close_data: pd.Series,
        high_data: pd.Series,
        low_data: pd.Series,
        volume_data: pd.Series,
        open_data: Optional[pd.Series] = None,
    ) -> tuple:
        """
        指標を計算
        """
        try:
            # データを一時的に保存（将来の拡張用）
            if open_data is not None:
                self._current_data = pd.DataFrame(
                    {
                        "open": open_data,
                        "high": high_data,
                        "low": low_data,
                        "close": close_data,
                        "volume": volume_data,
                    }
                )

            # 設定マッピングを使用した統一処理
            if indicator_type in self.indicator_config:
                return self._calculate_from_config(
                    indicator_type,
                    parameters,
                    close_data,
                    high_data,
                    low_data,
                    volume_data,
                )

            # アダプターシステムを使用した処理
            elif indicator_type in self.indicator_adapters:
                return self._calculate_from_adapter(
                    indicator_type,
                    parameters,
                    close_data,
                    high_data,
                    low_data,
                    volume_data,
                )

            else:
                logger.warning(f"未対応の指標タイプ: {indicator_type}")
                return None, None

        except Exception as e:
            logger.error(f"指標計算エラー ({indicator_type}): {e}")
            return None, None

    def _calculate_from_config(
        self,
        indicator_type: str,
        parameters: Dict[str, Any],
        close_data: pd.Series,
        high_data: pd.Series,
        low_data: pd.Series,
        volume_data: pd.Series,
    ) -> tuple:
        """設定マッピングベースの指標計算"""
        config = self.indicator_config[indicator_type]

        data_args = self._prepare_data_for_indicator(
            config, close_data, high_data, low_data, volume_data
        )
        param_args = self._prepare_parameters_for_indicator(config, parameters)
        result = self._call_adapter_function(
            config["adapter_function"], data_args, param_args
        )

        if config["result_type"] == "complex":
            return self._handle_complex_result(
                result, config, indicator_type, parameters
            )
        else:
            indicator_name = self._generate_indicator_name(
                config, indicator_type, parameters
            )
            return result, indicator_name

    def _calculate_from_adapter(
        self,
        indicator_type: str,
        parameters: Dict[str, Any],
        close_data: pd.Series,
        high_data: pd.Series,
        low_data: pd.Series,
        volume_data: pd.Series,
    ) -> tuple:
        """アダプターシステムでの指標計算"""
        try:
            adapter_function = self.indicator_adapters[indicator_type]

            # オートストラテジー用10個の指標のみ対応
            if indicator_type in ["SMA", "EMA", "RSI"]:
                period = parameters.get("period", 14)
                result = adapter_function(close_data, period)
                indicator_name = f"{indicator_type}_{period}"
            elif indicator_type == "STOCH":
                period = parameters.get("period", 14)
                result = adapter_function(high_data, low_data, close_data, period)
                indicator_name = f"{indicator_type}_{period}"
            elif indicator_type in ["CCI", "ADX"]:
                period = parameters.get("period", 14)
                result = adapter_function(high_data, low_data, close_data, period)
                indicator_name = f"{indicator_type}_{period}"
            elif indicator_type == "ATR":
                period = parameters.get("period", 14)
                result = adapter_function(high_data, low_data, close_data, period)
                indicator_name = f"{indicator_type}_{period}"
            elif indicator_type == "OBV":
                result = adapter_function(close_data, volume_data)
                indicator_name = indicator_type
            else:
                logger.warning(f"未対応の指標タイプ: {indicator_type}")
                return None, None

            return result, indicator_name

        except Exception as e:
            logger.error(f"アダプター指標計算エラー ({indicator_type}): {e}")
            return None, None

    def _prepare_data_for_indicator(
        self,
        config: Dict[str, Any],
        close_data: pd.Series,
        high_data: pd.Series,
        low_data: pd.Series,
        volume_data: pd.Series,
    ) -> list:
        """指標に必要なデータを準備"""
        data_map = {
            "close": close_data,
            "high": high_data,
            "low": low_data,
            "volume": volume_data,
        }
        required_data = config["required_data"]
        return [data_map[data_type] for data_type in required_data]

    def _prepare_parameters_for_indicator(
        self, config: Dict[str, Any], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """指標に必要なパラメータを準備"""
        prepared_params = {}
        param_config = config.get("parameters", {})
        for _, param_info in param_config.items():
            param_key = param_info["param_key"]
            default_value = param_info["default"]
            prepared_params[param_key] = parameters.get(param_key, default_value)
        return prepared_params

    def _call_adapter_function(
        self, adapter_function, data_args: list, param_args: Dict[str, Any]
    ) -> Any:
        """アダプター関数を呼び出し"""
        if param_args:
            return adapter_function(*data_args, **param_args)
        else:
            return adapter_function(*data_args)

    def _generate_indicator_name(
        self, config: Dict[str, Any], indicator_type: str, parameters: Dict[str, Any]
    ) -> str:
        """指標名を生成"""
        name_format = config["name_format"]
        format_params = {"indicator": indicator_type}
        param_config = config.get("parameters", {})
        for param_name, param_info in param_config.items():
            param_key = param_info["param_key"]
            default_value = param_info["default"]
            format_params[param_name] = parameters.get(param_name, default_value)
            format_params[param_key] = parameters.get(param_name, default_value)
        return name_format.format(**format_params)

    def _handle_complex_result(
        self,
        result: Any,
        config: Dict[str, Any],
        indicator_type: str,
        parameters: Dict[str, Any],
    ) -> tuple:
        """複合指標の結果を処理"""
        try:
            result_handler = config.get("result_handler")
            if result_handler == "macd_handler":
                if isinstance(result, dict) and "macd_line" in result:
                    indicator_name = self._generate_indicator_name(
                        config, indicator_type, parameters
                    )
                    return result["macd_line"], indicator_name
            elif result_handler == "bb_handler":
                if isinstance(result, dict) and "middle" in result:
                    indicator_name = self._generate_indicator_name(
                        config, indicator_type, parameters
                    )
                    return result["middle"], indicator_name

            # デフォルト処理
            indicator_name = self._generate_indicator_name(
                config, indicator_type, parameters
            )
            return result, indicator_name
        except Exception as e:
            logger.error(f"複合指標処理エラー ({indicator_type}): {e}")
            return None, None
