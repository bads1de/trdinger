"""
テクニカル指標統合サービス

pandas-taを直接活用し、冗長なラッパーを削除した効率的な実装。
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

from .config import IndicatorConfig, indicator_registry

logger = logging.getLogger(__name__)


class TechnicalIndicatorService:
    """テクニカル指標統合サービス"""

    def __init__(self):
        """サービスを初期化"""
        self.registry = indicator_registry

    def _get_indicator_config(self, indicator_type: str) -> IndicatorConfig:
        """指標設定を取得"""
        config = self.registry.get_indicator_config(indicator_type)
        if not config:
            raise ValueError(f"サポートされていない指標タイプ: {indicator_type}")
        return config

    def calculate_indicator(
        self, df: pd.DataFrame, indicator_type: str, params: Dict[str, Any]
    ) -> Union[np.ndarray, tuple]:
        """
        指定された指標を計算

        pandas-taを直接使用し、複雑なマッピング処理を削除。
        """
        try:
            # 基本的なパラメータ検証
            if indicator_type in ["SMA", "EMA", "WMA", "RSI"]:
                length = params.get("length", params.get("period", 14))
                if length <= 0:
                    raise ValueError(f"{indicator_type}: 期間は正の値が必要: {length}")

            # pandas-taを直接使用
            if indicator_type == "RSI":
                length = params.get("length", params.get("period", 14))
                return ta.rsi(df["Close"], length=length).values

            elif indicator_type == "SMA":
                length = params.get("length", params.get("period", 20))
                return ta.sma(df["Close"], length=length).values

            elif indicator_type == "EMA":
                length = params.get("length", params.get("period", 20))
                return ta.ema(df["Close"], length=length).values

            elif indicator_type == "MACD":
                fast = params.get("fast", 12)
                slow = params.get("slow", 26)
                signal = params.get("signal", 9)
                result = ta.macd(df["Close"], fast=fast, slow=slow, signal=signal)
                return (
                    result.iloc[:, 0].values,  # MACD
                    result.iloc[:, 1].values,  # Signal
                    result.iloc[:, 2].values,  # Histogram
                )

            elif indicator_type == "BBANDS":
                length = params.get("length", params.get("period", 20))
                std = params.get("std", 2.0)
                result = ta.bbands(df["Close"], length=length, std=std)
                return (
                    result.iloc[:, 0].values,  # Upper
                    result.iloc[:, 1].values,  # Middle
                    result.iloc[:, 2].values,  # Lower
                )

            # その他の指標は従来の方法で処理
            config = self._get_indicator_config(indicator_type)
            if config.adapter_function:
                return self._calculate_with_adapter(df, indicator_type, params, config)
            else:
                raise ValueError(f"指標 {indicator_type} の実装が見つかりません")

        except Exception as e:
            logger.error(f"指標計算エラー {indicator_type}: {e}")
            raise

    def _calculate_with_adapter(
        self,
        df: pd.DataFrame,
        indicator_type: str,
        params: Dict[str, Any],
        config: IndicatorConfig,
    ):
        """アダプター関数を使用した指標計算（後方互換性用）"""
        # 従来のロジックを簡素化して維持
        required_data = {}
        for data_key in config.required_data:
            column_name = self._resolve_column_name(df, data_key)
            if column_name:
                required_data[data_key] = df[column_name]

        # パラメータ正規化
        from .parameter_manager import normalize_params

        converted_params = normalize_params(indicator_type, params, config)

        # 関数呼び出し
        all_args = {**required_data, **converted_params}
        return config.adapter_function(**all_args)

    def _resolve_column_name(self, df: pd.DataFrame, data_key: str) -> Optional[str]:
        """
        データフレームから適切なカラム名を解決（簡素化版）
        """
        # 特別なマッピング
        key_mappings = {
            "open_data": "open",
            "open_": "open",
            "data0": "high",
            "data1": "low",
        }

        actual_key = key_mappings.get(data_key, data_key)

        # 大文字小文字のバリエーションをチェック
        for variant in [
            actual_key.capitalize(),
            actual_key.upper(),
            actual_key.lower(),
        ]:
            if variant in df.columns:
                return variant

        return None

    def get_supported_indicators(self) -> Dict[str, Any]:
        """
        サポートされている指標の情報を取得

        Returns:
            サポート指標の情報
        """
        infos = {}
        for name, config in self.registry._configs.items():
            if not config.adapter_function:
                continue
            infos[name] = {
                "parameters": config.get_parameter_ranges(),
                "result_type": config.result_type.value,
                "required_data": config.required_data,
                "scale_type": config.scale_type.value if config.scale_type else None,
            }
        return infos
