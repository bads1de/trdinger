"""
テクニカル指標統合サービス（簡素化版）

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
    """テクニカル指標統合サービス（簡素化版）"""

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
        指定された指標を計算（簡素化版）

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
        from .parameter_normalizer import normalize_params

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

    def _map_data_key_to_param(self, indicator_type: str, data_key: str) -> str:
        """
        データキーを関数パラメータ名にマッピング

        Args:
            indicator_type: 指標タイプ
            data_key: データキー（'close', 'high', 'low', 'open', 'volume'）

        Returns:
            関数パラメータ名
        """
        # 基本的なマッピング
        basic_mapping = {
            "close": "close",  # デフォルトはclose（関数固有でdataにマップ）
            "high": "high",
            "low": "low",
            "open": "open_data",  # オープンはopen_data名（avgprice対応）
            "volume": "volume",
        }

        # 指標固有のマッピング（必要に応じて拡張）
        indicator_specific_mapping = {
            "ATR": {"high": "high", "low": "low", "close": "close"},
            "STOCH": {"high": "high", "low": "low", "close": "close"},
            "WILLR": {"high": "high", "low": "low", "close": "close"},
            # 単一入力系
            "SMA": {"close": "data"},
            "EMA": {"close": "data"},
            "WMA": {"close": "data"},
            "TRIMA": {"close": "data"},
            "KAMA": {"close": "data"},
            "T3": {"close": "data"},
            "MA": {"close": "data"},
            "MIDPOINT": {"close": "data"},
            "RSI": {"close": "data"},
            "MACD": {"close": "data"},
            "MACDEXT": {"close": "data"},
            "MACDFIX": {"close": "data"},
            "PPO": {"close": "data"},
            "APO": {"close": "data"},
            "ROC": {"close": "data"},
            "ROCP": {"close": "data"},
            "ROCR": {"close": "data"},
            "ROCR100": {"close": "data"},
            "TRIX": {"close": "data"},
            # "HT_TRENDLINE": {"close": "data"},  # VALID_INDICATOR_TYPESに含まれていないため除去
            "STOCHRSI": {"close": "data"},
            # 追加モメンタム/ML系の単一入力はdataにマップ
            "QQE": {"close": "data"},
            "SMI": {"close": "data"},
            "KST": {"close": "data"},
            "STC": {"close": "data"},
            "ML_UP_PROB": {"close": "data"},
            "ML_DOWN_PROB": {"close": "data"},
            "ML_RANGE_PROB": {"close": "data"},
            # 追加: 本対応で新規追加した単一入力指標（VALID_INDICATOR_TYPESに含まれるもののみ）
            # "HMA": {"close": "data"},  # VALID_INDICATOR_TYPESに含まれていないため除去
            # "ZLMA": {"close": "data"},  # VALID_INDICATOR_TYPESに含まれていないため除去
            # "SWMA": {"close": "data"},  # VALID_INDICATOR_TYPESに含まれていないため除去
            # "ALMA": {"close": "data"},  # VALID_INDICATOR_TYPESに含まれていないため除去
            # "RMA": {"close": "data"},  # VALID_INDICATOR_TYPESに含まれていないため除去
            "TSI": {"close": "data"},
            "CFO": {"close": "data"},
            "CTI": {"close": "data"},
            "SMA_SLOPE": {"close": "data"},
            "PRICE_EMA_RATIO": {"close": "data"},
            "RSI_EMA_CROSS": {"close": "data"},
            # 新規追加の単一入力系
            "RMI": {"close": "data"},
            "DPO": {"close": "data"},
            # VWMA は close->data, volume->volume
            "VWMA": {"close": "data", "volume": "volume"},
            # RVGI は open_ を受け取る
            "RVGI": {
                "open_data": "open_",
                "high": "high",
                "low": "low",
                "close": "close",
            },
            # RVI も open_ を受け取る
            "RVI": {
                "open_data": "open_",
                "high": "high",
                "low": "low",
                "close": "close",
            },
            # BOP も open_ を受け取る
            "BOP": {
                "open_data": "open_",
                "high": "high",
                "low": "low",
                "close": "close",
            },
            # 統計系の単一入力関数
            "LINEARREG": {"close": "data"},
            "LINEARREG_SLOPE": {"close": "data"},
            "LINEARREG_ANGLE": {"close": "data"},
            "LINEARREG_INTERCEPT": {"close": "data"},
            "STDDEV": {"close": "data"},
            "VAR": {"close": "data"},
            "TSF": {"close": "data"},
            # 三角/数学変換系は data
            "ACOS": {"close": "data"},
            "ASIN": {"close": "data"},
            "ATAN": {"close": "data"},
            "COS": {"close": "data"},
            "COSH": {"close": "data"},
            "SIN": {"close": "data"},
            "SINH": {"close": "data"},
            "TAN": {"close": "data"},
            "TANH": {"close": "data"},
            "CEIL": {"close": "data"},
            "EXP": {"close": "data"},
            "FLOOR": {"close": "data"},
            "LN": {"close": "data"},
            "LOG10": {"close": "data"},
            "SQRT": {"close": "data"},
            # 高速ストキャスはstochで代用
            # pandas-taにはstochfが無い場合があるためSTOCHにフォールバック
            "STOCHF": {"high": "high", "low": "low", "close": "close"},
            # ULTOSC
            "ULTOSC": {
                "high": "high",
                "low": "low",
                "close": "close",
                "period1": "period1",
                "period2": "period2",
                "period3": "period3",
            },
            # Additional volume & momentum mappings
            "EOM": {"high": "high", "low": "low", "close": "close", "volume": "volume"},
            "KVO": {"high": "high", "low": "low", "close": "close", "volume": "volume"},
            "CMF": {"high": "high", "low": "low", "close": "close", "volume": "volume"},
            "VORTEX": {"high": "high", "low": "low", "close": "close"},
            # ボリンジャーバンド
            "BB": {"close": "data"},
            # 価格変換
            "AVGPRICE": {
                "open": "open_data",
                "high": "high",
                "low": "low",
                "close": "close",
            },
            "MEDPRICE": {"high": "high", "low": "low"},
            "TYPPRICE": {"high": "high", "low": "low", "close": "close"},
            "WCLPRICE": {"high": "high", "low": "low", "close": "close"},
            # Heikin Ashi
            "HA_CLOSE": {
                "open": "open_data",
                "high": "high",
                "low": "low",
                "close": "close",
            },
            "HA_OHLC": {
                "open": "open_data",
                "high": "high",
                "low": "low",
                "close": "close",
            },
        }

        # Config 側に param_map があればそれを最優先
        config = self._get_indicator_config(indicator_type)
        try:
            if config and getattr(config, "param_map", None):
                if data_key in config.param_map:
                    return config.param_map[data_key]
        except Exception:
            pass

        # 指標固有のマッピングがある場合はそれを使用
        if indicator_type in indicator_specific_mapping:
            specific_mapping = indicator_specific_mapping[indicator_type]
            if data_key in specific_mapping:
                return specific_mapping[data_key]

        # 基本マッピングを使用
        return basic_mapping.get(data_key.lower(), data_key)

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
