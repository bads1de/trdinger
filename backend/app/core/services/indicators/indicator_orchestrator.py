"""
テクニカル指標統合サービス

Numpyベースの指標計算関数を呼び出し、結果を整形する責務を担います。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple

from .config import indicator_registry, IndicatorConfig

logger = logging.getLogger(__name__)


class TechnicalIndicatorService:
    """テクニカル指標統合サービス"""

    def __init__(self):
        """サービスを初期化"""
        self.registry = indicator_registry

    def _get_indicator_config(self, indicator_type: str) -> IndicatorConfig:
        """
        指標設定を取得
        """
        config = self.registry.get_indicator_config(indicator_type)
        if not config or not config.adapter_function:
            supported = [
                name
                for name, cfg in self.registry._configs.items()
                if cfg.adapter_function
            ]
            raise ValueError(
                f"サポートされていない、またはアダプターが設定されていない指標タイプです: {indicator_type}. "
                f"サポート対象: {supported}"
            )
        return config

    def calculate_indicator(
        self, df: pd.DataFrame, indicator_type: str, params: Dict[str, Any]
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        指定された指標を計算します。

        Args:
            df: OHLCVデータを含むPandas DataFrame
            indicator_type: 計算する指標のタイプ (例: "RSI", "MACD")
            params: 指標計算に必要なパラメータ

        Returns:
            計算結果 (numpy配列または配列のタプル)
        """
        logger.info(f"TechnicalIndicatorService: {indicator_type}の計算開始")

        config = self._get_indicator_config(indicator_type)
        indicator_func = config.adapter_function

        logger.info(
            f"指標設定取得完了: {indicator_type}, 必要データ: {config.required_data}"
        )

        assert (
            indicator_func is not None
        ), "Adapter function cannot be None at this point."

        # 必要なデータをDataFrameからNumpy配列として抽出
        # backtesting.pyは大文字カラム名（Close, Open等）を使用するため、
        # 小文字の設定を大文字に変換して対応
        required_data = {}
        for data_key in config.required_data:
            # カラム名の大文字小文字を適切に処理
            actual_column = self._resolve_column_name(df, data_key)
            if actual_column is None:
                raise ValueError(
                    f"必要なカラム '{data_key}' がDataFrameにありません。利用可能なカラム: {list(df.columns)}"
                )

            # データキーを適切な関数パラメータ名にマッピング
            param_name = self._map_data_key_to_param(indicator_type, data_key)
            required_data[param_name] = df[actual_column].to_numpy()
            logger.info(
                f"データ抽出完了: {data_key} -> {actual_column} -> {param_name}, 長さ: {len(required_data[param_name])}"
            )

        # パラメータとデータを結合して関数を呼び出し
        all_args = {**required_data, **params}
        logger.info(
            f"関数呼び出し準備完了: {indicator_type}, 引数: {list(all_args.keys())}"
        )

        try:
            result = indicator_func(**all_args)
            logger.info(f"指標計算成功: {indicator_type}, 結果タイプ: {type(result)}")
            return result
        except Exception as e:
            logger.error(f"指標関数呼び出しエラー {indicator_type}: {e}", exc_info=True)
            raise

    def _resolve_column_name(self, df: pd.DataFrame, data_key: str) -> Optional[str]:
        """
        データフレームから適切なカラム名を解決

        Args:
            df: データフレーム
            data_key: 探すカラム名（小文字）

        Returns:
            実際のカラム名（見つからない場合はNone）
        """
        # 直接一致をチェック
        if data_key in df.columns:
            return data_key

        # 大文字小文字を変換してチェック
        capitalized_key = data_key.capitalize()
        if capitalized_key in df.columns:
            return capitalized_key

        # 全て大文字でチェック
        upper_key = data_key.upper()
        if upper_key in df.columns:
            return upper_key

        # 全て小文字でチェック
        lower_key = data_key.lower()
        if lower_key in df.columns:
            return lower_key

        return None

    def _map_data_key_to_param(self, indicator_type: str, data_key: str) -> str:
        """
        データキーを適切な関数パラメータ名にマッピング

        Args:
            indicator_type: 指標タイプ（例: "RSI", "MACD"）
            data_key: データキー（例: "close", "high", "low"）

        Returns:
            関数パラメータ名
        """
        # 単一データ系指標（RSI, SMA, EMA等）は"close"を"data"にマッピング
        single_data_indicators = [
            "RSI",
            "SMA",
            "EMA",
            "WMA",
            "DEMA",
            "TEMA",
            "TRIMA",
            "KAMA",
            "MAMA",
            "T3",
            "MACD",
            "MACDEXT",
            "MACDFIX",
            "BB",
        ]

        if indicator_type in single_data_indicators and data_key == "close":
            return "data"

        # その他の指標は元のキー名を使用（high, low, close等）
        return data_key

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
