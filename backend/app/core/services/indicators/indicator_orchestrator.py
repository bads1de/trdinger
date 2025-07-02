"""
テクニカル指標統合サービス

Numpyベースの指標計算関数を呼び出し、結果を整形する責務を担います。
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple

from .config import indicator_registry, IndicatorConfig, IndicatorResultType

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
        config = self._get_indicator_config(indicator_type)
        indicator_func = config.adapter_function

        # 必要なデータをDataFrameからNumpy配列として抽出
        required_data = {}
        for data_key in config.required_data:
            if data_key not in df.columns:
                raise ValueError(f"必要なカラム '{data_key}' がDataFrameにありません。")
            required_data[data_key] = df[data_key].to_numpy()

        # パラメータとデータを結合して関数を呼び出し
        all_args = {**required_data, **params}

        return indicator_func(**all_args)

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
