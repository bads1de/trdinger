"""
テクニカル指標統合サービス

Numpyベースの指標計算関数を呼び出し、結果を整形する責務を担います。
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import IndicatorConfig, indicator_registry
from .utils import ensure_numpy_array, normalize_data_for_trig

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

        assert (
            indicator_func is not None
        ), "Adapter function cannot be None at this point."

        # 必要なデータをDataFrameからNumpy配列として抽出
        # backtesting.pyは大文字カラム名（Close, Open等）を使用するため、
        # 小文字の設定を大文字に変換して対応
        required_data = {}
        for data_key in config.required_data:
            # カラム名の大文字小文字を適切に処理
            actual_column = self._resolve_column_name(df, data_key, indicator_type)
            if actual_column is None:
                raise ValueError(
                    f"必要なカラム '{data_key}' がDataFrameにありません。利用可能なカラム: {list(df.columns)}"
                )

            # データキーを適切な関数パラメータ名にマッピング
            param_name = self._map_data_key_to_param(indicator_type, data_key)
            array = df[actual_column].to_numpy()
            required_data[param_name] = ensure_numpy_array(array)

        # 必要に応じて入力データを正規化
        if config.needs_normalization:
            for key, data_array in required_data.items():
                required_data[key] = normalize_data_for_trig(data_array)

        # パラメータとデータを結合して関数を呼び出し
        all_args = {**required_data, **params}

        try:
            result = indicator_func(**all_args)
            return result
        except Exception as e:
            logger.error(f"指標関数呼び出しエラー {indicator_type}: {e}", exc_info=True)
            raise

    def _resolve_column_name(
        self, df: pd.DataFrame, data_key: str, indicator_type: Optional[str] = None
    ) -> Optional[str]:
        """
        データフレームから適切なカラム名を解決

        Args:
            df: データフレーム
            data_key: 探すカラム名（小文字）
            indicator_type: 指標タイプ（オプション）

        Returns:
            実際のカラム名（見つからない場合はNone）
        """
        # 特別なマッピング（指標タイプを考慮）
        special_mappings = {
            "data0": "high",  # デフォルトで高値を使用
            "data1": "low",  # デフォルトで安値を使用
        }

        # open_dataの特別な処理
        if data_key == "open_data":
            # open_dataは常に"open"カラムを参照する
            data_key = "open"

        if data_key in special_mappings:
            data_key = special_mappings[data_key]

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
