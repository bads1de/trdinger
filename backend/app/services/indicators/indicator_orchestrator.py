"""
テクニカル指標統合サービス

Numpyベースの指標計算関数を呼び出し、結果を整形する責務を担います。
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import inspect

from .config import IndicatorConfig, indicator_registry
from .utils import PandasTAError, normalize_data_for_trig

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
                raise PandasTAError(
                    f"必要なカラム '{data_key}' がDataFrameにありません。利用可能なカラム: {list(df.columns)}"
                )

            # データキーを適切な関数パラメータ名にマッピング
            param_name = self._map_data_key_to_param(indicator_type, data_key)
            # 型変換をなくし、Seriesを直接渡す
            required_data[param_name] = df[actual_column]

        # 必要に応じて入力データを正規化
        if config.needs_normalization:
            for key, data_series in required_data.items():
                # Seriesからnumpy配列に変換して正規化し、再度Seriesに戻す
                normalized_array = normalize_data_for_trig(data_series.to_numpy())
                required_data[key] = pd.Series(
                    normalized_array, index=data_series.index
                )

        # パラメータ名の変換（period -> length、ただし一部の指標は除外）
        # MAX/MIN/SUM は data 引数名を期待
        if indicator_type in ["MAX", "MIN", "SUM"]:
            if "close" in required_data:
                required_data = {**required_data}
                required_data["data"] = required_data.pop("close")

        # パラメータ正規化（period->length & デフォルト補完）
        from .parameter_normalizer import normalize_params

        converted_params = normalize_params(indicator_type, params, config)

        # パラメータとデータを結合して関数を呼び出し
        # 基本的な検証のみ実行（過度に厳格な検証を緩和）
        if indicator_type in ["SMA", "EMA", "WMA"]:
            length_val = converted_params.get("length", params.get("period"))
            # 基本的な期間検証のみ
            if isinstance(length_val, (int, np.integer)) and length_val <= 0:
                raise PandasTAError(
                    f"{indicator_type}: 期間は正の整数である必要があります: {length_val}"
                )

            # アダプタ関数のシグネチャに応じて余計な引数を落とす（互換性維持）
            tmp_all_args = {**required_data, **converted_params}
            sig = inspect.signature(indicator_func)
            allowed = set(sig.parameters.keys())
            filtered_args = {k: v for k, v in tmp_all_args.items() if k in allowed}
            result = indicator_func(**filtered_args)
            return result

        all_args = {**required_data, **converted_params}

        try:
            # 互換性のため、関数シグネチャに存在しない引数は除去
            sig = inspect.signature(indicator_func)
            allowed = set(sig.parameters.keys())
            filtered_args = {k: v for k, v in all_args.items() if k in allowed}
            result = indicator_func(**filtered_args)
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

        # 先頭大文字（例: Close）を優先してチェック
        capitalized_key = data_key.capitalize()
        if capitalized_key in df.columns:
            return capitalized_key

        # 全て大文字（例: CLOSE）をチェック
        upper_key = data_key.upper()
        if upper_key in df.columns:
            return upper_key

        # 小文字のカラム名（例: close）は許容しない（エッジケーステストの方針）
        return None
        # 見つからない場合はNone
        return None

        # 全て小文字でチェック
        lower_key = data_key.lower()
        if lower_key in df.columns:
            return lower_key

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
