"""
指標パラメータ管理システム

パラメータ生成とバリデーションを一元化するモジュール
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from app.services.indicators.config.indicator_config import (
    IndicatorConfig,
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_LENGTH = 14
NO_LENGTH_INDICATORS = {
    "SAR",
    "OBV",
    "VWAP",
    "AD",
    "ADOSC",
    "AO",
    "ICHIMOKU",
    "PVT",
    "PVOL",
    "PVR",
    "PPO",
    "APO",
    "ULTOSC",
    "BOP",
    "STC",
    "KDJ",
    "CDL_PIERCING",
    "CDL_HAMMER",
    "CDL_HANGING_MAN",
    "CDL_HARAMI",
    "CDL_DARK_CLOUD_COVER",
    "CDL_THREE_BLACK_CROWS",
    "CDL_THREE_WHITE_SOLDIERS",
    "CDL_MARUBOZU",
    "CDL_SPINNING_TOP",
    "CDL_SHOOTING_STAR",
    "CDL_ENGULFING",
    "CDL_MORNING_STAR",
    "CDL_EVENING_STAR",
    "CDL_DOJI",
    "HAMMER",
    "ENGULFING_PATTERN",
    "MORNING_STAR",
    "EVENING_STAR",
    "NVI",
    "PVI",
    "PVT",
    "CMF",
    "EOM",
    "KVO",
    "STOCH",
    "STOCHF",
    "KST",
    "SMI",
    "UO",
    "PVO",
    "TRANGE",
    "BB",
    "ACOS",
    "ASIN",
    "ATAN",
    "COS",
    "COSH",
    "SIN",
    "SINH",
    "TAN",
    "TANH",
    "SQRT",
    "EXP",
    "LN",
    "LOG10",
    "CEIL",
    "FLOOR",
    "ADD",
    "SUB",
    "MULT",
    "DIV",
    "WCP",
    "HLC3",
    "HL2",
    "OHLC4",
    "VP",
    "AOBV",
    "HWC",
}


class ParameterGenerationError(Exception):
    """パラメータ生成エラー"""


@dataclass
class ParameterRange:
    """パラメータ範囲情報"""

    min_value: Union[int, float]
    max_value: Union[int, float]
    default_value: Union[int, float]


# パラメータマッピング定義
PARAMETER_MAPPINGS = {
    # pandas-taスタイル (lengthを使わない指標)
    "no_length": {
        "SAR": {"period": None, "length": None},
        "OBV": {"period": None, "length": None},
        "VWAP": {"period": None, "length": None},
        "AD": {"period": None, "length": None},
        "ADOSC": {"fast": "fast", "slow": "slow"},
        "AO": {"period": None, "length": None},  # Awesome Oscillatorはパラメータなし
        "ICHIMOKU": {"tenkan": "tenkan", "kijun": "kijun", "senkou": "senkou"},
        "PVT": {"period": None, "length": None},
        "PVOL": {"period": None, "length": None},
        "PVR": {"period": None, "length": None},
        "ULTOSC": {"fast": "fast", "medium": "medium", "slow": "slow"},
        "BOP": {"open": "open_", "high": "high", "low": "low", "close": "close"},
        "STC": {"tclength": "tclength", "fast": "fast", "slow": "slow"},
        "STOCHRSI": {"period": "length", "fastk_period": "fastk_period", "fastd_period": "fastd_period"},
        "KDJ": {"k": "k", "d": "d"},
        "RVI": {"period": "length"},
        "KST": {"r1": "roc1", "r2": "roc2", "r3": "roc3", "r4": "roc4"},  # KSTのパラメータ不一致修正
        "SMI": {"fast": "fast", "slow": "slow", "signal": "signal"},
        "UO": {"fast": "fast", "medium": "medium", "slow": "slow"},
        "PVO": {"fast": "fast", "slow": "slow", "signal": "signal"},
        "TRANGE": {"period": None, "length": None},
        "BB": {"period": "length", "std": "std"},
        "ACOS": {},
        "ASIN": {},
        "ATAN": {},
        "COS": {},
        "COSH": {},
        "SIN": {},
        "SINH": {},
        "TAN": {},
        "TANH": {},
        "SQRT": {},
        "EXP": {},
        "LN": {},
        "LOG10": {},
        "CEIL": {},
        "FLOOR": {},
        "ADD": {},
        "SUB": {},
        "MULT": {},
        "DIV": {},
        "WCP": {"close": "data"},
        "HLC3": {"high": "high", "low": "low", "close": "close"},
        "HL2": {"high": "high", "low": "low"},
        "OHLC4": {"open": "open_", "high": "high", "low": "low", "close": "close"},
        "VP": {"width": "width"},
        "AOBV": {"fast": "fast", "slow": "slow"},
        "HWC": {"na": "na", "nb": "nb", "nc": "nc", "nd": "nd", "scalar": "scalar"},
    },
    # TA-Libスタイル (lengthをperiodにマップ)
    "length_to_period": {
        "ADX": {"period": "length"},
        "MFI": {"period": "length"},
        "WILLR": {"period": "length"},
        "AROON": {"period": "length"},
        "AROONOSC": {"period": "length"},
        "PLUS_DI": {"period": "length"},
        "MINUS_DI": {"period": "length"},
        "ADX": {"period": "length"},
        "DX": {"period": "length"},
        "PLUS_DM": {"period": "length"},
        "MINUS_DM": {"period": "length"},
        "RSI": {"period": "length"},
        "CCI": {"period": "length"},
        "MACD": {"fast": "fast", "slow": "slow", "signal": "signal"},
        "STOCH": {"fastk_period": "k", "d_length": "smooth_k", "slowd_period": "d"},
        "STOCHF": {"fastk_period": "k", "fastd_period": "d"},
        "ROC": {"period": "length"},
        "MOM": {"period": "length"},
        "UO": {"fast": "fast", "medium": "medium", "slow": "slow"},
        "TRIX": {"period": "length"},
        "ULTOSC": {"fast": "fast", "medium": "medium", "slow": "slow"},
        "PLUS_DI": {"period": "length"},
        "MINUS_DI": {"period": "length"},
        "ADOSC": {"fastperiod": "fast", "slowperiod": "slow"},
        "APO": {"fastperiod": "fast", "slowperiod": "slow"},
        "PPO": {"fastperiod": "fast", "slowperiod": "slow", "matype": "signal"},
        "ROCP": {"period": "length"},
        "ROCR": {"period": "length"},
        "ROCR100": {"period": "length"},
        "TRANGE": {"period": None, "length": None},
        "ATR": {"period": "length"},
        "NATR": {"period": "length"},
        "BBANDS": {"period": "length", "std": "std"},
        "ACCBANDS": {"period": "length"},
        "MASSI": {"fast": "fast", "slow": "slow"},
        "PDIST": {"period": "length"},
        "UI": {"period": "length"},
        "MIDPOINT": {"period": "period"},
        "MIDPRICE": {"period": "length"},
        "PVT": {"period": None, "length": None},
        "CMF": {"period": "length"},
        "EFI": {"period": "length"},
        "PVR": {"period": None, "length": None},
        "HL2": {"high": "high", "low": "low"},
        "HLC3": {"high": "high", "low": "low", "close": "close"},
        "OHLC4": {"open": "open_", "high": "high", "low": "low", "close": "close"},
        "WCP": {"close": "data"},
        "AD": {"period": None, "length": None},
        "OBV": {"period": None, "length": None},
        "VWAP": {"period": None, "length": None},
    }
}


class IndicatorParameterManager:
    """
    指標パラメータ管理クラス

    IndicatorConfigを基にパラメータの生成とバリデーションを一元管理する
    """

    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)

    def map_parameters(self, indicator_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        指標タイプに応じたパラメータマッピングを実行

        Args:
            indicator_type: 指標タイプ
            parameters: 元のパラメータ

        Returns:
            マッピング後のパラメータ
        """
        if indicator_type in PARAMETER_MAPPINGS["no_length"]:
            mapping = PARAMETER_MAPPINGS["no_length"][indicator_type]
            return self._apply_mapping(parameters, mapping)
        elif indicator_type in PARAMETER_MAPPINGS["length_to_period"]:
            mapping = PARAMETER_MAPPINGS["length_to_period"][indicator_type]
            return self._apply_mapping(parameters, mapping)
        else:
            # デフォルトマッピング - lengthをperiodに変換
            mapped = {}
            for key, value in parameters.items():
                if key == "length":
                    mapped["period"] = value
                else:
                    mapped[key] = value
            return mapped

    def _apply_mapping(self, parameters: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[str, Any]:
        """
        マッピング定義を適用

        Args:
            parameters: 元のパラメータ
            mapping: マッピング定義

        Returns:
            マッピング済みパラメータ
        """
        mapped = {}
        for target_key, source_key in mapping.items():
            if source_key is None:
                continue  # パラメータを無視
            elif source_key in parameters:
                mapped[target_key] = parameters[source_key]
            elif target_key in ["period", "length"]:
                # デフォルトのlength/period変換
                if "length" in parameters:
                    mapped[target_key] = parameters["length"]
                elif "period" in parameters:
                    mapped[target_key] = parameters["period"]
        # マッピングに含まれないパラメータもそのまま渡す
        for key, value in parameters.items():
            if key not in [v for v in mapping.values() if v is not None]:
                mapped[key] = value
        return mapped

    def get_standard_length(self, indicator_type: str) -> int:
        """
        指標タイプの標準lengthを取得

        Args:
            indicator_type: 指標タイプ

        Returns:
            標準length
        """
        if indicator_type in ["AO", "KDJ", "KST", "RVI", "VORTEX"]:
            return 10
        elif indicator_type in ["ADX", "MFI", "WILLR", "AROON", "AROONOSC"]:
            return 14
        else:
            return DEFAULT_LENGTH

    def generate_parameters(
        self, indicator_type: str, config: IndicatorConfig
    ) -> Dict[str, Any]:
        """
        指標タイプと設定に基づいてパラメータを生成

        Args:
            indicator_type: 指標タイプ（例：RSI, MACD）
            config: 指標設定

        Returns:
            生成されたパラメータ辞書

        Raises:
            ParameterGenerationError: パラメータ生成に失敗した場合
        """
        try:
            # 設定の妥当性チェック
            if config.indicator_name != indicator_type:
                raise ParameterGenerationError(
                    f"指標タイプが一致しません: 要求されたのは {indicator_type} ですが、実際は {config.indicator_name} でした"
                )

            if not config.parameters:
                # パラメータが定義されていない場合は空辞書を返す
                return {}

            # 標準的なパラメータ生成
            generated_params = self._generate_standard_parameters(config)

            # 生成されたパラメータをバリデーション
            if not self.validate_parameters(indicator_type, generated_params, config):
                raise ParameterGenerationError(
                    f"{indicator_type} のために生成されたパラメータがバリデーションに失敗しました: {generated_params}"
                )

            return generated_params

        except ParameterGenerationError:
            # 既にParameterGenerationErrorの場合は再発生
            raise
        except Exception as e:
            self.logger.error(f"{indicator_type} のパラメータ生成に失敗しました: {e}")
            raise ParameterGenerationError(
                f"{indicator_type} のパラメータ生成に失敗しました: {e}"
            )

    def validate_parameters(
        self, indicator_type: str, parameters: Dict[str, Any], config: IndicatorConfig
    ) -> bool:
        """
        パラメータの妥当性を検証

        Args:
            indicator_type: 指標タイプ
            parameters: 検証するパラメータ
            config: 指標設定

        Returns:
            バリデーション結果（True: 有効, False: 無効）
        """
        try:
            # 必須パラメータの存在確認
            for param_name, param_config in config.parameters.items():
                if param_name not in parameters:
                    self.logger.warning(
                        f"{indicator_type} に必要なパラメータ '{param_name}' がありません"
                    )
                    return False

                # 値の範囲チェック
                value = parameters[param_name]
                if not param_config.validate_value(value):
                    self.logger.warning(
                        f"パラメータ '{param_name}' の値 {value} は {indicator_type} の許容範囲外です"
                    )
                    return False

            # 余分なパラメータの確認
            for param_name in parameters:
                if param_name not in config.parameters:
                    self.logger.warning(
                        f"{indicator_type} に予期しないパラメータ '{param_name}' が含まれています"
                    )
                    return False

            return True

        except Exception as e:
            self.logger.error(f"{indicator_type} のパラメータ検証に失敗しました: {e}")
            return False

    def get_parameter_ranges(
        self, indicator_type: str, config: IndicatorConfig
    ) -> Dict[str, ParameterRange]:
        """
        指標のパラメータ範囲情報を取得

        Args:
            indicator_type: 指標タイプ
            config: 指標設定

        Returns:
            パラメータ範囲情報の辞書
        """
        ranges = {}
        for param_name, param_config in config.parameters.items():
            ranges[param_name] = {
                "min_value": param_config.min_value,
                "max_value": param_config.max_value,
                "default_value": param_config.default_value,
            }
        return ranges

    def _generate_standard_parameters(self, config: IndicatorConfig) -> Dict[str, Any]:
        """標準的なパラメータ生成"""
        params = {}
        for param_name, param_config in config.parameters.items():
            if (
                param_config.min_value is not None
                and param_config.max_value is not None
            ):
                if isinstance(param_config.default_value, int):
                    # 整数パラメータ
                    params[param_name] = random.randint(
                        int(param_config.min_value), int(param_config.max_value)
                    )
                else:
                    # 浮動小数点パラメータ
                    params[param_name] = random.uniform(
                        float(param_config.min_value), float(param_config.max_value)
                    )
            else:
                # 範囲が定義されていない場合はデフォルト値を使用
                params[param_name] = param_config.default_value
        return params

    def validate_and_clean_result(self, result: Any, indicator_type: str) -> Optional[Any]:
        """
        指標計算結果のNaN/Noneをクリーンアップ

        Args:
            result: 指標計算結果
            indicator_type: 指標タイプ

        Returns:
            クリーンアップ済み結果 (Noneはエラー)
        """
        if result is None:
            self.logger.warning(f"{indicator_type} の計算結果がNoneです")
            return None

        # pandas Series/DataFrameの場合
        try:
            import pandas as pd
            import numpy as np

            if isinstance(result, pd.Series):
                if result.empty:
                    self.logger.warning(f"{indicator_type} の結果Seriesが空です")
                    return None

                # NaNを削除または除去
                has_nan = result.isna().any()
                if has_nan:
                    self.logger.warning(f"{indicator_type} にNaNを含みます、クリーンアップします")
                    result = result.ffill().bfill()
                    # まだNaNが残る場合はデフォルト値で埋める
                    if result.isna().any():
                        default_value = self._get_default_value_for_indicator(indicator_type)
                        result = result.fillna(default_value)

                return result

            elif isinstance(result, pd.DataFrame):
                if result.empty:
                    self.logger.warning(f"{indicator_type} の結果DataFrameが空です")
                    return None

                # NaNをクリーンアップ
                has_nan = result.isna().any().any()
                if has_nan:
                    self.logger.warning(f"{indicator_type} のDataFrameにNaNを含みます、クリーンアップします")
                    result = result.ffill().bfill()

                return result

            elif isinstance(result, (list, tuple)):
                # リスト/タプルの場合、各要素をチェック
                if not result:
                    self.logger.warning(f"{indicator_type} の結果が空のリスト/タプルです")
                    return None

                cleaned = []
                for r in result:
                    if pd.isna(r):
                        default_value = self._get_default_value_for_indicator(indicator_type)
                        cleaned.append(default_value)
                        self.logger.warning(f"{indicator_type} のリスト要素にNaNが見つかり、デフォルト値へ置換")
                    else:
                        cleaned.append(r)
                return type(result)(cleaned)  # 元の型を維持

            else:
                # スカラー値の場合
                if pd.isna(result):
                    self.logger.warning(f"{indicator_type} のスカラー結果がNaNです、デフォルト値へ置換")
                    return self._get_default_value_for_indicator(indicator_type)
                return result

        except ImportError:
            # pandasが利用できない場合
            self.logger.warning("pandasが利用できないため、NaNチェックをスキップします")
            if result is None:
                return None
            return result

    def _get_default_value_for_indicator(self, indicator_type: str) -> float:
        """
        指標タイプに応じたデフォルト値を取得

        Args:
            indicator_type: 指標タイプ

        Returns:
            デフォルト値
        """
        # オシレーター系は50を中心に
        if indicator_type in ["RSI", "STOCH", "CCI", "MFI", "WILLR"]:
            return 50.0
        # トレンド指数系は0を中心に
        elif indicator_type in ["MACD", "ROC", "MOM", "ADX"]:
            return 0.0
        # ボラティリティ系はATRを基準に
        elif indicator_type in ["ATR", "NATR"]:
            return 1.0
        else:
            return 0.0
