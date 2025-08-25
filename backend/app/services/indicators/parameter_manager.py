"""
指標パラメータ管理システム

パラメータ生成とバリデーションを一元化するモジュール
"""

import logging
import random
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Union

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
    "RSI_EMA_CROSS",
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
    "MAMA",
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
    "HT_DCPERIOD",
    "HT_DCPHASE",
    "HT_PHASOR",
    "HT_SINE",
    "HT_TRENDMODE",
}


class ParameterGenerationError(Exception):
    """パラメータ生成エラー"""


@dataclass
class ParameterRange:
    """パラメータ範囲情報"""

    min: Union[int, float]
    max: Union[int, float]
    default: Union[int, float]


class IndicatorParameterManager:
    """
    指標パラメータ管理クラス

    IndicatorConfigを基にパラメータの生成とバリデーションを一元管理する
    """

    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)

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
                "min": param_config.min_value,
                "max": param_config.max_value,
                "default": param_config.default_value,
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


def normalize_params(
    indicator_type: str, params: Dict[str, Any], config: IndicatorConfig
) -> Dict[str, Any]:
    """
    指標パラメータ正規化ユーティリティ

    - period -> length 変換
    - 必須 length のデフォルト補完
    - 各指標固有のパラメータマッピング
    """
    converted_params: Dict[str, Any] = {}

    # 特殊処理が必要な指標の変換ルールを定義
    special_conversions = {
        # SAR: acceleration -> af, maximum -> max_af, 予期しないパラメータを除外
        "SAR": {
            "param_map": {"acceleration": "af", "maximum": "max_af"},
            "exclude_params": {"period", "length"},
            "include_all_mapped": False,
        },
        # VWMA: period -> length, close -> data (param_mapがあれば優先)
        "VWMA": {
            "param_map": {"period": "length", "close": "data"},
            "fallback_map": {"period": "length"},
            "use_config_param_map": True,
        },
        # RMA: period -> length, close -> data
        "RMA": {
            "param_map": {"period": "length", "close": "data"},
        },
        # STC: close -> data
        "STC": {
            "param_map": {"close": "data"},
        },
        # RSI_EMA_CROSS: close -> data
        "RSI_EMA_CROSS": {
            "param_map": {"close": "data"},
        },
    }

    # 特殊処理が必要な指標の場合
    if indicator_type in special_conversions:
        conversion_rule = special_conversions[indicator_type]

        for key, value in params.items():
            # 除外パラメータはスキップ
            if (
                "exclude_params" in conversion_rule
                and key in conversion_rule["exclude_params"]
            ):
                continue

            # パラメータマッピングの適用
            if "param_map" in conversion_rule and key in conversion_rule["param_map"]:
                converted_params[conversion_rule["param_map"][key]] = value
            elif (
                "use_config_param_map" in conversion_rule
                and hasattr(config, "param_map")
                and config.param_map
            ):
                # config.param_map の値にキーが含まれているかチェック
                if key in config.param_map.values():
                    converted_params[key] = value
                elif key in conversion_rule.get("fallback_map", {}):
                    converted_params[conversion_rule["fallback_map"][key]] = value
                elif conversion_rule.get("include_all_mapped", True):
                    converted_params[key] = value
            else:
                # マッピングルールがない場合はそのまま
                converted_params[key] = value

        return converted_params

    # パラメータを一切受け取らない指標（ボリューム系指標）
    volume_indicators = {"NVI", "PVI", "PVT", "AD"}
    if indicator_type in volume_indicators:
        # period や length などのパラメータを除外
        for key, value in params.items():
            if key not in {"period", "length"}:
                converted_params[key] = value
        return converted_params

    # NO_LENGTH_INDICATORS の包括的特殊処理
    if indicator_type in NO_LENGTH_INDICATORS:
        # これらの指標は period や length パラメータを受け取らない
        for key, value in params.items():
            if key not in {"period", "length"}:
                converted_params[key] = value
        return converted_params

    # すべての指標がlengthパラメータを使用するため、変換は不要
    # ただし、互換性のためperiodパラメータが提供された場合はlengthに変換
    for key, value in params.items():
        if key == "period":
            converted_params["length"] = value
        else:
            converted_params[key] = value

    # length 必須のアダプタにデフォルト補完
    try:
        if config.adapter_function is not None:
            sig = inspect.signature(config.adapter_function)
        else:
            sig = None


        # length パラメータの追加は明示的に必要な指標のみに制限
        # SAR には length パラメータを追加しない（af, max_af のみを使用）
        if indicator_type == "SAR":
            pass  # SAR には length を追加しない
        elif indicator_type in NO_LENGTH_INDICATORS:
            pass  # これらの指標には length を追加しない
        elif indicator_type.startswith("CDL_") and "length" not in converted_params:
            pass  # すべてのパターン認識指標には length を追加しない
        elif sig is not None and "length" in sig.parameters and "length" not in converted_params:
            # period_to_length_indicators に含まれる指標のみ length を自動追加
            default_len = params.get("period")
            if default_len is None and config.parameters:
                if "period" in config.parameters:
                    default_len = config.parameters["period"].default_value
                elif "length" in config.parameters:
                    default_len = config.parameters["length"].default_value
            converted_params["length"] = (
                default_len if default_len is not None else DEFAULT_LENGTH
            )
    except Exception:
        pass

    return converted_params
