"""
指標パラメータ管理システム

パラメータ生成とバリデーションを一元化するモジュール
"""

import logging
import random
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
