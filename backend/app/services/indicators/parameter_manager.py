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
from app.services.indicators.constraints import constraint_engine

logger = logging.getLogger(__name__)


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

            # 制約エンジンを使用してパラメータを調整
            generated_params = constraint_engine.apply_constraints(
                indicator_type, generated_params
            )

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

            # 制約エンジンを使用した追加バリデーション
            if not constraint_engine.validate_constraints(indicator_type, parameters):
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
