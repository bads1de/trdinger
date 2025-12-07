"""
指標パラメータ管理システム

パラメータ生成とバリデーションを一元化するモジュール
"""

import logging
import random
from typing import Any, Dict

from app.services.indicators.config.indicator_config import (
    IndicatorConfig,
)

logger = logging.getLogger(__name__)


class IndicatorParameterManager:
    """
    指標パラメータ管理クラス

    IndicatorConfigを基にパラメータの生成とバリデーションを一元管理する
    """

    def __init__(self):
        """初期化"""
        pass  # モジュールレベルの logger を使用

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
            Exception: パラメータ生成に失敗した場合
        """
        try:
            # 設定の妥当性チェック
            # indicator_typeが本名またはエイリアスであることを確認
            if config.indicator_name != indicator_type and indicator_type not in (
                config.aliases or []
            ):
                raise Exception(
                    f"指標タイプが一致しません: 要求されたのは {indicator_type} ですが、実際は {config.indicator_name} でした"
                )

            if not config.parameters:
                # パラメータが定義されていない場合は空辞書を返す
                return {}

            # 標準的なパラメータ生成
            generated_params = self._generate_standard_parameters(config)

            # 生成されたパラメータをバリデーション
            if not self.validate_parameters(indicator_type, generated_params, config):
                raise Exception(
                    f"{indicator_type} のために生成されたパラメータがバリデーションに失敗しました: {generated_params}"
                )

            return generated_params

        except Exception as e:
            logger.error(f"{indicator_type} のパラメータ生成に失敗しました: {e}")
            raise Exception(f"{indicator_type} のパラメータ生成に失敗しました: {e}")

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
                    logger.warning(
                        f"{indicator_type} に必要なパラメータ '{param_name}' がありません"
                    )
                    return False

                # 値の範囲チェック
                value = parameters[param_name]
                if not param_config.validate_value(value):
                    logger.warning(
                        f"パラメータ '{param_name}' の値 {value} は {indicator_type} の許容範囲外です"
                    )
                    return False

            # 余分なパラメータの確認
            for param_name in parameters:
                if param_name not in config.parameters:
                    logger.warning(
                        f"{indicator_type} に予期しないパラメータ '{param_name}' が含まれています"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"{indicator_type} のパラメータ検証に失敗しました: {e}")
            return False

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
