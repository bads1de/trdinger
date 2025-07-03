"""
指標パラメータ管理システム

パラメータ生成とバリデーションを一元化するモジュール
"""

import random
import logging

from typing import Dict, Any, Union
from dataclasses import dataclass

from app.core.services.indicators.config.indicator_config import (
    IndicatorConfig,
)

logger = logging.getLogger(__name__)


class ParameterGenerationError(Exception):
    """パラメータ生成エラー"""

    pass


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
                    f"Indicator type mismatch: expected {indicator_type}, got {config.indicator_name}"
                )

            if not config.parameters:
                # パラメータが定義されていない場合は空辞書を返す
                return {}

            generated_params = {}

            # 特別な処理が必要な指標
            if indicator_type == "MACD":
                generated_params = self._generate_macd_parameters(config)
            elif indicator_type == "BB":
                generated_params = self._generate_bollinger_bands_parameters(config)
            elif indicator_type == "STOCH":
                generated_params = self._generate_stochastic_parameters(config)
            else:
                # 標準的なパラメータ生成
                generated_params = self._generate_standard_parameters(config)

            # 生成されたパラメータをバリデーション
            if not self.validate_parameters(indicator_type, generated_params, config):
                raise ParameterGenerationError(
                    f"Generated parameters for {indicator_type} failed validation: {generated_params}"
                )

            self.logger.debug(
                f"Generated parameters for {indicator_type}: {generated_params}"
            )
            return generated_params

        except ParameterGenerationError:
            # 既にParameterGenerationErrorの場合は再発生
            raise
        except Exception as e:
            self.logger.error(f"Parameter generation failed for {indicator_type}: {e}")
            raise ParameterGenerationError(
                f"Failed to generate parameters for {indicator_type}: {e}"
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
                        f"Missing required parameter '{param_name}' for {indicator_type}"
                    )
                    return False

                # 値の範囲チェック
                value = parameters[param_name]
                if not param_config.validate_value(value):
                    self.logger.warning(
                        f"Parameter '{param_name}' value {value} is out of range for {indicator_type}"
                    )
                    return False

            # 余分なパラメータの確認
            for param_name in parameters:
                if param_name not in config.parameters:
                    self.logger.warning(
                        f"Unexpected parameter '{param_name}' for {indicator_type}"
                    )
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Parameter validation failed for {indicator_type}: {e}")
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

    def _generate_macd_parameters(self, config: IndicatorConfig) -> Dict[str, Any]:
        """MACDパラメータの生成（fast_period < slow_periodを保証）"""
        params = {}

        # 各パラメータの範囲を取得
        fast_config = config.parameters.get("fast_period")
        slow_config = config.parameters.get("slow_period")
        signal_config = config.parameters.get("signal_period")

        # fast_periodとslow_periodの生成
        fast_period = 12  # デフォルト値
        slow_period = 26  # デフォルト値

        if (
            fast_config
            and slow_config
            and fast_config.min_value is not None
            and fast_config.max_value is not None
            and slow_config.min_value is not None
            and slow_config.max_value is not None
        ):
            try:
                # fast_period < slow_periodを保証
                fast_period_val = random.randint(
                    int(fast_config.min_value), int(fast_config.max_value)
                )
                slow_min = max(int(slow_config.min_value), fast_period_val + 1)
                slow_max = int(slow_config.max_value)

                if slow_min <= slow_max:
                    slow_period_val = random.randint(slow_min, slow_max)
                    fast_period = fast_period_val
                    slow_period = slow_period_val
                else:
                    self.logger.warning(
                        "Could not generate slow_period > fast_period for MACD, using defaults."
                    )
            except ValueError:
                self.logger.warning(
                    "Invalid range for MACD fast/slow periods, using defaults."
                )
        else:
            self.logger.warning(
                "MACD fast/slow period range not fully defined, using default values."
            )

        params["fast_period"] = fast_period
        params["slow_period"] = slow_period

        # signal_periodの生成
        signal_period = 9  # デフォルト値
        if (
            signal_config
            and signal_config.min_value is not None
            and signal_config.max_value is not None
        ):
            try:
                signal_period = random.randint(
                    int(signal_config.min_value), int(signal_config.max_value)
                )
            except ValueError:
                self.logger.warning(
                    "Invalid range for MACD signal_period, using default."
                )
        else:
            self.logger.warning(
                "MACD signal_period range not defined, using default value."
            )
        params["signal_period"] = signal_period

        # すべてのパラメータが生成されたか確認
        if not all(
            key in params for key in ["fast_period", "slow_period", "signal_period"]
        ):
            raise ParameterGenerationError(
                "Failed to generate all required parameters for MACD. "
                "Check indicator configuration."
            )

        return params

    def _generate_bollinger_bands_parameters(
        self, config: IndicatorConfig
    ) -> Dict[str, Any]:
        """Bollinger Bandsパラメータの生成"""
        return self._generate_standard_parameters(config)

    def _generate_stochastic_parameters(
        self, config: IndicatorConfig
    ) -> Dict[str, Any]:
        """Stochasticパラメータの生成（matypeを0-8に制限）"""
        params = self._generate_standard_parameters(config)

        # matypeを0-8の範囲に制限
        if "slowk_matype" in params:
            params["slowk_matype"] = random.randint(0, 8)
        if "slowd_matype" in params:
            params["slowd_matype"] = random.randint(0, 8)

        return params
