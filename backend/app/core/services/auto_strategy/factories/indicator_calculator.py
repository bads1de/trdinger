"""
指標計算機

指標の計算ロジックを専門に担当するモジュール。
IndicatorInitializerから計算機能を分離し、責務を明確化します。
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional

from app.core.services.indicators.config import (
    indicator_registry,
    compatibility_manager,
)

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """
    指標計算を担当するクラス

    責務:
    - 純粋な指標計算ロジックの実行
    - IndicatorConfig に基づく統一された計算フロー
    - バリデーション済みパラメータの処理

    注意:
    - パラメータバリデーションは IndicatorInitializer で実行されます
    - このクラスはバリデーション済みのデータのみを処理します
    """

    def __init__(self):
        """初期化"""
        self.indicator_config = self._setup_indicator_config()
        self._current_data = None

        # 互換性モードを有効化（段階的移行のため）
        compatibility_manager.enable_compatibility_mode()

    def _setup_indicator_config(self) -> Dict[str, Dict[str, Any]]:
        """指標設定マッピングを設定（JSON形式対応）"""
        config = {}

        # オートストラテジー用の10個の指標を設定
        auto_strategy_indicators = [
            "SMA",
            "EMA",
            "MACD",
            "BB",
            "RSI",
            "STOCH",
            "CCI",
            "ADX",
            "ATR",
            "OBV",
        ]

        for indicator_name in auto_strategy_indicators:
            indicator_config = indicator_registry.get_indicator_config(indicator_name)
            if indicator_config:
                # JSON形式ベースの設定（統一）
                config[indicator_name] = {
                    "indicator_config": indicator_config,
                    "adapter_function": indicator_config.adapter_function,
                    "required_data": indicator_config.required_data,
                    "result_type": indicator_config.result_type.value,
                    "result_handler": indicator_config.result_handler,
                    "legacy_name_format": indicator_config.legacy_name_format,
                }
            else:
                logger.warning(f"指標 {indicator_name} の設定が見つかりません")
                # 設定が見つからない場合はスキップ

        return config

    def calculate_indicator(
        self,
        indicator_type: str,
        parameters: Dict[str, Any],
        close_data: pd.Series,
        high_data: pd.Series,
        low_data: pd.Series,
        volume_data: pd.Series,
        open_data: Optional[pd.Series] = None,
    ) -> tuple:
        """
        指標を計算（純粋な計算ロジック）

        注意: このメソッドはバリデーション済みのパラメータを受け取ることを前提とします。
        パラメータバリデーションは IndicatorInitializer で実行されます。

        Args:
            indicator_type: 指標タイプ
            parameters: バリデーション済みパラメータ
            close_data: 終値データ
            high_data: 高値データ
            low_data: 安値データ
            volume_data: 出来高データ
            open_data: 始値データ（オプション）

        Returns:
            tuple: (計算結果, 指標名)
        """
        try:
            # データを一時的に保存（将来の拡張用）
            if open_data is not None:
                self._current_data = pd.DataFrame(
                    {
                        "open": open_data,
                        "high": high_data,
                        "low": low_data,
                        "close": close_data,
                        "volume": volume_data,
                    }
                )

            # 統一された設定マッピングを使用した処理
            if indicator_type in self.indicator_config:
                return self._calculate_from_config(
                    indicator_type,
                    parameters,
                    close_data,
                    high_data,
                    low_data,
                    volume_data,
                )
            else:
                logger.warning(f"未対応の指標タイプ: {indicator_type}")
                return None, None

        except Exception as e:
            logger.error(f"指標計算エラー ({indicator_type}): {e}")
            return None, None

    def _calculate_from_config(
        self,
        indicator_type: str,
        parameters: Dict[str, Any],
        close_data: pd.Series,
        high_data: pd.Series,
        low_data: pd.Series,
        volume_data: pd.Series,
    ) -> tuple:
        """設定マッピングベースの指標計算"""
        config = self.indicator_config[indicator_type]

        data_args = self._prepare_data_for_indicator(
            config, close_data, high_data, low_data, volume_data
        )
        param_args = self._prepare_parameters_for_indicator(config, parameters)
        result = self._call_adapter_function(
            config["adapter_function"], data_args, param_args
        )

        if config["result_type"] == "complex":
            return self._handle_complex_result(
                result, config, indicator_type, parameters
            )
        else:
            indicator_name = config["indicator_config"].generate_json_name()
            return result, indicator_name

    def _prepare_data_for_indicator(
        self,
        config: Dict[str, Any],
        close_data: pd.Series,
        high_data: pd.Series,
        low_data: pd.Series,
        volume_data: pd.Series,
    ) -> list:
        """指標に必要なデータを準備"""
        data_map = {
            "close": close_data,
            "high": high_data,
            "low": low_data,
            "volume": volume_data,
        }
        required_data = config["required_data"]
        return [data_map[data_type] for data_type in required_data]

    def _prepare_parameters_for_indicator(
        self, config: Dict[str, Any], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """指標に必要なパラメータを準備（IndicatorConfig完全依存）"""
        prepared_params = {}

        # IndicatorConfigから完全にパラメータを取得
        indicator_config = config["indicator_config"]
        indicator_name = indicator_config.indicator_name

        # 指標固有のパラメータマッピング
        param_mapping = self._get_parameter_mapping(indicator_name)

        # IndicatorConfigからパラメータのデフォルト値を取得
        for param_name, param_config in indicator_config.parameters.items():
            value = parameters.get(param_name, param_config.default_value)

            # パラメータ名をアダプター関数の引数名にマッピング
            adapter_param_name = param_mapping.get(param_name, param_name)

            # 特定の指標で使用されないパラメータをスキップ
            if self._should_skip_parameter(indicator_name, param_name):
                continue

            prepared_params[adapter_param_name] = value

        return prepared_params

    def _get_parameter_mapping(self, indicator_name: str) -> Dict[str, str]:
        """指標固有のパラメータ名マッピングを取得"""
        mappings = {
            "MACD": {
                "fast_period": "fast",  # MomentumAdapter.macdの引数名
                "slow_period": "slow",  # MomentumAdapter.macdの引数名
                "signal_period": "signal",  # MomentumAdapter.macdの引数名
            },
            "STOCH": {
                "k_period": "k_period",  # MomentumAdapter.stochasticの引数名
                "d_period": "d_period",  # MomentumAdapter.stochasticの引数名
                # slowingパラメータはstochasticメソッドにはない
            },
            "BB": {"period": "period", "std_dev": "std_dev"},
        }
        return mappings.get(indicator_name, {})

    def _should_skip_parameter(self, indicator_name: str, param_name: str) -> bool:
        """特定の指標で使用されないパラメータかどうかを判定"""
        skip_rules = {
            "STOCH": [
                "slowing"
            ],  # stochasticメソッドではslowingパラメータは使用されない
        }

        skip_params = skip_rules.get(indicator_name, [])
        return param_name in skip_params

    def _call_adapter_function(
        self, adapter_function, data_args: list, param_args: Dict[str, Any]
    ) -> Any:
        """アダプター関数を呼び出し"""
        if param_args:
            return adapter_function(*data_args, **param_args)
        else:
            return adapter_function(*data_args)

    def _handle_complex_result(
        self,
        result: Any,
        config: Dict[str, Any],
        indicator_type: str,
        parameters: Dict[str, Any],
    ) -> tuple:
        """複合指標の結果を処理"""
        try:
            result_handler = config.get("result_handler")
            indicator_name = config["indicator_config"].generate_json_name()

            if result_handler == "macd_handler":
                if isinstance(result, dict) and "macd_line" in result:
                    return result["macd_line"], indicator_name
            elif result_handler == "bb_handler":
                if isinstance(result, dict) and "middle" in result:
                    return result["middle"], indicator_name

            # デフォルト処理
            return result, indicator_name
        except Exception as e:
            logger.error(f"複合指標処理エラー ({indicator_type}): {e}")
            return None, None
