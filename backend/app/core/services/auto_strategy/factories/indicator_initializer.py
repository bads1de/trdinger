"""
指標初期化器

指標の初期化とアダプター統合を担当するモジュール。
TALibAdapterシステムとの統合を重視した実装です。
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional

from ..models.strategy_gene import IndicatorGene
from .indicator_calculator import IndicatorCalculator
from app.core.services.indicators.config import indicator_registry
from app.core.services.indicators.parameter_manager import IndicatorParameterManager
from app.core.utils.data_utils import convert_to_series

logger = logging.getLogger(__name__)


class IndicatorInitializer:
    """
    指標初期化器

    指標の初期化と戦略への統合を担当します。
    計算ロジックはIndicatorCalculatorに委譲します。
    """

    def __init__(self):
        """初期化"""
        self.indicator_calculator = IndicatorCalculator()
        self.parameter_manager = IndicatorParameterManager()

    def calculate_indicator_only(
        self, indicator_type: str, parameters: Dict[str, Any], data: pd.DataFrame
    ) -> tuple:
        """
        指標計算のみを行う（戦略インスタンスへの追加は行わない）
        パラメータバリデーションを含む
        """
        try:
            resolved_indicator_type = indicator_registry.resolve_indicator_type(
                indicator_type
            )
            if not resolved_indicator_type:
                logger.warning(f"未対応の指標タイプ（代替なし）: {indicator_type}")
                return None, None
            indicator_type = resolved_indicator_type

            # パラメータバリデーション
            indicator_config = indicator_registry.get_indicator_config(indicator_type)
            if indicator_config:
                if not self.parameter_manager.validate_parameters(
                    indicator_type, parameters, indicator_config
                ):
                    logger.error(f"パラメータバリデーション失敗: {indicator_type}")
                    return None, None
            else:
                logger.warning(f"指標設定が見つかりません: {indicator_type}")

            close_data = pd.Series(data["close"].values, index=data.index)
            high_data = pd.Series(data["high"].values, index=data.index)
            low_data = pd.Series(data["low"].values, index=data.index)
            volume_data = pd.Series(data["volume"].values, index=data.index)
            open_data = pd.Series(data["open"].values, index=data.index)

            return self.indicator_calculator.calculate_indicator(
                indicator_type,
                parameters,
                close_data,
                high_data,
                low_data,
                volume_data,
                open_data,
            )

        except Exception as e:
            logger.error(f"指標計算エラー ({indicator_type}): {e}")
            return None, None

    def initialize_indicator(
        self, indicator_gene: IndicatorGene, data, strategy_instance
    ) -> Optional[str]:
        """
        単一指標の初期化（パラメータバリデーション付き）
        """
        try:
            indicator_type = indicator_gene.type
            parameters = indicator_gene.parameters
            original_type = indicator_type

            # 指標タイプ解決
            indicator_type = indicator_registry.resolve_indicator_type(indicator_type)
            if not indicator_type:
                return None

            # パラメータバリデーション
            indicator_config = indicator_registry.get_indicator_config(indicator_type)
            if indicator_config:
                validation_result = self.parameter_manager.validate_parameters(
                    indicator_type, parameters, indicator_config
                )
                if not validation_result:
                    logger.error(f"パラメータバリデーション失敗: {indicator_type}")
                    return None
            else:
                logger.warning(f"指標設定が見つかりません: {indicator_type}")

            # データ変換
            close_data = convert_to_series(data.Close)
            high_data = convert_to_series(data.High)
            low_data = convert_to_series(data.Low)
            volume_data = convert_to_series(data.Volume)
            open_data = convert_to_series(data.Open) if hasattr(data, "Open") else None

            # 指標計算
            result, indicator_name = self.indicator_calculator.calculate_indicator(
                indicator_type,
                parameters,
                close_data,
                high_data,
                low_data,
                volume_data,
                open_data,
            )

            if result is not None and indicator_name is not None:
                json_indicator_name = indicator_registry.generate_json_name(
                    original_type
                )

                if isinstance(result, dict):
                    indicator_config = indicator_registry.get_indicator_config(
                        original_type
                    )
                    if indicator_config and indicator_config.result_handler:
                        handler_key = indicator_config.result_handler
                        indicator_values = result.get(
                            handler_key, list(result.values())[0]
                        )
                else:
                    indicator_values = (
                        result.values if hasattr(result, "values") else result
                    )

                # backtesting.pyのIメソッドを使用して動的指標を作成
                # indicator_valuesをPandas Seriesに変換
                if not isinstance(indicator_values, pd.Series):
                    indicator_values = pd.Series(
                        indicator_values, index=close_data.index
                    )

                # backtesting.pyの正しい指標作成方法
                # 指標計算関数を作成（事前計算された値を返す関数）
                def create_indicator_func(values):
                    """
                    事前計算された指標値を返す関数を作成
                    backtesting.pyのIメソッドが期待する形式に合わせる

                    backtesting.pyは各バーで現在のデータ長に応じた配列を期待する
                    """
                    import numpy as np

                    def indicator_func(data):
                        # データの長さを取得
                        data_length = len(data)

                        # 指標値の配列を作成
                        if data_length <= len(values):
                            # データ長に合わせて指標値を切り取り
                            if hasattr(values, "iloc"):
                                result_values = values.iloc[:data_length].values
                            else:
                                result_values = np.array(values[:data_length])
                        else:
                            # データが指標値より長い場合は、最後の値で埋める
                            if hasattr(values, "iloc"):
                                base_values = values.values
                            else:
                                base_values = np.array(values)

                            # 不足分を最後の値で埋める
                            last_value = base_values[-1] if len(base_values) > 0 else 0
                            padding = np.full(
                                data_length - len(base_values), last_value
                            )
                            result_values = np.concatenate([base_values, padding])

                        return result_values

                    return indicator_func

                try:
                    strategy_instance.indicators[json_indicator_name] = (
                        strategy_instance.I(
                            create_indicator_func(indicator_values),
                            strategy_instance.data.Close,
                            name=json_indicator_name,
                        )
                    )
                except Exception:
                    import traceback

                    traceback.print_exc()
                    return None

                legacy_indicator_name = self._get_legacy_indicator_name(
                    original_type, parameters
                )
                if legacy_indicator_name != json_indicator_name:
                    strategy_instance.indicators[legacy_indicator_name] = (
                        strategy_instance.indicators[json_indicator_name]
                    )

                return json_indicator_name

            return None

        except Exception as e:
            logger.error(f"指標初期化エラー ({indicator_gene.type}): {e}")
            import traceback

            traceback.print_exc()
            return None

    def _get_legacy_indicator_name(self, indicator_type: str, parameters: dict) -> str:
        """レガシー形式の指標名を生成（後方互換性用）"""
        try:
            return indicator_registry.generate_legacy_name(indicator_type, parameters)
        except Exception as e:
            logger.warning(f"レガシー指標名生成エラー ({indicator_type}): {e}")
            return indicator_type

    def get_supported_indicators(self) -> list:
        """サポートされている指標のリストを取得"""
        return list(indicator_registry.get_supported_indicator_names())

    def is_supported_indicator(self, indicator_type: str) -> bool:
        """指標がサポートされているかチェック（代替指標も含む）"""
        return indicator_registry.is_indicator_supported(indicator_type)
