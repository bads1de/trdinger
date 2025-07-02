"""
指標初期化器

指標の初期化とアダプター統合を担当するモジュール。
指標の初期化と戦略への統合を担当するモジュール。
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, List

from ..models.strategy_gene import IndicatorGene
from app.core.services.indicators import TechnicalIndicatorService
from app.core.services.indicators.config import indicator_registry
from app.core.services.indicators.parameter_manager import IndicatorParameterManager
from app.core.utils.data_utils import convert_to_series
from app.core.services.indicators.config.indicator_config import IndicatorResultType

logger = logging.getLogger(__name__)


class IndicatorInitializer:
    """
    指標初期化器

    指標の初期化と戦略への統合を担当します。
    計算ロジックはTechnicalIndicatorServiceに委譲します。
    """

    def __init__(self):
        """初期化"""
        self.technical_indicator_service = TechnicalIndicatorService()
        self.parameter_manager = IndicatorParameterManager()

    def _create_multi_value_wrapper(self, adapter_function, index: int):
        """
        複数値指標の特定のインデックスを返すラッパー関数を作成

        Args:
            adapter_function: 元の指標関数
            index: 取得するインデックス（0=MACD線, 1=シグナル線, 2=ヒストグラム等）

        Returns:
            指定されたインデックスの値のみを返す関数
        """

        def wrapper(*args, **kwargs):
            result = adapter_function(*args, **kwargs)
            if isinstance(result, tuple) and len(result) > index:
                return result[index]
            else:
                # フォールバック: 結果がtupleでない場合はそのまま返す
                return result

        return wrapper

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

            return self.technical_indicator_service.calculate_indicator(
                data,
                indicator_type,
                parameters,
            )

        except Exception as e:
            logger.error(f"指標計算エラー ({indicator_type}): {e}")
            return None, None

    def initialize_indicator(
        self, indicator_gene: IndicatorGene, strategy_instance: Any
    ) -> Optional[List[str]]:
        """
        単一指標をbacktesting.pyに正しく登録します（動的計算版）。

        Args:
            indicator_gene (IndicatorGene): 初期化する指標の遺伝子情報。
            strategy_instance (Any): backtesting.pyの戦略インスタンス。

        Returns:
            Optional[List[str]]: 登録された指標の名前のリスト。失敗した場合はNone。
        """
        try:
            indicator_type = indicator_gene.type
            parameters = indicator_gene.parameters
            original_type = indicator_gene.type

            # 指標タイプ解決
            resolved_indicator_type = indicator_registry.resolve_indicator_type(
                indicator_type
            )
            if not resolved_indicator_type:
                logger.warning(f"未対応の指標タイプ（代替なし）: {indicator_type}")
                return None
            indicator_type = resolved_indicator_type

            # パラメータバリデーション
            indicator_config = indicator_registry.get_indicator_config(indicator_type)
            if not indicator_config:
                logger.error(f"指標設定が見つかりません: {indicator_type}")
                return None

            if not self.parameter_manager.validate_parameters(
                indicator_type, parameters, indicator_config
            ):
                logger.error(f"パラメータバリデーション失敗: {indicator_type}")
                return None

            adapter_function = indicator_config.adapter_function
            required_data_keys = indicator_config.required_data

            # 動的にデータを取得
            input_data = [
                getattr(strategy_instance.data, key.capitalize())
                for key in required_data_keys
            ]

            # パラメータ値を取得
            param_values = [
                parameters[p.name] for p in indicator_config.parameters.values()
            ]

            json_indicator_name = indicator_registry.generate_json_name(original_type)

            # 複数値指標の場合は個別に登録
            if indicator_config.result_type == IndicatorResultType.COMPLEX:
                # 複数値指標（MACD, Bollinger Bands等）の処理
                output_names = []

                # 指標の種類に応じて適切な名前を設定
                if original_type == "MACD":
                    component_names = ["macd_line", "signal_line", "histogram"]
                elif original_type == "BB":
                    component_names = ["upper", "middle", "lower"]
                elif original_type == "STOCH":
                    component_names = ["slowk", "slowd"]
                else:
                    # デフォルトの名前
                    component_names = [f"component_{i}" for i in range(3)]

                for i, component_name in enumerate(component_names):
                    # 各コンポーネント用のラッパー関数を作成
                    wrapper_function = self._create_multi_value_wrapper(
                        adapter_function, i
                    )

                    # backtesting.pyに個別に登録
                component_indicator = strategy_instance.I(
                    self.technical_indicator_service.calculate_indicator,
                    data,
                    indicator_type,
                    parameters,
                    name=f"{original_type}_{component_name}",
                )

                    # 戦略インスタンスに登録
                    component_json_name = f"{json_indicator_name}_{i}"
                    strategy_instance.indicators[component_json_name] = (
                        component_indicator
                    )
                    output_names.append(component_json_name)

                logger.info(f"複数値指標を登録: {output_names}")
                return output_names
            else:
                # 単一値指標（SMA, RSI等）の処理
            indicator_result = strategy_instance.I(
                self.technical_indicator_service.calculate_indicator,
                data,
                indicator_type,
                parameters,
                name=original_type,
            )

                strategy_instance.indicators[json_indicator_name] = indicator_result
                logger.info(f"単一値指標を登録: {json_indicator_name}")

                # 後方互換性のためのレガシー名登録
                legacy_indicator_name = self._get_legacy_indicator_name(
                    original_type, parameters
                )
                if legacy_indicator_name != json_indicator_name:
                    strategy_instance.indicators[legacy_indicator_name] = (
                        indicator_result
                    )

                return [json_indicator_name]

        except Exception as e:
            logger.error(
                f"指標初期化エラー ({indicator_gene.type}): {e}", exc_info=True
            )
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
