"""
戦略ファクトリー

StrategyGeneから動的にbacktesting.py互換のStrategy継承クラスを生成します。
責任を分離し、各機能を専用モジュールに委譲します。
"""

from typing import Type, Tuple
import logging
from backtesting import Strategy

from ..models.strategy_gene import StrategyGene, IndicatorGene
from .indicator_initializer import IndicatorInitializer
from .condition_evaluator import ConditionEvaluator
from .data_converter import DataConverter
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class StrategyFactory:
    """
    戦略ファクトリー

    StrategyGeneから動的にStrategy継承クラスを生成し、
    既存のTALibAdapterシステムと統合します。
    """

    def __init__(self):
        """初期化"""
        # 分離されたコンポーネント
        self.indicator_initializer = IndicatorInitializer()
        self.condition_evaluator = ConditionEvaluator()
        self.data_converter = DataConverter()

    def create_strategy_class(self, gene: StrategyGene) -> Type[Strategy]:
        """
        遺伝子から動的にStrategy継承クラスを生成

        Args:
            gene: 戦略遺伝子

        Returns:
            backtesting.py互換のStrategy継承クラス

        Raises:
            ValueError: 遺伝子が無効な場合
        """
        # 遺伝子の妥当性検証
        is_valid, errors = gene.validate()
        if not is_valid:
            raise ValueError(f"Invalid strategy gene: {', '.join(errors)}")

        # ファクトリー参照を保存
        factory = self

        # 動的クラス生成
        class GeneratedStrategy(Strategy):
            """動的生成された戦略クラス"""

            def __init__(self, broker=None, data=None, params=None):
                super().__init__(broker, data, params)
                self.gene = gene
                self.indicators = {}
                self.factory = factory  # ファクトリーへの参照

            def init(self):
                """指標の初期化"""
                try:
                    # 各指標を初期化
                    for indicator_gene in gene.indicators:
                        if indicator_gene.enabled:
                            self._init_indicator(indicator_gene)

                    logger.info(f"戦略初期化完了: {len(self.indicators)}個の指標")

                except Exception as e:
                    logger.error(f"戦略初期化エラー: {e}")
                    raise

            def next(self):
                """売買ロジック"""
                try:
                    # エントリー条件チェック
                    if not self.position and self._check_entry_conditions():
                        self.buy()

                    # イグジット条件チェック
                    elif self.position and self._check_exit_conditions():
                        self.sell()

                    # リスク管理
                    self._apply_risk_management()

                except Exception as e:
                    logger.error(f"売買ロジックエラー: {e}")
                    # エラーが発生してもバックテストを継続

            def _init_indicator(self, indicator_gene: IndicatorGene):
                """単一指標の初期化"""
                # IndicatorInitializerに委譲
                indicator_name = (
                    self.factory.indicator_initializer.initialize_indicator(
                        indicator_gene, self.data, self
                    )
                )
                if indicator_name:
                    logger.debug(f"指標初期化完了: {indicator_name}")
                else:
                    logger.warning(f"指標初期化失敗: {indicator_gene.type}")

            def _convert_to_series(self, bt_array):
                """backtesting.pyの_ArrayをPandas Seriesに変換"""
                return self.factory.data_converter.convert_to_series(bt_array)

            def _check_entry_conditions(self) -> bool:
                """エントリー条件をチェック"""
                return self.factory.condition_evaluator.check_entry_conditions(
                    gene.entry_conditions, self
                )

            def _check_exit_conditions(self) -> bool:
                """イグジット条件をチェック"""
                return self.factory.condition_evaluator.check_exit_conditions(
                    gene.exit_conditions, self
                )

            def _evaluate_condition(self, condition):
                """単一条件を評価"""
                return self.factory.condition_evaluator.evaluate_condition(
                    condition, self
                )

            def _get_condition_value(self, operand):
                """条件のオペランドから値を取得"""
                return self.factory.condition_evaluator.get_condition_value(
                    operand, self
                )

            def _get_condition_value_legacy(self, operand):
                """条件のオペランドから値を取得（OI/FR対応版・レガシー）"""
                try:
                    # 数値の場合
                    if isinstance(operand, (int, float)):
                        return float(operand)

                    # 文字列の場合（指標名、価格、またはOI/FR）
                    if isinstance(operand, str):
                        # 基本価格データ
                        if operand == "price" or operand == "close":
                            return self.data.Close[-1]
                        elif operand == "high":
                            return self.data.High[-1]
                        elif operand == "low":
                            return self.data.Low[-1]
                        elif operand == "open":
                            return self.data.Open[-1]
                        elif operand == "volume":
                            return self.data.Volume[-1]

                        # OI/FRデータ（新規追加）
                        elif operand == "OpenInterest":
                            return self._get_oi_fr_value("OpenInterest")
                        elif operand == "FundingRate":
                            return self._get_oi_fr_value("FundingRate")

                        # 技術指標
                        elif operand in self.indicators:
                            indicator = self.indicators[operand]
                            return indicator[-1] if len(indicator) > 0 else None

                    return None

                except Exception as e:
                    logger.error(f"オペランド値取得エラー: {e}")
                    return None

            def _get_oi_fr_value(self, data_type: str):
                """OI/FRデータから値を取得（堅牢版）"""
                try:
                    # backtesting.pyのdataオブジェクトからOI/FRデータにアクセス
                    if hasattr(self.data, data_type):
                        data_series = getattr(self.data, data_type)

                        # データ系列の型チェックと変換
                        if hasattr(data_series, "__len__") and len(data_series) > 0:
                            # pandas Series, numpy array, listなどに対応
                            try:
                                if hasattr(data_series, "iloc"):
                                    # pandas Series
                                    value = data_series.iloc[-1]
                                elif hasattr(data_series, "__getitem__"):
                                    # numpy array, list
                                    value = data_series[-1]
                                else:
                                    logger.warning(
                                        f"{data_type}データの型が不明: {type(data_series)}"
                                    )
                                    return 0.0

                                # NaN値チェック
                                if pd.isna(value) or (
                                    isinstance(value, float) and np.isnan(value)
                                ):
                                    logger.warning(
                                        f"{data_type}データにNaN値が含まれています"
                                    )
                                    # 有効な値を後ろから探す
                                    for i in range(len(data_series) - 2, -1, -1):
                                        if hasattr(data_series, "iloc"):
                                            prev_value = data_series.iloc[i]
                                        else:
                                            prev_value = data_series[i]

                                        if not pd.isna(prev_value) and not (
                                            isinstance(prev_value, float)
                                            and np.isnan(prev_value)
                                        ):
                                            return float(prev_value)

                                    # 全てNaNの場合
                                    logger.warning(f"{data_type}データが全てNaNです")
                                    return 0.0

                                return float(value)

                            except (IndexError, KeyError) as e:
                                logger.warning(
                                    f"{data_type}データのインデックスエラー: {e}"
                                )
                                return 0.0
                        else:
                            logger.warning(f"{data_type}データが空です")
                            return 0.0
                    else:
                        logger.warning(f"{data_type}データが利用できません")
                        return 0.0

                except Exception as e:
                    logger.error(f"{data_type}データ取得エラー: {e}")
                    return 0.0

            def _check_crossover(
                self,
                left_operand: str | float,
                right_operand: str | float,
                direction: str,
            ) -> bool:
                """クロスオーバーをチェック"""
                try:
                    left_current = self._get_condition_value(left_operand)
                    right_current = self._get_condition_value(right_operand)

                    # いずれかの値がNoneの場合は評価できないためFalseを返す
                    if left_current is None or right_current is None:
                        return False

                    # 前の値も取得（簡略化）
                    if len(self.data.Close) < 2:
                        return False

                    # 簡略化: 現在の値のみで判定
                    if direction == "above":
                        return left_current > right_current
                    else:  # below
                        return left_current < right_current

                except Exception as e:
                    logger.error(f"クロスオーバーチェックエラー: {e}")
                    return False

            def _apply_risk_management(self):
                """リスク管理を適用"""
                self.factory.condition_evaluator.apply_risk_management(self)

        # クラス名を設定
        GeneratedStrategy.__name__ = f"GeneratedStrategy_{gene.id}"
        GeneratedStrategy.__qualname__ = GeneratedStrategy.__name__

        return GeneratedStrategy

    def validate_gene(self, gene: StrategyGene) -> Tuple[bool, list]:
        """
        遺伝子の妥当性を詳細に検証

        Args:
            gene: 検証する戦略遺伝子

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # 基本的な妥当性チェック
        is_valid, basic_errors = gene.validate()
        errors.extend(basic_errors)

        # 指標の対応状況チェック
        for indicator in gene.indicators:
            if (
                indicator.enabled
                and not self.indicator_initializer.is_supported_indicator(
                    indicator.type
                )
            ):
                errors.append(f"未対応の指標: {indicator.type}")

        # 演算子の対応状況チェック
        for condition in gene.entry_conditions + gene.exit_conditions:
            if not self.condition_evaluator.is_supported_operator(condition.operator):
                errors.append(f"未対応の演算子: {condition.operator}")

        return len(errors) == 0, errors
