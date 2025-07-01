"""
条件評価器

エントリー・イグジット条件の評価と売買ロジックを担当するモジュール。
"""

import logging
from typing import List, Any, Optional
import pandas as pd

from ..models.strategy_gene import Condition

logger = logging.getLogger(__name__)


class ConditionEvaluator:
    """
    条件評価器

    エントリー・イグジット条件の評価と売買ロジックを担当します。
    """

    def __init__(self):
        """初期化"""
        pass

    def check_entry_conditions(
        self, entry_conditions: List[Condition], strategy_instance
    ) -> bool:
        """
        エントリー条件をチェック

        Args:
            entry_conditions: エントリー条件のリスト
            strategy_instance: 戦略インスタンス

        Returns:
            全ての条件を満たす場合True
        """
        try:
            print(f"    🔍 エントリー条件チェック開始: {len(entry_conditions)}個の条件")

            for i, condition in enumerate(entry_conditions):
                result = self.evaluate_condition(condition, strategy_instance)
                print(
                    f"      条件{i+1}: {condition.left_operand} {condition.operator} {condition.right_operand} = {result}"
                )
                if not result:
                    print(
                        f"    ❌ エントリー条件{i+1}が不満足のため、エントリーしません"
                    )
                    return False

            print(f"    ✅ 全てのエントリー条件を満足")
            return True
        except Exception as e:
            print(f"    ❌ エントリー条件チェックエラー: {e}")
            logger.error(f"エントリー条件チェックエラー: {e}")
            return False

    def check_exit_conditions(
        self, exit_conditions: List[Condition], strategy_instance
    ) -> bool:
        """
        イグジット条件をチェック

        Args:
            exit_conditions: イグジット条件のリスト
            strategy_instance: 戦略インスタンス

        Returns:
            いずれかの条件を満たす場合True
        """
        try:
            for condition in exit_conditions:
                if self.evaluate_condition(condition, strategy_instance):
                    return True
            return False
        except Exception as e:
            logger.error(f"イグジット条件チェックエラー: {e}")
            return False

    def evaluate_condition(self, condition: Condition, strategy_instance) -> bool:
        """
        単一条件を評価

        Args:
            condition: 評価する条件
            strategy_instance: 戦略インスタンス

        Returns:
            条件を満たす場合True
        """
        try:
            left_value = self.get_condition_value(
                condition.left_operand, strategy_instance
            )
            right_value = self.get_condition_value(
                condition.right_operand, strategy_instance
            )

            print(f"        → 左辺値: {condition.left_operand} = {left_value}")
            print(f"        → 右辺値: {condition.right_operand} = {right_value}")

            if left_value is None or right_value is None:
                print(f"        → 値がNoneのため条件評価失敗")
                return False

            # 演算子に基づく比較
            operator = condition.operator
            result = False
            if operator == ">":
                result = left_value > right_value
            elif operator == "<":
                result = left_value < right_value
            elif operator == ">=":
                result = left_value >= right_value
            elif operator == "<=":
                result = left_value <= right_value
            elif operator == "==":
                result = abs(left_value - right_value) < 1e-6  # 浮動小数点の比較
            elif operator == "!=":
                result = abs(left_value - right_value) >= 1e-6
            else:
                print(f"        → 未対応の演算子: {operator}")
                logger.warning(f"未対応の演算子: {operator}")
                return False

            print(
                f"        → 比較結果: {left_value} {operator} {right_value} = {result}"
            )
            return result

        except Exception as e:
            print(f"        → 条件評価エラー: {e}")
            logger.error(f"条件評価エラー: {e}")
            return False

    def get_condition_value(self, operand: Any, strategy_instance) -> Optional[float]:
        """
        条件のオペランドから値を取得

        Args:
            operand: オペランド（数値、文字列、指標名など）
            strategy_instance: 戦略インスタンス

        Returns:
            オペランドの値（取得できない場合はNone）
        """
        try:
            # 辞書形式の場合（新しい形式）
            if isinstance(operand, dict):
                op_type = operand.get("type")
                op_value = operand.get("value")

                if op_value is None:
                    logger.warning(f"オペランド辞書に 'value' がありません: {operand}")
                    return None

                if op_type == "literal":
                    return float(op_value)
                elif op_type == "indicator":
                    resolved_name = self._resolve_indicator_name(
                        str(op_value), strategy_instance
                    )
                    if resolved_name and resolved_name in strategy_instance.indicators:
                        indicator = strategy_instance.indicators[resolved_name]
                        return self._get_indicator_current_value(indicator)
                    else:
                        logger.warning(
                            f"辞書形式の指標 '{op_value}' が見つかりません。"
                        )
                        return None
                elif op_type == "price":
                    if op_value == "close":
                        return strategy_instance.data.Close[-1]
                    elif op_value == "high":
                        return strategy_instance.data.High[-1]
                    elif op_value == "low":
                        return strategy_instance.data.Low[-1]
                    elif op_value == "open":
                        return strategy_instance.data.Open[-1]
                    elif op_value == "volume":
                        return strategy_instance.data.Volume[-1]
                else:
                    logger.warning(f"未対応のオペランドタイプ: {op_type}")
                    return None

            # 数値の場合
            if isinstance(operand, (int, float)):
                return float(operand)

            # 文字列の場合（指標名、価格、またはOI/FR）
            if isinstance(operand, str):
                # 数値文字列の場合（例: "50", "30.5"）
                try:
                    return float(operand)
                except ValueError:
                    pass  # 数値でない場合は続行

                # 基本価格データ
                if operand == "price" or operand == "close":
                    return strategy_instance.data.Close[-1]
                elif operand == "high":
                    return strategy_instance.data.High[-1]
                elif operand == "low":
                    return strategy_instance.data.Low[-1]
                elif operand == "open":
                    return strategy_instance.data.Open[-1]
                elif operand == "volume":
                    return strategy_instance.data.Volume[-1]

                # OI/FRデータ（新規追加）
                elif operand == "OpenInterest":
                    return self._get_oi_fr_value("OpenInterest", strategy_instance)
                elif operand == "FundingRate":
                    return self._get_oi_fr_value("FundingRate", strategy_instance)

                # 技術指標（JSON形式対応）
                else:
                    resolved_name = self._resolve_indicator_name(
                        operand, strategy_instance
                    )
                    if resolved_name:
                        indicator = strategy_instance.indicators[resolved_name]
                        return self._get_indicator_current_value(indicator)
                    else:
                        # 指標が見つからない場合のログ出力（数値文字列の場合は警告しない）
                        if not operand.replace(".", "").replace("-", "").isdigit():
                            available_indicators = list(
                                strategy_instance.indicators.keys()
                            )
                            logger.warning(
                                f"指標 '{operand}' が見つかりません。利用可能な指標: {available_indicators}"
                            )
                        return None

            return None

        except Exception as e:
            logger.error(f"オペランド値取得エラー: {e}")
            return None

    def _get_indicator_current_value(self, indicator):
        """
        指標の現在値を安全に取得

        Args:
            indicator: 指標データ（Pandas Series、リスト、またはbacktesting.pyの_Array）

        Returns:
            現在値（最新の値）またはNone
        """
        try:
            if indicator is None:
                return None

            # backtesting.pyの_Arrayの場合（最優先でチェック）
            if hasattr(indicator, "__getitem__") and hasattr(indicator, "__len__"):
                if len(indicator) > 0:
                    value = indicator[-1]
                    # NaN チェック
                    if pd.isna(value):
                        return None
                    return float(value)

            # Pandas Seriesの場合
            elif hasattr(indicator, "iloc") and len(indicator) > 0:
                value = indicator.iloc[-1]
                # NaN チェック
                if pd.isna(value):
                    return None
                return float(value)

            # リストまたは配列の場合
            elif hasattr(indicator, "__len__") and len(indicator) > 0:
                value = indicator[-1]
                # NaN チェック
                if pd.isna(value):
                    return None
                return float(value)

            # スカラー値の場合
            elif isinstance(indicator, (int, float)):
                if pd.isna(indicator):
                    return None
                return float(indicator)

            logger.warning(f"未対応の指標タイプ: {type(indicator)}")
            return None

        except Exception as e:
            logger.error(f"指標現在値取得エラー: {e}, 指標タイプ: {type(indicator)}")
            return None

    def _resolve_indicator_name(self, operand: str, strategy_instance) -> Optional[str]:
        """
        指標名を解決（レガシー形式からJSON形式への変換対応）

        Args:
            operand: 指標名（レガシー形式またはJSON形式）
            strategy_instance: 戦略インスタンス

        Returns:
            解決された指標名（見つからない場合はNone）
        """
        try:
            # 直接存在する場合はそのまま使用（JSON形式）
            if operand in strategy_instance.indicators:
                return operand

            # レガシー形式の場合、JSON形式に変換して検索
            if "_" in operand:
                base_name = operand.split("_")[0]
                if base_name in strategy_instance.indicators:
                    return base_name

            # 特別なケース：MACD関連の指標
            if operand in ["MACD_line", "MACD_signal", "MACD_histogram"]:
                if "MACD" in strategy_instance.indicators:
                    return "MACD"

            return None

        except Exception as e:
            logger.error(f"指標名解決エラー: {e}")
            return None

    def _get_oi_fr_value(self, data_type: str, strategy_instance) -> Optional[float]:
        """
        OI/FRデータの値を取得

        Args:
            data_type: データタイプ（"OpenInterest" または "FundingRate"）
            strategy_instance: 戦略インスタンス

        Returns:
            データの値（取得できない場合はNone）
        """
        try:
            # データが利用可能かチェック
            if hasattr(strategy_instance.data, data_type):
                data_series = getattr(strategy_instance.data, data_type)
                if len(data_series) > 0:
                    return data_series[-1]

            # フォールバック値
            if data_type == "OpenInterest":
                return 1000000.0  # デフォルトOI値
            elif data_type == "FundingRate":
                return 0.0001  # デフォルトFR値（0.01%）

            return None

        except Exception as e:
            logger.error(f"OI/FRデータ取得エラー ({data_type}): {e}")
            return None

    def get_supported_operators(self) -> List[str]:
        """サポートされている演算子のリストを取得"""
        return [">", "<", ">=", "<=", "==", "!="]

    def is_supported_operator(self, operator: str) -> bool:
        """演算子がサポートされているかチェック"""
        return operator in self.get_supported_operators()
