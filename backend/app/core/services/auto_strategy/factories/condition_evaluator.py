"""
条件評価器

エントリー・イグジット条件の評価と売買ロジックを担当するモジュール。
"""

import logging
from typing import List, Any, Optional

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
            for condition in entry_conditions:
                if not self.evaluate_condition(condition, strategy_instance):
                    return False
            return True
        except Exception as e:
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

            if left_value is None or right_value is None:
                return False

            # 演算子に基づく比較
            operator = condition.operator
            if operator == ">":
                return left_value > right_value
            elif operator == "<":
                return left_value < right_value
            elif operator == ">=":
                return left_value >= right_value
            elif operator == "<=":
                return left_value <= right_value
            elif operator == "==":
                return abs(left_value - right_value) < 1e-6  # 浮動小数点の比較
            elif operator == "!=":
                return abs(left_value - right_value) >= 1e-6
            else:
                logger.warning(f"未対応の演算子: {operator}")
                return False

        except Exception as e:
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
                        return indicator[-1] if len(indicator) > 0 else None
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
                    logger.debug(
                        f"レガシー形式の指標名 '{operand}' をJSON形式 '{base_name}' に変換"
                    )
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
