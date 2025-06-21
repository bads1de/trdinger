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
        self, 
        entry_conditions: List[Condition], 
        strategy_instance
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
        self, 
        exit_conditions: List[Condition], 
        strategy_instance
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
            left_value = self.get_condition_value(condition.left_operand, strategy_instance)
            right_value = self.get_condition_value(condition.right_operand, strategy_instance)

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
            elif operator == "cross_above":
                return self._check_cross_above(left_value, right_value, strategy_instance)
            elif operator == "cross_below":
                return self._check_cross_below(left_value, right_value, strategy_instance)
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

                # 技術指標
                elif operand in strategy_instance.indicators:
                    indicator = strategy_instance.indicators[operand]
                    return indicator[-1] if len(indicator) > 0 else None

            return None

        except Exception as e:
            logger.error(f"オペランド値取得エラー: {e}")
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

    def _check_cross_above(
        self, 
        left_value: float, 
        right_value: float, 
        strategy_instance
    ) -> bool:
        """
        クロスアバブ（上抜け）をチェック
        
        Args:
            left_value: 左オペランドの現在値
            right_value: 右オペランドの現在値
            strategy_instance: 戦略インスタンス
            
        Returns:
            上抜けが発生した場合True
        """
        try:
            # 最低2つのデータポイントが必要
            if len(strategy_instance.data.Close) < 2:
                return False

            # 前回の値を取得（簡略化：価格データを使用）
            prev_left = strategy_instance.data.Close[-2]
            prev_right = right_value  # 右側は通常固定値または指標

            # クロスアバブ条件：前回は下にあり、現在は上にある
            return prev_left <= prev_right and left_value > right_value

        except Exception as e:
            logger.error(f"クロスアバブチェックエラー: {e}")
            return False

    def _check_cross_below(
        self, 
        left_value: float, 
        right_value: float, 
        strategy_instance
    ) -> bool:
        """
        クロスビロー（下抜け）をチェック
        
        Args:
            left_value: 左オペランドの現在値
            right_value: 右オペランドの現在値
            strategy_instance: 戦略インスタンス
            
        Returns:
            下抜けが発生した場合True
        """
        try:
            # 最低2つのデータポイントが必要
            if len(strategy_instance.data.Close) < 2:
                return False

            # 前回の値を取得（簡略化：価格データを使用）
            prev_left = strategy_instance.data.Close[-2]
            prev_right = right_value  # 右側は通常固定値または指標

            # クロスビロー条件：前回は上にあり、現在は下にある
            return prev_left >= prev_right and left_value < right_value

        except Exception as e:
            logger.error(f"クロスビローチェックエラー: {e}")
            return False

    def apply_risk_management(self, strategy_instance):
        """
        リスク管理を適用
        
        Args:
            strategy_instance: 戦略インスタンス
        """
        try:
            # 基本的なリスク管理（将来拡張可能）
            if hasattr(strategy_instance, 'position') and strategy_instance.position:
                # ポジションサイズの制限
                max_position_size = getattr(strategy_instance, 'max_position_size', 1.0)
                if abs(strategy_instance.position.size) > max_position_size:
                    logger.warning("ポジションサイズが上限を超過")

                # 最大保有期間の制限
                max_holding_period = getattr(strategy_instance, 'max_holding_period', 100)
                if hasattr(strategy_instance.position, 'entry_bar'):
                    holding_period = len(strategy_instance.data.Close) - strategy_instance.position.entry_bar
                    if holding_period > max_holding_period:
                        logger.info("最大保有期間に達したため決済")
                        strategy_instance.position.close()

        except Exception as e:
            logger.error(f"リスク管理エラー: {e}")

    def get_supported_operators(self) -> List[str]:
        """サポートされている演算子のリストを取得"""
        return [">", "<", ">=", "<=", "==", "!=", "cross_above", "cross_below"]

    def is_supported_operator(self, operator: str) -> bool:
        """演算子がサポートされているかチェック"""
        return operator in self.get_supported_operators()
