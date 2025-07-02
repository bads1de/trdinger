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
            # print(f"    🔍 エントリー条件チェック開始: {len(entry_conditions)}個の条件")

            for i, condition in enumerate(entry_conditions):
                result = self.evaluate_condition(condition, strategy_instance)
                # print(
                #     f"      条件{i+1}: {condition.left_operand} {condition.operator} {condition.right_operand} = {result}"
                # )
                if not result:
                    # print(
                    #     f"    ❌ エントリー条件{i+1}が不満足のため、エントリーしません"
                    # )
                    return False

            # print(f"    ✅ 全てのエントリー条件を満足")
            return True
        except Exception as e:
            # print(f"    ❌ エントリー条件チェックエラー: {e}")
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

            # print(f"        → 左辺値: {condition.left_operand} = {left_value}")
            # print(f"        → 右辺値: {condition.right_operand} = {right_value}")

            if left_value is None or right_value is None:
                # print(f"        → 値がNoneのため条件評価失敗")
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
                # print(f"        → 未対応の演算子: {operator}")
                logger.warning(f"未対応の演算子: {operator}")
                return False

            # print(
            #     f"        → 比較結果: {left_value} {operator} {right_value} = {result}"
            # )
            return result

        except Exception as e:
            # print(f"        → 条件評価エラー: {e}")
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
        指標名を解決（新しい動的指標システム対応）

        Args:
            operand: 指標名（基本名、コンポーネント名、またはレガシー形式）
            strategy_instance: 戦略インスタンス

        Returns:
            解決された指標名（見つからない場合はNone）
        """
        try:
            # 1. 完全一致を試す (例: "MACD_0", "RSI", "SMA_20")
            if operand in strategy_instance.indicators:
                return operand

            # 2. 複数値指標の特別な名前解決ルール
            resolved_name = self._resolve_multi_value_indicator(
                operand, strategy_instance
            )
            if resolved_name:
                return resolved_name

            # 3. 前方一致を試す (例: "MACD" が "MACD_0" にマッチ)
            for indicator_name in strategy_instance.indicators:
                if indicator_name.startswith(operand):
                    logger.debug(
                        f"前方一致で指標名を解決: '{operand}' -> '{indicator_name}'"
                    )
                    return indicator_name

            # 4. レガシー形式の解決を試す
            legacy_resolved = self._resolve_legacy_indicator_name(
                operand, strategy_instance
            )
            if legacy_resolved:
                return legacy_resolved

            logger.warning(
                f"指標名 '{operand}' が解決できませんでした。利用可能な指標: {list(strategy_instance.indicators.keys())}"
            )
            return None
        except Exception as e:
            logger.error(f"指標名解決エラー: {e}", exc_info=True)
            return None

    def _resolve_multi_value_indicator(
        self, operand: str, strategy_instance
    ) -> Optional[str]:
        """
        複数値指標の名前解決（新しい動的指標システム対応）

        Args:
            operand: 指標名（基本名またはコンポーネント指定）
            strategy_instance: 戦略インスタンス

        Returns:
            解決された指標名（見つからない場合はNone）
        """
        # 複数値指標のコンポーネント名マッピング
        multi_value_mappings = {
            # MACD関連
            "MACD": "MACD_0",  # デフォルトはMACD線
            "MACD_line": "MACD_0",  # MACD線
            "MACD_signal": "MACD_1",  # シグナル線
            "MACD_histogram": "MACD_2",  # ヒストグラム
            "macd_line": "MACD_0",  # 小文字対応
            "macd_signal": "MACD_1",
            "macd_histogram": "MACD_2",
            # Bollinger Bands関連
            "BB": "BB_1",  # デフォルトは中央線（SMA）
            "BB_upper": "BB_0",  # 上限バンド
            "BB_middle": "BB_1",  # 中央線
            "BB_lower": "BB_2",  # 下限バンド
            "bb_upper": "BB_0",  # 小文字対応
            "bb_middle": "BB_1",
            "bb_lower": "BB_2",
            "BollingerBands": "BB_1",  # フル名
            "BollingerBands_upper": "BB_0",
            "BollingerBands_middle": "BB_1",
            "BollingerBands_lower": "BB_2",
            # Stochastic関連
            "STOCH": "STOCH_0",  # デフォルトは%K
            "STOCH_K": "STOCH_0",  # %K
            "STOCH_D": "STOCH_1",  # %D
            "stoch_k": "STOCH_0",  # 小文字対応
            "stoch_d": "STOCH_1",
            "Stochastic": "STOCH_0",  # フル名
            "Stochastic_K": "STOCH_0",
            "Stochastic_D": "STOCH_1",
        }

        # マッピングテーブルから解決を試す
        if operand in multi_value_mappings:
            target_name = multi_value_mappings[operand]
            if target_name in strategy_instance.indicators:
                logger.debug(f"複数値指標名を解決: '{operand}' -> '{target_name}'")
                return target_name

        return None

    def _resolve_legacy_indicator_name(
        self, operand: str, strategy_instance
    ) -> Optional[str]:
        """
        レガシー形式の指標名解決

        Args:
            operand: レガシー形式の指標名（例: "SMA_20", "RSI_14"）
            strategy_instance: 戦略インスタンス

        Returns:
            解決された指標名（見つからない場合はNone）
        """
        # レガシー形式から新しい形式への変換を試す
        # 例: "SMA_20" -> "SMA", "RSI_14" -> "RSI"

        # アンダースコア区切りの場合、基本名を抽出
        if "_" in operand:
            base_name = operand.split("_")[0]
            if base_name in strategy_instance.indicators:
                logger.debug(f"レガシー指標名を解決: '{operand}' -> '{base_name}'")
                return base_name

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
