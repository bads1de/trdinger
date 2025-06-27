"""
遺伝子バリデーション

戦略遺伝子の妥当性検証を担当するモジュール。
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class GeneValidator:
    """
    遺伝子バリデーター

    戦略遺伝子の妥当性検証を担当します。
    """

    def __init__(self):
        """初期化"""
        self.valid_indicator_types = self._get_valid_indicator_types()
        self.valid_operators = self._get_valid_operators()
        self.valid_data_sources = self._get_valid_data_sources()

    def _get_valid_indicator_types(self) -> List[str]:
        """有効な指標タイプのリストを取得"""
        # 共通定数から取得して一貫性を保つ
        try:
            from ...indicators.constants import ALL_INDICATORS

            return ALL_INDICATORS.copy()
        except ImportError:
            # フォールバック: オートストラテジー用10個の指標
            return [
                "SMA",  # Simple Moving Average
                "EMA",  # Exponential Moving Average
                "MACD",  # Moving Average Convergence Divergence
                "BB",  # Bollinger Bands
                "RSI",  # Relative Strength Index
                "STOCH",  # Stochastic
                "CCI",  # Commodity Channel Index
                "ADX",  # Average Directional Movement Index
                "ATR",  # Average True Range
                "OBV",  # On Balance Volume
            ]

    def _get_valid_operators(self) -> List[str]:
        """有効な演算子のリストを取得"""
        return [">", "<", ">=", "<=", "==", "cross_above", "cross_below"]

    def _get_valid_data_sources(self) -> List[str]:
        """有効なデータソースのリストを取得"""
        return [
            "close",
            "open",
            "high",
            "low",
            "volume",
            "OpenInterest",
            "FundingRate",
        ]

    def validate_indicator_gene(self, indicator_gene) -> bool:
        """
        指標遺伝子の妥当性を検証

        Args:
            indicator_gene: 指標遺伝子オブジェクト

        Returns:
            妥当性（True/False）
        """
        try:
            if not indicator_gene.type or not isinstance(indicator_gene.type, str):
                return False

            if not isinstance(indicator_gene.parameters, dict):
                return False

            # 有効な指標タイプの確認
            if indicator_gene.type not in self.valid_indicator_types:
                return False

            # パラメータの妥当性確認
            if "period" in indicator_gene.parameters:
                period = indicator_gene.parameters["period"]
                if not isinstance(period, (int, float)) or period <= 0:
                    return False

            return True

        except Exception as e:
            logger.error(f"指標遺伝子バリデーションエラー: {e}")
            return False

    def validate_condition(self, condition) -> bool:
        """
        条件の妥当性を検証

        Args:
            condition: 条件オブジェクト

        Returns:
            妥当性（True/False）
        """
        try:
            # オペレーターの検証
            if condition.operator not in self.valid_operators:
                return False

            # オペランドの検証（指標名またはデータソース名）
            if isinstance(condition.left_operand, str):
                if not (
                    self._is_indicator_name(condition.left_operand)
                    or condition.left_operand in self.valid_data_sources
                ):
                    return False

            if isinstance(condition.right_operand, str):
                if not (
                    self._is_indicator_name(condition.right_operand)
                    or condition.right_operand in self.valid_data_sources
                ):
                    return False

            return True

        except Exception as e:
            logger.error(f"条件バリデーションエラー: {e}")
            return False

    def validate_strategy_gene(self, strategy_gene) -> Tuple[bool, List[str]]:
        """
        戦略遺伝子の妥当性を検証

        Args:
            strategy_gene: 戦略遺伝子オブジェクト

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        try:
            # 指標数の制約チェック
            max_indicators = getattr(strategy_gene, "MAX_INDICATORS", 5)
            if len(strategy_gene.indicators) > max_indicators:
                errors.append(
                    f"指標数が上限({max_indicators})を超えています: {len(strategy_gene.indicators)}"
                )

            # 指標の妥当性チェック
            for i, indicator in enumerate(strategy_gene.indicators):
                if not self.validate_indicator_gene(indicator):
                    errors.append(f"指標{i}が無効です: {indicator.type}")

            # 条件の妥当性チェック
            for i, condition in enumerate(strategy_gene.entry_conditions):
                if not self.validate_condition(condition):
                    errors.append(f"エントリー条件{i}が無効です")

            for i, condition in enumerate(strategy_gene.exit_conditions):
                if not self.validate_condition(condition):
                    errors.append(f"イグジット条件{i}が無効です")

            # 最低限の条件チェック
            if not strategy_gene.entry_conditions:
                errors.append("エントリー条件が設定されていません")

            if not strategy_gene.exit_conditions:
                errors.append("イグジット条件が設定されていません")

            # 有効な指標の存在チェック
            enabled_indicators = [
                ind for ind in strategy_gene.indicators if ind.enabled
            ]
            if not enabled_indicators:
                errors.append("有効な指標が設定されていません")

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"戦略遺伝子バリデーションエラー: {e}")
            errors.append(f"バリデーション処理エラー: {e}")
            return False, errors

    def _is_indicator_name(self, name: str) -> bool:
        """
        指標名かどうかを判定

        Args:
            name: 判定対象の文字列

        Returns:
            指標名の場合True
        """
        try:
            # 指標名のパターン: "INDICATOR_PERIOD" (例: "SMA_20", "RSI_14")
            if "_" in name:
                indicator_type = name.split("_")[0]
                return indicator_type in self.valid_indicator_types

            # 単純な指標名（期間なし）
            return name in self.valid_indicator_types

        except Exception as e:
            logger.error(f"指標名判定エラー: {e}")
            return False

    def validate_risk_management(self, risk_management: dict) -> Tuple[bool, List[str]]:
        """
        リスク管理設定の妥当性を検証

        Args:
            risk_management: リスク管理設定

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        try:
            # ストップロスの検証
            if "stop_loss" in risk_management:
                stop_loss = risk_management["stop_loss"]
                if (
                    not isinstance(stop_loss, (int, float))
                    or stop_loss <= 0
                    or stop_loss >= 1
                ):
                    errors.append("ストップロスは0-1の範囲で設定してください")

            # テイクプロフィットの検証
            if "take_profit" in risk_management:
                take_profit = risk_management["take_profit"]
                if not isinstance(take_profit, (int, float)) or take_profit <= 0:
                    errors.append("テイクプロフィットは正の値で設定してください")

            # ポジションサイズの検証
            if "position_size" in risk_management:
                position_size = risk_management["position_size"]
                if (
                    not isinstance(position_size, (int, float))
                    or position_size <= 0
                    or position_size > 1
                ):
                    errors.append("ポジションサイズは0-1の範囲で設定してください")

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"リスク管理バリデーションエラー: {e}")
            errors.append(f"リスク管理バリデーション処理エラー: {e}")
            return False, errors

    def validate_metadata(self, metadata: dict) -> Tuple[bool, List[str]]:
        """
        メタデータの妥当性を検証

        Args:
            metadata: メタデータ

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        try:
            # フィットネススコアの検証
            if "fitness_score" in metadata:
                fitness_score = metadata["fitness_score"]
                if not isinstance(fitness_score, (int, float)):
                    errors.append("フィットネススコアは数値で設定してください")

            # 生成日時の検証
            if "created_at" in metadata:
                created_at = metadata["created_at"]
                if not isinstance(created_at, str):
                    errors.append("生成日時は文字列で設定してください")

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"メタデータバリデーションエラー: {e}")
            errors.append(f"メタデータバリデーション処理エラー: {e}")
            return False, errors

    def get_validation_summary(self, strategy_gene) -> dict:
        """
        バリデーション結果のサマリーを取得

        Args:
            strategy_gene: 戦略遺伝子オブジェクト

        Returns:
            バリデーション結果のサマリー
        """
        try:
            is_valid, errors = self.validate_strategy_gene(strategy_gene)

            # リスク管理とメタデータの検証
            risk_valid, risk_errors = self.validate_risk_management(
                strategy_gene.risk_management
            )
            metadata_valid, metadata_errors = self.validate_metadata(
                strategy_gene.metadata
            )

            return {
                "overall_valid": is_valid and risk_valid and metadata_valid,
                "strategy_valid": is_valid,
                "risk_management_valid": risk_valid,
                "metadata_valid": metadata_valid,
                "errors": errors + risk_errors + metadata_errors,
                "indicator_count": len(strategy_gene.indicators),
                "enabled_indicator_count": len(
                    [ind for ind in strategy_gene.indicators if ind.enabled]
                ),
                "entry_condition_count": len(strategy_gene.entry_conditions),
                "exit_condition_count": len(strategy_gene.exit_conditions),
            }

        except Exception as e:
            logger.error(f"バリデーションサマリー作成エラー: {e}")
            return {
                "overall_valid": False,
                "strategy_valid": False,
                "risk_management_valid": False,
                "metadata_valid": False,
                "errors": [f"バリデーションサマリー作成エラー: {e}"],
                "indicator_count": 0,
                "enabled_indicator_count": 0,
                "entry_condition_count": 0,
                "exit_condition_count": 0,
            }
