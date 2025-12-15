"""
オペランドグループ化システム

テクニカル指標とデータソースをスケール別にグループ化し、
意味のある比較条件の生成を支援します。
"""

import logging
from enum import Enum
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# デバッグログ設定 (開発用)
DEBUG_MODE = False


class OperandGroup(Enum):
    """オペランドのスケールグループ"""

    PRICE_BASED = "price_based"  # 価格ベース（価格と同じスケール）
    PRICE_RATIO = "price_ratio"  # 価格比率（正規化指標）
    PERCENTAGE_0_100 = "percentage_0_100"  # 0-100%オシレーター
    PERCENTAGE_NEG100_100 = "percentage_neg100_100"  # ±100オシレーター
    ZERO_CENTERED = "zero_centered"  # ゼロ中心の変化率・モメンタム
    SPECIAL_SCALE = "special_scale"  # 特殊スケール（OI/FR）


class OperandGroupingSystem:
    """
    オペランドグループ化システム

    指標とデータソースを適切なスケールグループに分類し、
    同一グループ内での比較を優先する仕組みを提供します。
    """

    def __init__(self):
        """グループ化システムを初期化"""
        self._group_mappings = self._initialize_group_mappings()
        self._compatibility_matrix = self._initialize_compatibility_matrix()

    def _initialize_group_mappings(self) -> Dict[str, OperandGroup]:
        """オペランドとグループのマッピングを初期化"""
        mappings = {}

        # 価格ベース指標を追加（価格と同じスケール）
        mappings.update(self._get_price_based_mappings())

        return mappings

    def _get_price_based_mappings(self) -> Dict[str, OperandGroup]:
        """価格ベース指標マッピングを取得"""
        return {
            # Price-based indicators
            "SMA": OperandGroup.PRICE_BASED,
            "EMA": OperandGroup.PRICE_BASED,
            "close": OperandGroup.PRICE_BASED,
            "open": OperandGroup.PRICE_BASED,
            "high": OperandGroup.PRICE_BASED,
            "low": OperandGroup.PRICE_BASED,
            # 0-100%オシレーター
            "RSI": OperandGroup.PERCENTAGE_0_100,
            "STOCH": OperandGroup.PERCENTAGE_0_100,
            "ADX": OperandGroup.PERCENTAGE_0_100,
            "MFI": OperandGroup.PERCENTAGE_0_100,
            "QQE": OperandGroup.PERCENTAGE_0_100,
            # ±100オシレーター
            "CCI": OperandGroup.PERCENTAGE_NEG100_100,
            # ゼロ中心の変化率・モメンタム指標
            "MACD": OperandGroup.ZERO_CENTERED,
            "ROC": OperandGroup.ZERO_CENTERED,
            "MOM": OperandGroup.ZERO_CENTERED,
            # Trend indicators
            "SAR": OperandGroup.PRICE_BASED,
            "KELTNER": OperandGroup.PRICE_BASED,
            "DONCHIAN": OperandGroup.PRICE_BASED,
            "SUPERTREND": OperandGroup.PRICE_BASED,
            # Volatility indicators
            "ATR": OperandGroup.PRICE_BASED,
            "UI": OperandGroup.PERCENTAGE_0_100,
            # Volume indicators
            "OBV": OperandGroup.ZERO_CENTERED,
            "VWAP": OperandGroup.PRICE_BASED,
            "EFI": OperandGroup.ZERO_CENTERED,
            "CMF": OperandGroup.ZERO_CENTERED,
            # Special scale
            "volume": OperandGroup.SPECIAL_SCALE,
            "OpenInterest": OperandGroup.SPECIAL_SCALE,
            "FundingRate": OperandGroup.SPECIAL_SCALE,
            # Additional indicators
            "WILLR": OperandGroup.ZERO_CENTERED,
            "BB": OperandGroup.PRICE_BASED,
            "ACCBANDS": OperandGroup.PRICE_BASED,
            "AD": OperandGroup.ZERO_CENTERED,
            "ADOSC": OperandGroup.ZERO_CENTERED,
            "SQUEEZE": OperandGroup.ZERO_CENTERED,
        }

    def _initialize_compatibility_matrix(
        self,
    ) -> Dict[Tuple[OperandGroup, OperandGroup], float]:
        """グループ間の互換性マトリックスを初期化

        Returns:
            グループペアと互換性スコア（0.0-1.0）のマッピング
            1.0 = 完全に互換（同一グループ）
            0.8 = 高い互換性
            0.3 = 低い互換性
            0.1 = 非常に低い互換性
        """
        matrix = {}

        # 同一グループ内は完全に互換
        for group in OperandGroup:
            matrix[(group, group)] = 1.0

        # 価格ベースと価格比率は高い互換性（どちらも価格関連スケール）
        matrix[(OperandGroup.PRICE_BASED, OperandGroup.PRICE_RATIO)] = 0.8
        matrix[(OperandGroup.PRICE_RATIO, OperandGroup.PRICE_BASED)] = 0.8

        # 0-100%オシレーター同士は高い互換性
        matrix[(OperandGroup.PERCENTAGE_0_100, OperandGroup.PERCENTAGE_0_100)] = 1.0

        # ±100オシレーターとゼロ中心は中程度の互換性
        matrix[(OperandGroup.PERCENTAGE_NEG100_100, OperandGroup.ZERO_CENTERED)] = 0.3
        matrix[(OperandGroup.ZERO_CENTERED, OperandGroup.PERCENTAGE_NEG100_100)] = 0.3

        # 特殊スケール同士は低い互換性
        matrix[(OperandGroup.SPECIAL_SCALE, OperandGroup.SPECIAL_SCALE)] = 0.3

        # その他の組み合わせは非常に低い互換性
        for group1 in OperandGroup:
            for group2 in OperandGroup:
                if (group1, group2) not in matrix:
                    matrix[(group1, group2)] = 0.1

        return matrix

    def get_operand_group(self, operand: str) -> OperandGroup:
        """オペランドのグループを取得

        Args:
            operand: オペランド名

        Returns:
            オペランドのグループ
        """
        logger.debug("get_operand_group called for operand='%s'", operand)

        # 直接マッピングがある場合
        if operand in self._group_mappings:
            group = self._group_mappings[operand]
            if DEBUG_MODE:
                logger.debug("Direct mapping found for %s -> %s", operand, group.value)
            return group

        # パターンマッチングで判定
        group = self._classify_by_pattern(operand)
        if DEBUG_MODE:
            logger.debug("Pattern matching resulted for %s -> %s", operand, group.value)
        return group

    def _classify_by_pattern(self, operand: str) -> OperandGroup:
        """パターンマッチングによるオペランド分類

        Args:
            operand: オペランド名

        Returns:
            推定されるグループ
        """
        operand_upper = operand.upper()

        # 0-100%オシレーターのパターン
        if any(
            pattern in operand_upper
            for pattern in ["RSI", "STOCH", "ADX", "MFI", "QQE", "UI"]
        ):
            return OperandGroup.PERCENTAGE_0_100

        # ±100オシレーターのパターン
        if any(pattern in operand_upper for pattern in ["CCI", "CMO", "AROONOSC"]):
            return OperandGroup.PERCENTAGE_NEG100_100

        # ゼロ中心のパターン (新規指標拡張)
        if any(
            pattern in operand_upper
            for pattern in [
                "MACD",
                "MOM",
                "OBV",
                "ROC",
                "EFI",
                "CMF",
                "SQUEEZE",
                "WILLR",
                "AD",
                "ADOSC",
            ]
        ):
            return OperandGroup.ZERO_CENTERED

        # 新規Trend系パターン + ボリンジャーバンド
        if any(
            pattern in operand_upper
            for pattern in [
                "SMA",
                "EMA",
                "KELTNER",
                "DONCHIAN",
                "SUPERTREND",
                "BB",
                "ACCBANDS",
                "SAR",
            ]
        ):
            return OperandGroup.PRICE_BASED

        # 新規Volatility系パターン
        if any(
            pattern in operand_upper
            for pattern in ["KELTNER", "DONCHIAN", "SUPERTREND"]
        ):
            return OperandGroup.PRICE_BASED
        if any(pattern in operand_upper for pattern in ["ACCBANDS", "UI"]):
            return OperandGroup.PERCENTAGE_0_100

        # 新規Volume系パターン
        if any(pattern in operand_upper for pattern in ["EFI", "CMF"]):
            return OperandGroup.ZERO_CENTERED
        if "VWAP" in operand_upper:
            return OperandGroup.PRICE_BASED

        # 特殊スケールのパターン
        if any(
            pattern in operand_upper
            for pattern in ["OPENINTEREST", "FUNDING_RATE", "VOLUME"]
        ):
            return OperandGroup.SPECIAL_SCALE

        # デフォルトは価格ベース
        if DEBUG_MODE:
            logger.debug("Default classification for %s -> PRICE_BASED", operand)
        return OperandGroup.PRICE_BASED

    def get_compatibility_score(self, operand1: str, operand2: str) -> float:
        """2つのオペランド間の互換性スコアを取得

        Args:
            operand1: 第1オペランド
            operand2: 第2オペランド

        Returns:
            互換性スコア（0.0-1.0）
        """
        if DEBUG_MODE:
            logger.debug(
                "get_compatibility_score called for %s vs %s", operand1, operand2
            )

        group1 = self.get_operand_group(operand1)
        group2 = self.get_operand_group(operand2)

        score = self._compatibility_matrix.get((group1, group2), 0.1)

        if DEBUG_MODE:
            logger.debug(
                "Compatibility score for %s (%s) vs %s (%s) = %.2f",
                operand1,
                group1.value,
                operand2,
                group2.value,
                score,
            )

        return score

    def get_compatible_operands(
        self,
        target_operand: str,
        available_operands: List[str],
        min_compatibility: float = 0.8,
    ) -> List[str]:
        """指定されたオペランドと互換性の高いオペランドリストを取得

        Args:
            target_operand: 対象オペランド
            available_operands: 利用可能なオペランドリスト
            min_compatibility: 最小互換性スコア

        Returns:
            互換性の高いオペランドリスト
        """
        compatible = []

        for operand in available_operands:
            if operand == target_operand:
                continue

            score = self.get_compatibility_score(target_operand, operand)
            if score >= min_compatibility:
                compatible.append(operand)

        return compatible

    def validate_condition(
        self, left_operand: str, right_operand: str
    ) -> Tuple[bool, str]:
        """条件の妥当性を検証

        Args:
            left_operand: 左オペランド
            right_operand: 右オペランド

        Returns:
            (妥当性, 理由)のタプル
        """
        if isinstance(right_operand, (int, float)):
            # 数値との比較は常に有効
            return True, "数値との比較"

        compatibility = self.get_compatibility_score(left_operand, str(right_operand))

        if compatibility >= 0.8:
            return True, f"高い互換性 (スコア: {compatibility:.2f})"
        elif compatibility >= 0.3:
            return True, f"中程度の互換性 (スコア: {compatibility:.2f})"
        else:
            return False, f"低い互換性 (スコア: {compatibility:.2f})"


# グローバルインスタンス
operand_grouping_system = OperandGroupingSystem()


# グローバル関数インターフェース
def get_operand_group(operand: str) -> OperandGroup:
    """
    オペランドのグループを取得（グローバルインターフェース）

    Args:
        operand: オペランド名

    Returns:
        オペランドのグループ
    """
    return operand_grouping_system.get_operand_group(operand)


def _classify_by_pattern(operand: str) -> OperandGroup:
    """
    パターンマッチングによるオペランド分類（グローバルインターフェース）

    Args:
        operand: オペランド名

    Returns:
        推定されるグループ
    """
    return operand_grouping_system._classify_by_pattern(operand)


def get_compatibility_score(operand1: str, operand2: str) -> float:
    """
    2つのオペランド間の互換性スコアを取得（グローバルインターフェース）

    Args:
        operand1: 第1オペランド
        operand2: 第2オペランド

    Returns:
        互換性スコア（0.0-1.0）
    """
    return operand_grouping_system.get_compatibility_score(operand1, operand2)


def validate_condition(left_operand: str, right_operand) -> Tuple[bool, str]:
    """
    条件の妥当性を検証（グローバルインターフェース）

    Args:
        left_operand: 左オペランド
        right_operand: 右オペランド（strまたは数値）

    Returns:
        (妥当性, 理由)のタプル
    """
    return operand_grouping_system.validate_condition(left_operand, right_operand)


