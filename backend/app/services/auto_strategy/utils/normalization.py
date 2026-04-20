"""
正規化ユーティリティ関数

auto_strategy全体で使用されるパラメータ正規化・Enum正規化の共通機能を提供します。
"""

import logging
from enum import Enum
from typing import Any, Union

logger = logging.getLogger(__name__)


class NormalizationUtils:
    """正規化ユーティリティクラス

    パラメータの正規化、Enum名の正規化、デフォルト遺伝子生成などの
    共通機能を提供します。
    """

    @staticmethod
    def normalize_parameter(
        value: Union[int, float], min_val: int = 1, max_val: int = 200
    ) -> float:
        """パラメータ値を正規化（0-1の範囲に変換）

        指定された最小値と最大値の範囲に値をクランプし、
        0-1の範囲に正規化します。

        Args:
            value: 正規化対象の値
            min_val: 最小値（デフォルト: 1）
            max_val: 最大値（デフォルト: 200）

        Returns:
            float: 0-1の範囲に正規化された値。
                値がmin_val未満の場合は0.0、max_val超の場合は1.0になります。
                valueが数値でない場合は警告をログ出力し、0.1を返します。

        Note:
            - min_val >= max_valの場合、ゼロ除算が発生するため適切な範囲を指定してください。
            - 値が範囲外の場合、クランプ处理后に正規化されます。
        """
        if not isinstance(value, (int, float)):
            logger.warning(
                f"数値でないパラメータを正規化: {value}, デフォルト値0.1を返却"
            )
            return 0.1

        # 範囲内に制限
        clamped_value = max(min_val, min(max_val, value))

        # 0-1の範囲に正規化
        normalized = (clamped_value - min_val) / (max_val - min_val)

        return float(normalized)

    @staticmethod
    def normalize_enum_name(value: Any, default: str = "") -> str:
        """Enumまたは文字列を比較用の文字列に正規化

        Enum値や文字列、その他の値を文字列表現に変換します。
        Enum値の場合は .value 属性を抽出します。

        Args:
            value: 正規化する値（Enum、文字列、またはNone）。
            default: 値がNoneの場合に返すデフォルト文字列。

        Returns:
            str: 正規化された文字列。
        """
        if value is None:
            return default
        if hasattr(value, "value"):
            value = value.value
        return str(value)

    @staticmethod
    def create_default_strategy_gene(strategy_gene_class: type) -> Any:
        """デフォルトの戦略遺伝子を作成

        StrategyGene.create_defaultメソッドに委譲して、
        指定されたクラスのデフォルトインスタンスを生成します。

        Args:
            strategy_gene_class: デフォルトインスタンスを生成する遺伝子クラス。
                create_defaultクラスメソッドを持っている必要があります。

        Returns:
            指定されたクラスのデフォルトインスタンス。

        Raises:
            AttributeError: strategy_gene_classにcreate_defaultメソッドが存在しない場合。
        """
        return strategy_gene_class.create_default()


# 外部で使用可能な便利関数
create_default_strategy_gene = NormalizationUtils.create_default_strategy_gene
normalize_parameter = NormalizationUtils.normalize_parameter
normalize_enum_name = NormalizationUtils.normalize_enum_name
