"""
遺伝子基底クラス

すべての遺伝子モデルの共通基底クラスを提供します。
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

logger = logging.getLogger(__name__)


class BaseGene(ABC):
    """
    遺伝子クラスの基底クラス

    PositionSizingGeneとTPSLGeneの共通機能を統合した抽象基底クラスです。
    to_dict(), from_dict(), validate() の共通実装を提供します。
    また、遺伝的操作（交叉・突然変異）のための共通インターフェースとデフォルト実装を提供します。
    """

    # 遺伝的操作のための設定（サブクラスでオーバーライド）
    NUMERIC_FIELDS: List[str] = []
    ENUM_FIELDS: List[str] = []
    CHOICE_FIELDS: List[str] = []
    NUMERIC_RANGES: Dict[str, Tuple[float, float]] = {}

    def to_dict(self) -> Dict[str, Any]:
        """オブジェクトを辞書形式に変換"""
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):  # プライベート属性は除外
                continue

            # Enumの処理
            if hasattr(value, "value"):
                result[key] = value.value
            # datetimeの処理
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            # その他の値
            else:
                result[key] = value

        return result

    @staticmethod
    def _is_enum_type(param_type) -> bool:
        """パラメータタイプがEnum型かどうかをチェック"""
        return hasattr(param_type, "__members__")

    @staticmethod
    def _is_datetime_type(param_type) -> bool:
        """パラメータタイプがdatetime型かどうかをチェック"""
        return param_type == datetime

    @staticmethod
    def _convert_enum_value(value: Any, param_type) -> Any:
        """
        値をEnum型に変換

        文字列が渡された場合、指定されたEnumクラスのメンバーに変換を試みます。
        変換に失敗した場合は、Enumの最初のメンバーをデフォルト値として返します。

        Args:
            value: 変換対象の値
            param_type: 目標とするEnumクラス

        Returns:
            Enumメンバーインスタンス
        """
        if isinstance(value, str):
            try:
                return param_type(value)
            except ValueError:
                logger.warning(f"無効なEnum値 {value} を無視、デフォルト値を設定")
                # Enumの最初の値をデフォルトとして返す
                return next(iter(param_type))
        return value

    @staticmethod
    def _convert_datetime_value(value: Any) -> Any:
        """
        値をdatetime型に変換（ISOフォーマット対応）

        文字列が渡された場合、ISO 8601フォーマットからのパースを試みます。
        失敗した場合は現在時刻を返します。

        Args:
            value: 変換対象の値

        Returns:
            datetimeオブジェクト
        """
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                logger.warning(f"無効なdatetime値 {value} を無視、デフォルト値を設定")
                return datetime.now()  # デフォルトとして現在時刻
        return value

    @staticmethod
    def _convert_value(value: Any, param_type) -> Any:
        """一般的な値変換"""
        if BaseGene._is_enum_type(param_type):
            return BaseGene._convert_enum_value(value, param_type)
        elif BaseGene._is_datetime_type(param_type):
            return BaseGene._convert_datetime_value(value)
        else:
            return value

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any:
        """辞書形式からオブジェクトを復元"""
        init_params = {}

        # クラスアノテーションを取得（継承チェーンを含む）
        annotations = {}
        for base in reversed(cls.__mro__):
            annotations.update(getattr(base, "__annotations__", {}))

        # 1. アノテーションがあるフィールドについては型変換を試みる
        for param_name, param_type in annotations.items():
            if param_name in data:
                raw_value = data[param_name]
                init_params[param_name] = cls._convert_value(raw_value, param_type)

        # 2. アノテーションにないフィールドもすべて含める（型変換なし）
        for key, value in data.items():
            if key not in init_params:
                init_params[key] = value

        # 3. 実際のクラス生成
        # クラスが受け取れない引数が含まれている場合のガード（任意だが、通常は**init_paramsで十分）
        return cls(**init_params)

    def validate(self) -> Tuple[bool, List[str]]:
        """遺伝子の妥当性を検証"""
        errors = []

        try:
            # 基本的な属性チェック
            if hasattr(self, "enabled") and not isinstance(
                getattr(self, "enabled", True), bool
            ):
                errors.append("enabled属性がbool型である必要があります")

            # サブクラス固有の検証を呼び出し
            self._validate_parameters(errors)

        except Exception as e:
            errors.append(f"検証処理でエラーが発生: {e}")

        return len(errors) == 0, errors

    @abstractmethod
    def _validate_parameters(self, errors: List[str]) -> None:
        """サブクラスで固有のパラメータ検証を実装"""

    def _validate_range(
        self,
        value: Union[int, float],
        min_val: Union[int, float],
        max_val: Union[int, float],
        param_name: str,
        errors: List[str],
    ) -> bool:
        """
        数値パラメータの範囲検証

        Args:
            value: 判定対象の値
            min_val: 最小許容値
            max_val: 最大許容値
            param_name: パラメータ名（エラーメッセージ用）
            errors: エラーメッセージを追記するリスト

        Returns:
            範囲内であればTrue
        """
        if not (min_val <= value <= max_val):
            errors.append(
                f"{param_name}は{min_val}-{max_val}の範囲である必要があります"
            )
            return False
        return True

    def mutate(self, mutation_rate: float = 0.1) -> "BaseGene":
        """
        遺伝子の突然変異（デフォルト実装）
        GeneticUtils.mutate_generic_geneを使用します。
        特殊なロジックが必要な場合はサブクラスでオーバーライドしてください。
        """
        from .genetic_utils import GeneticUtils

        return GeneticUtils.mutate_generic_gene(
            gene=self,
            gene_class=self.__class__,
            mutation_rate=mutation_rate,
            numeric_fields=self.NUMERIC_FIELDS,
            enum_fields=self.ENUM_FIELDS,
            numeric_ranges=self.NUMERIC_RANGES,
        )

    @classmethod
    def crossover(
        cls, parent1: "BaseGene", parent2: "BaseGene"
    ) -> Tuple["BaseGene", "BaseGene"]:
        """
        遺伝子の交叉（デフォルト実装）
        GeneticUtils.crossover_generic_genesを使用します。
        特殊なロジックが必要な場合はサブクラスでオーバーライドしてください。
        """
        from .genetic_utils import GeneticUtils

        return GeneticUtils.crossover_generic_genes(
            parent1_gene=parent1,
            parent2_gene=parent2,
            gene_class=cls,
            numeric_fields=cls.NUMERIC_FIELDS,
            enum_fields=cls.ENUM_FIELDS,
            choice_fields=cls.CHOICE_FIELDS,
        )
