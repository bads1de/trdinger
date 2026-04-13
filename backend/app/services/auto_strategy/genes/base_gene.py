"""
遺伝子基底クラス

すべての遺伝子モデルの共通基底クラスを提供します。
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import is_dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Union, get_type_hints
from typing_extensions import Self

from app.utils.serialization import dataclass_to_dict

from app.types import SerializableValue

logger = logging.getLogger(__name__)


class BaseGene(ABC):
    """
    遺伝子クラスの基底クラス

    PositionSizingGeneとTPSLGeneの共通機能を統合した抽象基底クラスです。
    to_dict(), from_dict(), validate() の共通実装を提供します。
    また、遺伝的操作（交叉・突然変異）のための共通インターフェースとデフォルト実装を提供します。
    """

    __slots__ = ()

    # 遺伝的操作のための設定（サブクラスでオーバーライド）
    NUMERIC_FIELDS: List[str] = []
    ENUM_FIELDS: List[str] = []
    CHOICE_FIELDS: List[str] = []
    NUMERIC_RANGES: Dict[str, Tuple[float, float]] = {}
    _SKIP_FIELD_CONVERSION = object()

    def to_dict(self) -> dict[str, SerializableValue]:
        """オブジェクトを辞書形式に変換"""
        # dataclass サブクラスは汎用ユーティリティに委譲
        if is_dataclass(self):
            return dataclass_to_dict(self)

        # 非 dataclass サブクラス向けフォールバック
        result = {}

        keys: List[str] = []
        if hasattr(self, "__slots__"):
            keys = list(self.__slots__)
        elif hasattr(self, "__dict__"):
            keys = list(self.__dict__.keys())

        for key in keys:
            if key.startswith("_"):
                continue

            value = getattr(self, key, None)

            if value is not None and hasattr(value, "value"):
                result[key] = value.value
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
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
    def _convert_enum_value(
        value: object, param_type: type
    ) -> SerializableValue | object:
        """
        値をEnum型に変換

        文字列が渡された場合、指定されたEnumクラスのメンバーに変換を試みます。
        変換に失敗した場合はフィールド設定をスキップし、クラス既定値へフォールバックします。

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
                logger.warning(f"無効なEnum値 {value} を無視、既定値へフォールバック")
                return BaseGene._SKIP_FIELD_CONVERSION
        return value

    @staticmethod
    def _convert_datetime_value(value: object) -> datetime | object:
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
    def _convert_value(value: object, param_type: type) -> SerializableValue | object:
        """一般的な値変換"""
        if BaseGene._is_enum_type(param_type):
            return BaseGene._convert_enum_value(value, param_type)
        elif BaseGene._is_datetime_type(param_type):
            return BaseGene._convert_datetime_value(value)
        else:
            return value

    @classmethod
    def _get_resolved_annotations(cls) -> dict[str, type]:
        """継承階層を含む型注釈を解決済みで取得する。"""
        annotations: dict[str, type] = {}

        # Collect globalns from ALL modules in the MRO (not just cls.__module__)
        # This handles cases where base classes are defined in different modules
        # that import typing constructs (Tuple, List, Dict, etc.)
        combined_globalns: dict[str, object] = {}
        for base in cls.__mro__:
            base_module = sys.modules.get(getattr(base, '__module__', ''))
            if base_module is not None:
                try:
                    combined_globalns.update(vars(base_module))
                except Exception as e:
                    logger.debug("モジュール変数の更新に失敗しました (%s): %s", base.__name__, e)

        localns: dict[str, object] = {}
        for base in cls.__mro__:
            try:
                localns.update(vars(base))
            except TypeError:
                continue

        for base in reversed(cls.__mro__):
            raw_annotations = getattr(base, "__annotations__", {})
            if not raw_annotations:
                continue

            try:
                annotations.update(
                    get_type_hints(base, globalns=combined_globalns, localns=localns)
                )
            except Exception as e:
                # Try per-annotation to salvage what we can
                logger.debug("型ヒントの取得に失敗しました (%s): %s", base.__name__, e)
                for name, raw in raw_annotations.items():
                    if name in annotations:
                        continue
                    try:
                        resolved = get_type_hints(
                            type(name, (), {name: raw}),
                            globalns=combined_globalns,
                            localns=localns,
                        )
                        annotations[name] = resolved[name]
                    except Exception as e:
                        logger.debug("個別アノテーションの解決に失敗しました (%s): %s", name, e)
                        annotations[name] = raw  # type: ignore[assignment]

        return annotations

    @classmethod
    def from_dict(cls, data: dict[str, SerializableValue]) -> Self:
        """辞書形式からオブジェクトを復元"""
        init_params = {}
        skipped_fields = set()

        # クラスアノテーションを取得（継承チェーンを含む）
        annotations = cls._get_resolved_annotations()

        # 1. アノテーションがあるフィールドについては型変換を試みる
        for param_name, param_type in annotations.items():
            if param_name in data:
                raw_value = data[param_name]
                converted_value = cls._convert_value(raw_value, param_type)
                if converted_value is cls._SKIP_FIELD_CONVERSION:
                    skipped_fields.add(param_name)
                    continue
                init_params[param_name] = converted_value

        # 2. アノテーションにないフィールドもすべて含める（型変換なし）
        for key, value in data.items():
            if key not in init_params and key not in skipped_fields:
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
