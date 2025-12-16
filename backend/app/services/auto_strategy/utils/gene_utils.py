"""
遺伝子関連ユーティリティ関数

auto_strategy全体で使用される遺伝子関連の共通機能を提供します。
"""

import logging
import random
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Gene関連クラスの基底クラスと共通機能
# =============================================================================


class BaseGene(ABC):
    """
    遺伝子クラスの基底クラス

    PositionSizingGeneとTPSLGeneの共通機能を統合した抽象基底クラスです。
    to_dict(), from_dict(), validate() の共通実装を提供します。
    """

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
        """Enum型への変換"""
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
        """datetime型への変換"""
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

        # クラスアノテーションがある場合
        if hasattr(cls, "__annotations__") and cls.__annotations__:
            annotations = cls.__annotations__

            for param_name, param_type in annotations.items():
                if param_name in data:
                    raw_value = data[param_name]
                    init_params[param_name] = cls._convert_value(raw_value, param_type)
        else:
            # アノテーションがないまたは空の場合はデータ全体を使用
            logger.warning(
                f"クラス {cls.__name__} に型アノテーションがないため、辞書データを直接使用"
            )
            init_params = data.copy()

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
        """範囲検証のヘルパー関数"""
        if not (min_val <= value <= max_val):
            errors.append(
                f"{param_name}は{min_val}-{max_val}の範囲である必要があります"
            )
            return False
        return True


class GeneticUtils:
    """遺伝的アルゴリズム関連ユーティリティ"""

    @staticmethod
    def create_child_metadata(
        parent1_metadata: Dict[str, Any],
        parent2_metadata: Dict[str, Any],
        parent1_id: str,
        parent2_id: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        交叉子のメタデータを生成

        Args:
            parent1_metadata: 親1のメタデータ
            parent2_metadata: 親2のメタデータ
            parent1_id: 親1のID
            parent2_id: 親2のID

        Returns:
            (child1_metadata, child2_metadata) のタプル
        """
        # 子1のメタデータ：親1のメタデータを継承し、親情報を追加
        child1_metadata = parent1_metadata.copy()
        child1_metadata["crossover_parent1"] = parent1_id
        child1_metadata["crossover_parent2"] = parent2_id

        # 子2のメタデータ：親2のメタデータを継承し、親情報を追加
        child2_metadata = parent2_metadata.copy()
        child2_metadata["crossover_parent1"] = parent1_id
        child2_metadata["crossover_parent2"] = parent2_id

        return child1_metadata, child2_metadata

    @staticmethod
    def prepare_crossover_metadata(strategy_parent1, strategy_parent2):
        """
        戦略遺伝子交叉用のメタデータ作成ユーティリティ

        Args:
            strategy_parent1: 親1のStrategyGene
            strategy_parent2: 親2のStrategyGene

        Returns:
            (child1_metadata, child2_metadata) のタプル
        """
        return GeneticUtils.create_child_metadata(
            parent1_metadata=strategy_parent1.metadata,
            parent2_metadata=strategy_parent2.metadata,
            parent1_id=strategy_parent1.id,
            parent2_id=strategy_parent2.id,
        )

    @staticmethod
    def crossover_generic_genes(
        parent1_gene,
        parent2_gene,
        gene_class,
        numeric_fields: Optional[List[str]] = None,
        enum_fields: Optional[List[str]] = None,
        choice_fields: Optional[List[str]] = None,
    ):
        """
        ジェネリック遺伝子交叉関数

        Args:
            parent1_gene: 親1遺伝子
            parent2_gene: 親2遺伝子
            gene_class: 遺伝子クラス
            numeric_fields: 数値フィールドのリスト（平均化）
            enum_fields: Enumフィールドのリスト（ランダム選択）
            choice_fields: どちらかを選択するフィールドのリスト

        Returns:
            (child1_gene, child2_gene) のタプル
        """
        if numeric_fields is None:
            numeric_fields = []
        if enum_fields is None:
            enum_fields = []
        if choice_fields is None:
            choice_fields = []

        # 共通のフィールド処理
        parent1_dict = parent1_gene.__dict__.copy()
        parent2_dict = parent2_gene.__dict__.copy()

        child1_params = {}
        child2_params = {}

        # 全フィールドを取得
        all_fields = set(parent1_dict.keys()) & set(parent2_dict.keys())
        all_fields.discard("_")  # プライベート属性を除外

        for field in all_fields:
            if field in choice_fields:
                # どちらかを選択
                child1_params[field] = random.choice(
                    [parent1_dict[field], parent2_dict[field]]
                )
                child2_params[field] = random.choice(
                    [parent1_dict[field], parent2_dict[field]]
                )
            elif field in numeric_fields:
                # 数値フィールドは平均化
                val1 = parent1_dict[field]
                val2 = parent2_dict[field]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    child1_params[field] = (val1 + val2) / 2
                    child2_params[field] = (val1 + val2) / 2
                else:
                    child1_params[field] = val1
                    child2_params[field] = val2
            elif field in enum_fields:
                # Enumフィールドはランダム選択
                child1_params[field] = random.choice(
                    [parent1_dict[field], parent2_dict[field]]
                )
                child2_params[field] = random.choice(
                    [parent1_dict[field], parent2_dict[field]]
                )
            else:
                # デフォルト：交互に選択
                if random.random() < 0.5:
                    child1_params[field] = parent1_dict[field]
                    child2_params[field] = parent2_dict[field]
                else:
                    child1_params[field] = parent2_dict[field]
                    child2_params[field] = parent1_dict[field]

        # 子遺伝子作成
        child1 = gene_class(**child1_params)
        child2 = gene_class(**child2_params)

        return child1, child2

    @staticmethod
    def mutate_generic_gene(
        gene,
        gene_class,
        mutation_rate: float = 0.1,
        numeric_fields: Optional[List[str]] = None,
        enum_fields: Optional[List[str]] = None,
        numeric_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        ジェネリック遺伝子突然変異関数

        Args:
            gene: 突然変異対象の遺伝子
            gene_class: 遺伝子クラス
            mutation_rate: 突然変異率
            numeric_fields: 数値フィールドのリスト
            enum_fields: Enumフィールドのリスト
            numeric_ranges: 数値フィールドの範囲 {"field_name": (min_val, max_val)}

        Returns:
            突然変異後の遺伝子
        """
        import random

        if numeric_fields is None:
            numeric_fields = []
        if enum_fields is None:
            enum_fields = []
        if numeric_ranges is None:
            numeric_ranges = {}

        # 遺伝子のコピーを作成
        mutated_params = gene.__dict__.copy()

        # 数値フィールドの突然変異
        for field in numeric_fields:
            if random.random() < mutation_rate and field in mutated_params:
                current_value = mutated_params[field]
                if isinstance(current_value, (int, float)):
                    # 値の範囲を取得（指定がない場合はデフォルト範囲）
                    min_val, max_val = numeric_ranges.get(field, (0, 100))

                    # 現在の値を中心とした範囲で突然変異
                    mutation_factor = random.uniform(0.8, 1.2)
                    new_value = current_value * mutation_factor

                    # 範囲内に制限
                    new_value = max(min_val, min(max_val, new_value))

                    mutated_params[field] = new_value

                    # int型だった場合はintに変換
                    if isinstance(current_value, int):
                        mutated_params[field] = int(new_value)

        # Enumフィールドの突然変異
        for field in enum_fields:
            if random.random() < mutation_rate and field in mutated_params:
                current_enum = mutated_params[field]
                if hasattr(current_enum, "__class__"):
                    enum_class = current_enum.__class__
                    # ランダムに別の値を選択
                    mutated_params[field] = random.choice(list(enum_class))

        return gene_class(**mutated_params)


class GeneUtils:
    """遺伝子関連ユーティリティ"""

    @staticmethod
    def normalize_parameter(
        value: Union[int, float], min_val: int = 1, max_val: int = 200
    ) -> float:
        """
        パラメータ値を正規化（0-1の範囲に変換）

        Args:
            value: 正規化対象の値
            min_val: 最小値（デフォルト: 1）
            max_val: 最大値（デフォルト: 200）

        Returns:
            0-1の範囲に正規化した値
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
    def create_default_strategy_gene(strategy_gene_class):
        """デフォルトの戦略遺伝子を作成"""
        try:
            # 動的インポートを避けるため、引数として渡すか、呼び出し側でインポートする
            # ここでは基本的な構造のみを提供
            from ..genes import (
                Condition,
                IndicatorGene,
                PositionSizingGene,
                PositionSizingMethod,
                TPSLGene,
            )

            indicators = [
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ]

            # デフォルト条件
            long_entry_conditions = [
                Condition(left_operand="close", operator=">", right_operand="open")
            ]
            short_entry_conditions = [
                Condition(left_operand="close", operator="<", right_operand="open")
            ]

            # デフォルトリスク管理
            risk_management = {"position_size": 0.1}

            # デフォルトTP/SL遺伝子
            tpsl_gene = TPSLGene(
                take_profit_pct=0.01, stop_loss_pct=0.005, enabled=True
            )

            # デフォルトポジションサイジング遺伝子
            position_sizing_gene = PositionSizingGene(
                method=PositionSizingMethod.FIXED_QUANTITY,
                fixed_quantity=1000,
                enabled=True,
            )

            # メタデータ
            metadata = {
                "generated_by": "create_default_strategy_gene",
                "source": "fallback",
                "indicators_count": len(indicators),
                "tpsl_gene_included": tpsl_gene is not None,
                "position_sizing_gene_included": position_sizing_gene is not None,
            }

            return strategy_gene_class(
                id=str(uuid.uuid4()),  # 新しいIDを生成
                indicators=indicators,
                long_entry_conditions=long_entry_conditions,
                short_entry_conditions=short_entry_conditions,
                risk_management=risk_management,
                tpsl_gene=tpsl_gene,
                position_sizing_gene=position_sizing_gene,
                metadata=metadata,
            )
        except Exception as inner_e:
            logger.error(f"デフォルト戦略遺伝子作成エラー: {inner_e}")
            raise ValueError(f"デフォルト戦略遺伝子の作成に失敗: {inner_e}")


# 外部で使用可能な便利関数
create_default_strategy_gene = GeneUtils.create_default_strategy_gene
normalize_parameter = GeneUtils.normalize_parameter

# GeneticUtilsの便利関数
create_child_metadata = GeneticUtils.create_child_metadata
prepare_crossover_metadata = GeneticUtils.prepare_crossover_metadata





