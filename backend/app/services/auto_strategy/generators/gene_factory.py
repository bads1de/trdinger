"""
遺伝子生成ファクトリー

遺伝子生成ロジックをファクトリーパターンで統合します。
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List

from ..config.auto_strategy_config import GAConfig
from ..models.strategy_models import (
    Condition,
    IndicatorGene,
    StrategyGene,
    TPSLGene,
    create_random_tpsl_gene,
    PositionSizingGene,
    create_random_position_sizing_gene,
)
from .random_gene_generator import RandomGeneGenerator

logger = logging.getLogger(__name__)


class GeneratorType(Enum):
    """遺伝子生成器タイプ"""

    RANDOM = "random"
    SMART = "smart"
    DEFAULT = "default"


class BaseGeneGenerator(ABC):
    """遺伝子生成器の基底クラス"""

    def __init__(self, config: GAConfig):
        self.config = config

    @abstractmethod
    def generate_indicators(self) -> List[IndicatorGene]:
        """指標遺伝子を生成"""
        pass

    @abstractmethod
    def generate_conditions(
        self, indicators: List[IndicatorGene]
    ) -> tuple[List[Condition], List[Condition], List[Condition]]:
        """条件を生成（エントリー、ロングエントリー、ショートエントリー）"""
        pass

    def generate_tpsl_gene(self) -> TPSLGene:
        """TP/SL遺伝子を生成"""
        return create_random_tpsl_gene()

    def generate_position_sizing_gene(self) -> PositionSizingGene:
        """ポジションサイジング遺伝子を生成"""
        return create_random_position_sizing_gene(self.config)

    def generate_risk_management(self) -> Dict[str, float]:
        """リスク管理設定を生成"""
        return {
            "position_size": 0.1,  # デフォルト値（実際にはposition_sizing_geneが使用される）
        }

    def generate_complete_gene(self) -> StrategyGene:
        """完全な戦略遺伝子を生成"""
        try:
            # 指標生成
            indicators = self.generate_indicators()

            # 条件生成
            entry_conditions, long_entry_conditions, short_entry_conditions = (
                self.generate_conditions(indicators)
            )

            # その他の遺伝子生成
            tpsl_gene = self.generate_tpsl_gene()
            position_sizing_gene = self.generate_position_sizing_gene()
            risk_management = self.generate_risk_management()

            return StrategyGene(
                indicators=indicators,
                entry_conditions=entry_conditions,  # 後方互換性
                exit_conditions=[],  # 空のリスト（TP/SLで管理）
                long_entry_conditions=list(long_entry_conditions),  # 型を明示的に変換
                short_entry_conditions=list(short_entry_conditions),  # 型を明示的に変換
                risk_management=risk_management,
                tpsl_gene=tpsl_gene,
                position_sizing_gene=position_sizing_gene,
                metadata={"generated_by": self.__class__.__name__},
            )

        except Exception as e:
            logger.error(f"遺伝子生成エラー: {e}")
            return self._create_fallback_gene()

    def _create_fallback_gene(self) -> StrategyGene:
        """フォールバック遺伝子を作成"""
        return StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            entry_conditions=[],
            exit_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            risk_management={},  # デフォルト値
            tpsl_gene=create_random_tpsl_gene(),  # ランダム生成を使用
            position_sizing_gene=create_random_position_sizing_gene(
                self.config
            ),  # ランダム生成を使用
            metadata={"generated_by": "Fallback"},
        )


# RandomGeneGeneratorはrandom_gene_generator.pyで定義されたものを使用


class SmartGeneGenerator(BaseGeneGenerator):
    """スマート遺伝子生成器"""

    def __init__(self, config: GAConfig):
        super().__init__(config)
        self._smart_generator = None

    def _get_smart_generator(self):
        """SmartConditionGeneratorを遅延初期化"""
        if self._smart_generator is None:
            from ..generators.condition_generator import ConditionGenerator

            self._smart_generator = SmartConditionGenerator(
                enable_smart_generation=True
            )
        return self._smart_generator

    def generate_indicators(self) -> List[IndicatorGene]:
        """スマートに指標を生成"""
        try:
            # ランダム生成を使用（SmartConditionGeneratorは条件生成に特化）
            random_gen = RandomGeneGenerator(self.config, enable_smart_generation=False)
            return random_gen._generate_random_indicators()
        except Exception as e:
            logger.error(f"スマート指標生成エラー: {e}")
            # フォールバック: ランダム生成
            random_gen = RandomGeneGenerator(self.config, enable_smart_generation=False)
            return random_gen._generate_random_indicators()

    def generate_conditions(
        self, indicators: List[IndicatorGene]
    ) -> tuple[List[Condition], List[Condition], List[Condition]]:
        """スマートに条件を生成"""
        try:
            smart_gen = self._get_smart_generator()
            smart_gen.indicators = indicators
            long_conditions, short_conditions, exit_conditions = (
                smart_gen.generate_balanced_conditions(indicators)
            )

            # 後方互換性のためのエントリー条件
            entry_conditions = long_conditions[:1] if long_conditions else []

            # List[Union[Condition, ConditionGroup]] を List[Condition] に変換
            # ConditionGroupは無視し、Conditionのみを抽出
            from ..models.strategy_models import ConditionGroup

            def extract_conditions(conditions):
                result = []
                for cond in conditions:
                    if isinstance(cond, Condition):
                        result.append(cond)
                    elif isinstance(cond, ConditionGroup):
                        # ConditionGroup内のConditionをすべて抽出
                        result.extend(cond.conditions)
                return result

            entry_conditions = extract_conditions(entry_conditions)
            long_conditions = extract_conditions(long_conditions)
            short_conditions = extract_conditions(short_conditions)

            return entry_conditions, long_conditions, short_conditions

        except Exception as e:
            logger.error(f"スマート条件生成エラー: {e}")
            # フォールバック: ランダム生成
            # 注意: ランダムジェネレータの条件生成メソッドに合わせた呼び方
            fallback_entry = [
                Condition(left_operand="close", operator=">", right_operand="SMA")
            ]
            fallback_long = [
                Condition(left_operand="RSI", operator="<", right_operand=30),
                Condition(left_operand="close", operator=">", right_operand="SMA"),
            ]
            fallback_short = [
                Condition(left_operand="RSI", operator=">", right_operand=70)
            ]
            return fallback_entry, fallback_long, fallback_short


class DefaultGeneGenerator(BaseGeneGenerator):
    """デフォルト遺伝子生成器"""

    def generate_indicators(self) -> List[IndicatorGene]:
        """デフォルト指標を生成"""
        return [
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        ]

    def generate_conditions(
        self, indicators: List[IndicatorGene]
    ) -> tuple[List[Condition], List[Condition], List[Condition]]:
        """デフォルト条件を生成"""
        entry_conditions = [
            Condition(left_operand="RSI", operator="<", right_operand=30)
        ]
        long_conditions = [
            Condition(left_operand="RSI", operator="<", right_operand=30),
            Condition(left_operand="close", operator=">", right_operand="SMA"),
        ]
        short_conditions = [
            Condition(left_operand="RSI", operator=">", right_operand=70),
            Condition(left_operand="close", operator="<", right_operand="SMA"),
        ]

        return entry_conditions, long_conditions, short_conditions


class GeneGeneratorFactory:
    """遺伝子生成器ファクトリー"""

    @staticmethod
    def create_generator(generator_type: GeneratorType, config: GAConfig):
        """生成器タイプに応じた生成器を作成"""
        if generator_type == GeneratorType.RANDOM:
            # 実際のRandomGeneGeneratorを使用（型チェックなし）
            return RandomGeneGenerator(config)
        elif generator_type == GeneratorType.SMART:
            return SmartGeneGenerator(config)
        elif generator_type == GeneratorType.DEFAULT:
            return DefaultGeneGenerator(config)
        else:
            logger.warning(f"未知の生成器タイプ: {generator_type}, デフォルトを使用")
            return DefaultGeneGenerator(config)
