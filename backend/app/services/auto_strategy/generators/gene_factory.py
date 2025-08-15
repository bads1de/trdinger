"""
遺伝子生成ファクトリー

遺伝子生成ロジックをファクトリーパターンで統合します。
"""

import logging
import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List

from ..models.ga_config import GAConfig
from ..models.gene_strategy import Condition, IndicatorGene, StrategyGene
from ..models.gene_tpsl import TPSLGene, create_random_tpsl_gene
from ..models.gene_position_sizing import (
    PositionSizingGene,
    create_random_position_sizing_gene,
)

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
                long_entry_conditions=long_entry_conditions,
                short_entry_conditions=short_entry_conditions,
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
        from ..utils.auto_strategy_utils import AutoStrategyUtils

        return AutoStrategyUtils.create_default_strategy_gene(
            strategy_gene_class=StrategyGene
        )


class RandomGeneGenerator(BaseGeneGenerator):
    """ランダム遺伝子生成器"""

    def generate_indicators(self) -> List[IndicatorGene]:
        """ランダムに指標を生成"""
        try:
            from app.services.indicators.config import indicator_registry

            indicators = []
            allowed_indicators = self.config.allowed_indicators or [
                "SMA",
                "EMA",
                "RSI",
                "MACD",
                "BB",
            ]
            num_indicators = min(
                random.randint(2, self.config.max_indicators), len(allowed_indicators)
            )

            selected_types = random.sample(allowed_indicators, num_indicators)

            for indicator_type in selected_types:
                parameters = indicator_registry.generate_parameters_for_indicator(
                    indicator_type
                )
                indicators.append(
                    IndicatorGene(
                        type=indicator_type, parameters=parameters, enabled=True
                    )
                )

            return indicators

        except Exception as e:
            logger.error(f"ランダム指標生成エラー: {e}")
            return [
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            ]

    def generate_conditions(
        self, indicators: List[IndicatorGene]
    ) -> tuple[List[Condition], List[Condition], List[Condition]]:
        """ランダムに条件を生成"""
        try:

            # 基本的なランダム条件生成
            entry_conditions = []
            long_conditions = []
            short_conditions = []

            # エントリー条件（後方互換性）
            if indicators:
                indicator = random.choice(indicators)
                entry_conditions.append(
                    Condition(
                        left_operand=indicator.type,
                        operator=random.choice([">", "<", ">=", "<="]),
                        right_operand=random.uniform(20, 80),
                    )
                )

            # ロング条件
            for _ in range(random.randint(1, 3)):
                if indicators:
                    indicator = random.choice(indicators)
                    long_conditions.append(
                        Condition(
                            left_operand=indicator.type,
                            operator=random.choice([">", ">="]),
                            right_operand=random.uniform(30, 70),
                        )
                    )

            # ショート条件
            for _ in range(random.randint(1, 3)):
                if indicators:
                    indicator = random.choice(indicators)
                    short_conditions.append(
                        Condition(
                            left_operand=indicator.type,
                            operator=random.choice(["<", "<="]),
                            right_operand=random.uniform(30, 70),
                        )
                    )

            return entry_conditions, long_conditions, short_conditions

        except Exception as e:
            logger.error(f"ランダム条件生成エラー: {e}")
            return [], [], []


class SmartGeneGenerator(BaseGeneGenerator):
    """スマート遺伝子生成器"""

    def __init__(self, config: GAConfig):
        super().__init__(config)
        self._smart_generator = None

    def _get_smart_generator(self):
        """SmartConditionGeneratorを遅延初期化"""
        if self._smart_generator is None:
            from ..generators.smart_condition_generator import SmartConditionGenerator

            self._smart_generator = SmartConditionGenerator(
                enable_smart_generation=True
            )
        return self._smart_generator

    def generate_indicators(self) -> List[IndicatorGene]:
        """スマートに指標を生成"""
        try:
            smart_gen = self._get_smart_generator()
            return smart_gen._generate_indicators(self.config)
        except Exception as e:
            logger.error(f"スマート指標生成エラー: {e}")
            # フォールバック: ランダム生成
            random_gen = RandomGeneGenerator(self.config)
            return random_gen.generate_indicators()

    def generate_conditions(
        self, indicators: List[IndicatorGene]
    ) -> tuple[List[Condition], List[Condition], List[Condition]]:
        """スマートに条件を生成"""
        try:
            smart_gen = self._get_smart_generator()
            smart_gen.indicators = indicators
            long_conditions, short_conditions, exit_conditions = (
                smart_gen.generate_conditions(indicators)
            )

            # 後方互換性のためのエントリー条件
            entry_conditions = long_conditions[:1] if long_conditions else []

            return entry_conditions, long_conditions, short_conditions

        except Exception as e:
            logger.error(f"スマート条件生成エラー: {e}")
            # フォールバック: ランダム生成
            random_gen = RandomGeneGenerator(self.config)
            return random_gen.generate_conditions(indicators)


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
    def create_generator(
        generator_type: GeneratorType, config: GAConfig
    ) -> BaseGeneGenerator:
        """生成器タイプに応じた生成器を作成"""
        if generator_type == GeneratorType.RANDOM:
            return RandomGeneGenerator(config)
        elif generator_type == GeneratorType.SMART:
            return SmartGeneGenerator(config)
        elif generator_type == GeneratorType.DEFAULT:
            return DefaultGeneGenerator(config)
        else:
            logger.warning(f"未知の生成器タイプ: {generator_type}, デフォルトを使用")
            return DefaultGeneGenerator(config)
