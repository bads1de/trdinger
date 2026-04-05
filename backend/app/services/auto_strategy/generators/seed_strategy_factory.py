"""
シード戦略ファクトリ

GA初期集団に注入する「シード戦略」を生成するファクトリクラス。
実戦的な戦略をハードコードし、GAの探索効率を向上させます。

設計思想:
- シード戦略は「実績のある」または「理論的に有効」な戦略のテンプレート
- GAはこれらを「親」として交配・変異させ、さらに最適化する
- 初期集団の10-20%をシード戦略で置き換えることで、探索効率を向上
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, List, Optional

from ..config.constants import TPSLMethod
from ..genes.conditions import Condition, ConditionGroup
from ..genes.indicator import IndicatorGene
from ..genes.strategy import StrategyGene
from ..genes.tpsl import TPSLGene
from ..utils.indicator_references import build_indicator_reference_name

logger = logging.getLogger(__name__)


class SeedStrategyFactory:
    """
    シード戦略ファクトリ

    GA初期集団に注入するための実戦的な戦略テンプレートを生成します。
    """

    @staticmethod
    def _create_indicator_gene(
        indicator_type: str, parameters: dict[str, Any]
    ) -> IndicatorGene:
        """シード用の指標遺伝子を生成"""
        return IndicatorGene(
            type=indicator_type,
            parameters=parameters,
            enabled=True,
            id=str(uuid.uuid4()),
        )

    @staticmethod
    def _indicator_ref_name(
        indicator: IndicatorGene, output_index: Optional[int] = None
    ) -> str:
        """実行時の登録規約に合わせた指標参照名を生成"""
        return build_indicator_reference_name(indicator, output_index)

    @classmethod
    def _indicator_ref(
        cls, indicator: IndicatorGene, output_index: Optional[int] = None
    ) -> dict[str, str]:
        """Condition で使う指標参照辞書を生成"""
        return {
            "type": "indicator",
            "name": cls._indicator_ref_name(indicator, output_index),
        }

    @staticmethod
    def _seed_metadata(seed_name: str) -> dict[str, str]:
        """シード戦略の共通メタデータを生成"""
        return {"seed_strategy": seed_name, "version": "1.0"}

    @classmethod
    def get_all_seeds(cls) -> List[StrategyGene]:
        """
        すべてのシード戦略を取得

        Returns:
            シード戦略のリスト
        """
        seeds = [
            cls.create_dmi_extreme_trend(),
            cls.create_rsi_momentum(),
            cls.create_bollinger_breakout(),
            cls.create_kama_adx_hybrid(),
            cls.create_wae(),
            cls.create_trendilo(),
        ]
        logger.info(f"シード戦略を {len(seeds)} 個生成しました")
        return seeds

    @classmethod
    def get_seed_by_name(cls, name: str) -> Optional[StrategyGene]:
        """
        名前でシード戦略を取得

        Args:
            name: 戦略名 (例: "dmi_extreme", "rsi_momentum")

        Returns:
            対応するシード戦略、見つからない場合はNone
        """
        mapping = {
            "dmi_extreme": cls.create_dmi_extreme_trend,
            "rsi_momentum": cls.create_rsi_momentum,
            "bollinger_breakout": cls.create_bollinger_breakout,
            "kama_adx": cls.create_kama_adx_hybrid,
            "wae": cls.create_wae,
            "trendilo": cls.create_trendilo,
        }
        factory_func = mapping.get(name.lower())
        if factory_func:
            return factory_func()
        return None

    # =========================================================================
    # Strategy A: DMI Extreme Trend
    # =========================================================================
    @classmethod
    def create_dmi_extreme_trend(cls) -> StrategyGene:
        """
        DMI Extreme Trend 戦略

        超強力なトレンド発生時のみエントリーする戦略。
        DI > 45 AND ADX > 45 という厳しいフィルターを使用。

        Returns:
            StrategyGene
        """
        # 指標: ADX (ADX, DMP, DMN を返す)
        adx_indicator = cls._create_indicator_gene("ADX", {"length": 14})
        indicators = [adx_indicator]

        # Long: DMP > 45 AND ADX > 45
        long_conditions: list[Condition | ConditionGroup] = [
            ConditionGroup(
                operator="AND",
                conditions=[
                    Condition(
                        left_operand=cls._indicator_ref(adx_indicator, 1),
                        operator=">",
                        right_operand=45.0,
                        direction="long",
                    ),
                    Condition(
                        left_operand=cls._indicator_ref(adx_indicator, 0),
                        operator=">",
                        right_operand=45.0,
                        direction="long",
                    ),
                ],
            )
        ]

        # Short: DMN > 45 AND ADX > 45
        short_conditions: list[Condition | ConditionGroup] = [
            ConditionGroup(
                operator="AND",
                conditions=[
                    Condition(
                        left_operand=cls._indicator_ref(adx_indicator, 2),
                        operator=">",
                        right_operand=45.0,
                        direction="short",
                    ),
                    Condition(
                        left_operand=cls._indicator_ref(adx_indicator, 0),
                        operator=">",
                        right_operand=45.0,
                        direction="short",
                    ),
                ],
            )
        ]

        # TP/SL: ATRベース (2.5x SL, 5x TP)
        tpsl_gene = TPSLGene(
            method=TPSLMethod.VOLATILITY_BASED,
            stop_loss_pct=0.025,
            take_profit_pct=0.05,
            atr_multiplier_sl=2.5,
            atr_multiplier_tp=5.0,
            atr_period=14,
            enabled=True,
        )

        return StrategyGene.assemble(
            indicators=indicators,
            long_entry_conditions=long_conditions,
            short_entry_conditions=short_conditions,
            tpsl_gene=tpsl_gene,
            metadata=cls._seed_metadata("dmi_extreme_trend"),
        )

    # =========================================================================
    # Strategy B: RSI Momentum
    # =========================================================================
    @classmethod
    def create_rsi_momentum(cls) -> StrategyGene:
        """
        RSI Momentum 戦略

        RSIをトレンドフォロー指標として使用（逆張りではない）。
        RSI > 75 で強い上昇トレンド、RSI < 25 で強い下落トレンドと判断。

        Returns:
            StrategyGene
        """
        rsi_indicator = cls._create_indicator_gene("RSI", {"period": 14})
        indicators = [rsi_indicator]

        # Long: RSI > 75 (強い上昇モメンタム)
        long_conditions: list[Condition | ConditionGroup] = [
            Condition(
                left_operand=cls._indicator_ref(rsi_indicator),
                operator=">",
                right_operand=75.0,
                direction="long",
            )
        ]

        # Short: RSI < 25 (強い下落モメンタム)
        short_conditions: list[Condition | ConditionGroup] = [
            Condition(
                left_operand=cls._indicator_ref(rsi_indicator),
                operator="<",
                right_operand=25.0,
                direction="short",
            )
        ]

        tpsl_gene = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            risk_reward_ratio=2.0,
            enabled=True,
        )

        return StrategyGene.assemble(
            indicators=indicators,
            long_entry_conditions=long_conditions,
            short_entry_conditions=short_conditions,
            tpsl_gene=tpsl_gene,
            metadata=cls._seed_metadata("rsi_momentum"),
        )

    # =========================================================================
    # Strategy C: Bollinger Breakout
    # =========================================================================
    @classmethod
    def create_bollinger_breakout(cls) -> StrategyGene:
        """
        Bollinger Breakout 戦略

        ボリンジャーバンドのブレイクアウトを狙う。
        Close > Upper で Long, Close < Lower で Short。

        Returns:
            StrategyGene
        """
        bbands_indicator = cls._create_indicator_gene(
            "BBANDS", {"length": 20, "std": 2.0}
        )
        indicators = [bbands_indicator]

        # Long: Close > BBU (Upper Band)
        long_conditions: list[Condition | ConditionGroup] = [
            Condition(
                left_operand="close",
                operator=">",
                right_operand=cls._indicator_ref(bbands_indicator, 0),
                direction="long",
            )
        ]

        # Short: Close < BBL (Lower Band)
        short_conditions: list[Condition | ConditionGroup] = [
            Condition(
                left_operand="close",
                operator="<",
                right_operand=cls._indicator_ref(bbands_indicator, 2),
                direction="short",
            )
        ]

        tpsl_gene = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=0.04,
            take_profit_pct=0.08,
            risk_reward_ratio=2.0,
            enabled=True,
        )

        return StrategyGene.assemble(
            indicators=indicators,
            long_entry_conditions=long_conditions,
            short_entry_conditions=short_conditions,
            tpsl_gene=tpsl_gene,
            metadata=cls._seed_metadata("bollinger_breakout"),
        )

    # =========================================================================
    # Strategy D: KAMA-ADX Hybrid
    # =========================================================================
    @classmethod
    def create_kama_adx_hybrid(cls) -> StrategyGene:
        """
        KAMA-ADX Hybrid 戦略

        Kaufman Adaptive Moving Average (KAMA) によるトレンド判定と、
        ADXによる強いトレンドフィルターを組み合わせた複合戦略。

        Returns:
            StrategyGene
        """
        kama_indicator = cls._create_indicator_gene("KAMA", {"length": 30})
        adx_indicator = cls._create_indicator_gene("ADX", {"length": 13})
        indicators = [kama_indicator, adx_indicator]

        # Long: Close > KAMA AND DMP > 40 AND ADX > 20
        long_conditions: list[Condition | ConditionGroup] = [
            ConditionGroup(
                operator="AND",
                conditions=[
                    Condition(
                        left_operand="close",
                        operator=">",
                        right_operand=cls._indicator_ref(kama_indicator),
                        direction="long",
                    ),
                    Condition(
                        left_operand=cls._indicator_ref(adx_indicator, 1),
                        operator=">",
                        right_operand=40.0,
                        direction="long",
                    ),
                    Condition(
                        left_operand=cls._indicator_ref(adx_indicator, 0),
                        operator=">",
                        right_operand=20.0,
                        direction="long",
                    ),
                ],
            )
        ]

        # Short: Close < KAMA AND DMN > 40 AND ADX > 20
        short_conditions: list[Condition | ConditionGroup] = [
            ConditionGroup(
                operator="AND",
                conditions=[
                    Condition(
                        left_operand="close",
                        operator="<",
                        right_operand=cls._indicator_ref(kama_indicator),
                        direction="short",
                    ),
                    Condition(
                        left_operand=cls._indicator_ref(adx_indicator, 2),
                        operator=">",
                        right_operand=40.0,
                        direction="short",
                    ),
                    Condition(
                        left_operand=cls._indicator_ref(adx_indicator, 0),
                        operator=">",
                        right_operand=20.0,
                        direction="short",
                    ),
                ],
            )
        ]

        tpsl_gene = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=0.03,
            take_profit_pct=0.05,
            risk_reward_ratio=1.67,
            enabled=True,
        )

        return StrategyGene.assemble(
            indicators=indicators,
            long_entry_conditions=long_conditions,
            short_entry_conditions=short_conditions,
            tpsl_gene=tpsl_gene,
            metadata=cls._seed_metadata("kama_adx_hybrid"),
        )

    # =========================================================================
    # Strategy E: WAE (Waddah Attar Explosion)
    # =========================================================================
    @classmethod
    def create_wae(cls) -> StrategyGene:
        """
        WAE (Waddah Attar Explosion) 戦略

        MACD Delta、BB Width、ATRベースのDead Zoneを組み合わせた
        ボラティリティ爆発戦略。

        注意: WAE は複合指標のため、MACD, BBANDS, ATR を組み合わせる。
        ただし、BB Width や MACD Delta はカスタム計算が必要なため、
        ここでは簡易版として「BB Width > ATR * Multiplier」を使用。

        Returns:
            StrategyGene
        """
        macd_indicator = cls._create_indicator_gene(
            "MACD", {"fast": 12, "slow": 26, "signal": 9}
        )
        bbands_indicator = cls._create_indicator_gene(
            "BBANDS", {"length": 20, "std": 2.0}
        )
        atr_indicator = cls._create_indicator_gene("ATR", {"length": 100})
        indicators = [macd_indicator, bbands_indicator, atr_indicator]

        # 簡易版WAE Long: MACD > Signal AND MACD > 0
        # (本来はBB Width > Dead Zone の条件も必要だが、条件式の制約上省略)
        long_conditions: list[Condition | ConditionGroup] = [
            ConditionGroup(
                operator="AND",
                conditions=[
                    Condition(
                        left_operand=cls._indicator_ref(macd_indicator, 0),
                        operator=">",
                        right_operand=cls._indicator_ref(macd_indicator, 1),
                        direction="long",
                    ),
                    Condition(
                        left_operand=cls._indicator_ref(macd_indicator, 0),
                        operator=">",
                        right_operand=0.0,
                        direction="long",
                    ),
                ],
            )
        ]

        # Short: MACD < Signal AND MACD < 0
        short_conditions: list[Condition | ConditionGroup] = [
            ConditionGroup(
                operator="AND",
                conditions=[
                    Condition(
                        left_operand=cls._indicator_ref(macd_indicator, 0),
                        operator="<",
                        right_operand=cls._indicator_ref(macd_indicator, 1),
                        direction="short",
                    ),
                    Condition(
                        left_operand=cls._indicator_ref(macd_indicator, 0),
                        operator="<",
                        right_operand=0.0,
                        direction="short",
                    ),
                ],
            )
        ]

        tpsl_gene = TPSLGene(
            method=TPSLMethod.VOLATILITY_BASED,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            atr_multiplier_sl=2.0,
            atr_multiplier_tp=4.0,
            atr_period=14,
            enabled=True,
        )

        return StrategyGene.assemble(
            indicators=indicators,
            long_entry_conditions=long_conditions,
            short_entry_conditions=short_conditions,
            tpsl_gene=tpsl_gene,
            metadata=cls._seed_metadata("wae"),
        )

    # =========================================================================
    # Strategy F: Trendilo (ALMA Momentum)
    # =========================================================================
    @classmethod
    def create_trendilo(cls) -> StrategyGene:
        """
        Trendilo 戦略

        T3 Moving Average による長期トレンド判定と、
        ADX によるフィルター、ALMA モメンタムをトリガーに使用。

        注意: ALMA(Change(Close)) は直接計算できないため、
        ここでは T3 + ADX の組み合わせを使用。

        Returns:
            StrategyGene
        """
        t3_indicator = cls._create_indicator_gene("T3", {"length": 200, "a": 0.7})
        adx_indicator = cls._create_indicator_gene("ADX", {"length": 14})
        indicators = [t3_indicator, adx_indicator]

        # Long: Close > T3 AND ADX > 20
        long_conditions: list[Condition | ConditionGroup] = [
            ConditionGroup(
                operator="AND",
                conditions=[
                    Condition(
                        left_operand="close",
                        operator=">",
                        right_operand=cls._indicator_ref(t3_indicator),
                        direction="long",
                    ),
                    Condition(
                        left_operand=cls._indicator_ref(adx_indicator, 0),
                        operator=">",
                        right_operand=20.0,
                        direction="long",
                    ),
                ],
            )
        ]

        # Short: Close < T3 AND ADX > 20
        short_conditions: list[Condition | ConditionGroup] = [
            ConditionGroup(
                operator="AND",
                conditions=[
                    Condition(
                        left_operand="close",
                        operator="<",
                        right_operand=cls._indicator_ref(t3_indicator),
                        direction="short",
                    ),
                    Condition(
                        left_operand=cls._indicator_ref(adx_indicator, 0),
                        operator=">",
                        right_operand=20.0,
                        direction="short",
                    ),
                ],
            )
        ]

        tpsl_gene = TPSLGene(
            method=TPSLMethod.VOLATILITY_BASED,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            atr_multiplier_sl=2.0,
            atr_multiplier_tp=4.0,
            atr_period=14,
            enabled=True,
        )

        return StrategyGene.assemble(
            indicators=indicators,
            long_entry_conditions=long_conditions,
            short_entry_conditions=short_conditions,
            tpsl_gene=tpsl_gene,
            metadata=cls._seed_metadata("trendilo"),
        )


def inject_seeds_into_population(
    population: List[Any],
    seed_injection_rate: float = 0.1,
) -> List[Any]:
    """
    集団にシード戦略を注入

    Args:
        population: 初期集団（ランダム生成済み）
        seed_injection_rate: シード注入率 (0.0 - 1.0)

    Returns:
        シード戦略が注入された集団
    """
    if seed_injection_rate <= 0:
        return population

    seeds = SeedStrategyFactory.get_all_seeds()
    num_to_inject = min(
        int(len(population) * seed_injection_rate),
        len(seeds),
    )

    if num_to_inject == 0:
        return population

    # 集団の先頭をシード戦略で置き換え
    for i in range(num_to_inject):
        seed = seeds[i % len(seeds)]
        # DEAP Individual クラスへの変換が必要な場合はここで処理
        # 現時点では StrategyGene をそのまま使用
        population[i] = seed
        logger.debug(
            f"シード戦略 {i + 1}/{num_to_inject} を注入: {seed.metadata.get('seed_strategy')}"
        )

    logger.info(
        f"シード戦略を {num_to_inject} 個注入しました (注入率: {seed_injection_rate * 100:.1f}%)"
    )
    return population
