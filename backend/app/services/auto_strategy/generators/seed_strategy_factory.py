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

from ..genes.conditions import Condition, ConditionGroup
from ..genes.indicator import IndicatorGene
from ..genes.strategy import StrategyGene
from ..genes.tpsl import TPSLGene
from ..config.constants import TPSLMethod

logger = logging.getLogger(__name__)


class SeedStrategyFactory:
    """
    シード戦略ファクトリ

    GA初期集団に注入するための実戦的な戦略テンプレートを生成します。
    """

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
        # 指標: ADX (DMP, DMN, ADX を返す)
        indicators = [
            IndicatorGene(
                type="ADX",
                parameters={"length": 14},
                enabled=True,
                id=str(uuid.uuid4()),
            ),
        ]

        # Long: DMP > 45 AND ADX > 45
        long_conditions = [
            ConditionGroup(
                operator="AND",
                conditions=[
                    Condition(
                        left_operand={"type": "indicator", "name": "DMP_14"},
                        operator=">",
                        right_operand=45.0,
                        direction="long",
                    ),
                    Condition(
                        left_operand={"type": "indicator", "name": "ADX_14"},
                        operator=">",
                        right_operand=45.0,
                        direction="long",
                    ),
                ],
            )
        ]

        # Short: DMN > 45 AND ADX > 45
        short_conditions = [
            ConditionGroup(
                operator="AND",
                conditions=[
                    Condition(
                        left_operand={"type": "indicator", "name": "DMN_14"},
                        operator=">",
                        right_operand=45.0,
                        direction="short",
                    ),
                    Condition(
                        left_operand={"type": "indicator", "name": "ADX_14"},
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
            metadata={"seed_strategy": "dmi_extreme_trend", "version": "1.0"},
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
        indicators = [
            IndicatorGene(
                type="RSI",
                parameters={"period": 14},
                enabled=True,
                id=str(uuid.uuid4()),
            ),
        ]

        # Long: RSI > 75 (強い上昇モメンタム)
        long_conditions = [
            Condition(
                left_operand={"type": "indicator", "name": "RSI_14"},
                operator=">",
                right_operand=75.0,
                direction="long",
            )
        ]

        # Short: RSI < 25 (強い下落モメンタム)
        short_conditions = [
            Condition(
                left_operand={"type": "indicator", "name": "RSI_14"},
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
            metadata={"seed_strategy": "rsi_momentum", "version": "1.0"},
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
        indicators = [
            IndicatorGene(
                type="BBANDS",
                parameters={"length": 20, "std": 2.0},
                enabled=True,
                id=str(uuid.uuid4()),
            ),
        ]

        # Long: Close > BBU (Upper Band)
        long_conditions = [
            Condition(
                left_operand="close",
                operator=">",
                right_operand={"type": "indicator", "name": "BBU_20_2.0"},
                direction="long",
            )
        ]

        # Short: Close < BBL (Lower Band)
        short_conditions = [
            Condition(
                left_operand="close",
                operator="<",
                right_operand={"type": "indicator", "name": "BBL_20_2.0"},
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
            metadata={"seed_strategy": "bollinger_breakout", "version": "1.0"},
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
        indicators = [
            IndicatorGene(
                type="KAMA",
                parameters={"length": 30},
                enabled=True,
                id=str(uuid.uuid4()),
            ),
            IndicatorGene(
                type="ADX",
                parameters={"length": 13},
                enabled=True,
                id=str(uuid.uuid4()),
            ),
        ]

        # Long: Close > KAMA AND DMP > 40 AND ADX > 20
        long_conditions = [
            ConditionGroup(
                operator="AND",
                conditions=[
                    Condition(
                        left_operand="close",
                        operator=">",
                        right_operand={"type": "indicator", "name": "KAMA_30"},
                        direction="long",
                    ),
                    Condition(
                        left_operand={"type": "indicator", "name": "DMP_13"},
                        operator=">",
                        right_operand=40.0,
                        direction="long",
                    ),
                    Condition(
                        left_operand={"type": "indicator", "name": "ADX_13"},
                        operator=">",
                        right_operand=20.0,
                        direction="long",
                    ),
                ],
            )
        ]

        # Short: Close < KAMA AND DMN > 40 AND ADX > 20
        short_conditions = [
            ConditionGroup(
                operator="AND",
                conditions=[
                    Condition(
                        left_operand="close",
                        operator="<",
                        right_operand={"type": "indicator", "name": "KAMA_30"},
                        direction="short",
                    ),
                    Condition(
                        left_operand={"type": "indicator", "name": "DMN_13"},
                        operator=">",
                        right_operand=40.0,
                        direction="short",
                    ),
                    Condition(
                        left_operand={"type": "indicator", "name": "ADX_13"},
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
            metadata={"seed_strategy": "kama_adx_hybrid", "version": "1.0"},
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
        indicators = [
            IndicatorGene(
                type="MACD",
                parameters={"fast": 12, "slow": 26, "signal": 9},
                enabled=True,
                id=str(uuid.uuid4()),
            ),
            IndicatorGene(
                type="BBANDS",
                parameters={"length": 20, "std": 2.0},
                enabled=True,
                id=str(uuid.uuid4()),
            ),
            IndicatorGene(
                type="ATR",
                parameters={"length": 100},
                enabled=True,
                id=str(uuid.uuid4()),
            ),
        ]

        # 簡易版WAE Long: MACD > Signal AND MACD > 0
        # (本来はBB Width > Dead Zone の条件も必要だが、条件式の制約上省略)
        long_conditions = [
            ConditionGroup(
                operator="AND",
                conditions=[
                    Condition(
                        left_operand={"type": "indicator", "name": "MACD_12_26_9"},
                        operator=">",
                        right_operand={"type": "indicator", "name": "MACDs_12_26_9"},
                        direction="long",
                    ),
                    Condition(
                        left_operand={"type": "indicator", "name": "MACD_12_26_9"},
                        operator=">",
                        right_operand=0.0,
                        direction="long",
                    ),
                ],
            )
        ]

        # Short: MACD < Signal AND MACD < 0
        short_conditions = [
            ConditionGroup(
                operator="AND",
                conditions=[
                    Condition(
                        left_operand={"type": "indicator", "name": "MACD_12_26_9"},
                        operator="<",
                        right_operand={"type": "indicator", "name": "MACDs_12_26_9"},
                        direction="short",
                    ),
                    Condition(
                        left_operand={"type": "indicator", "name": "MACD_12_26_9"},
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
            metadata={"seed_strategy": "wae", "version": "1.0"},
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
        indicators = [
            IndicatorGene(
                type="T3",
                parameters={"length": 200, "a": 0.7},  # 少し短めに調整
                enabled=True,
                id=str(uuid.uuid4()),
            ),
            IndicatorGene(
                type="ADX",
                parameters={"length": 14},
                enabled=True,
                id=str(uuid.uuid4()),
            ),
        ]

        # Long: Close > T3 AND ADX > 20
        long_conditions = [
            ConditionGroup(
                operator="AND",
                conditions=[
                    Condition(
                        left_operand="close",
                        operator=">",
                        right_operand={"type": "indicator", "name": "T3_200_0.7"},
                        direction="long",
                    ),
                    Condition(
                        left_operand={"type": "indicator", "name": "ADX_14"},
                        operator=">",
                        right_operand=20.0,
                        direction="long",
                    ),
                ],
            )
        ]

        # Short: Close < T3 AND ADX > 20
        short_conditions = [
            ConditionGroup(
                operator="AND",
                conditions=[
                    Condition(
                        left_operand="close",
                        operator="<",
                        right_operand={"type": "indicator", "name": "T3_200_0.7"},
                        direction="short",
                    ),
                    Condition(
                        left_operand={"type": "indicator", "name": "ADX_14"},
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
            metadata={"seed_strategy": "trendilo", "version": "1.0"},
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
