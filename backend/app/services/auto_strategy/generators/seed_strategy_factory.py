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
from ..genes.conditions import Condition, ConditionGroup, EntryDirection
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

    # シード戦略のマッピング（名前 -> 作成メソッド）
    _SEED_MAPPING: dict[str, str] = {
        "dmi_extreme": "create_dmi_extreme_trend",
        "rsi_momentum": "create_rsi_momentum",
        "bollinger_breakout": "create_bollinger_breakout",
        "kama_adx": "create_kama_adx_hybrid",
        "wae": "create_wae",
        "trendilo": "create_trendilo",
    }

    # 戦略ごとのパラメータ定数
    class DMIExtremeParams:
        DI_THRESHOLD = 45.0
        ADX_THRESHOLD = 45.0
        ATR_LENGTH = 14
        ATR_SL_MULTIPLIER = 2.5
        ATR_TP_MULTIPLIER = 5.0
        SL_PCT = 0.025
        TP_PCT = 0.05

    class RSIMomentumParams:
        RSI_LONG_THRESHOLD = 75.0
        RSI_SHORT_THRESHOLD = 25.0
        RSI_PERIOD = 14
        SL_PCT = 0.02
        TP_PCT = 0.04
        RISK_REWARD_RATIO = 2.0

    class BollingerBreakoutParams:
        BB_LENGTH = 20
        BB_STD = 2.0
        SL_PCT = 0.04
        TP_PCT = 0.08
        RISK_REWARD_RATIO = 2.0

    class KAMAADXHybridParams:
        KAMA_LENGTH = 30
        ADX_LENGTH = 13
        DI_THRESHOLD = 40.0
        ADX_THRESHOLD = 20.0
        SL_PCT = 0.03
        TP_PCT = 0.05
        RISK_REWARD_RATIO = 1.67

    class WAEParams:
        MACD_FAST = 12
        MACD_SLOW = 26
        MACD_SIGNAL = 9
        BB_LENGTH = 20
        BB_STD = 2.0
        ATR_LENGTH = 100
        ATR_PERIOD = 14
        ATR_SL_MULTIPLIER = 2.0
        ATR_TP_MULTIPLIER = 4.0
        SL_PCT = 0.03
        TP_PCT = 0.06

    class TrendiloParams:
        T3_LENGTH = 200
        T3_A = 0.7
        ADX_LENGTH = 14
        ADX_THRESHOLD = 20.0
        ATR_PERIOD = 14
        ATR_SL_MULTIPLIER = 2.0
        ATR_TP_MULTIPLIER = 4.0
        SL_PCT = 0.02
        TP_PCT = 0.04

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
    def _create_volatility_based_tpsl(
        cls,
        sl_pct: float,
        tp_pct: float,
        atr_sl_multiplier: float,
        atr_tp_multiplier: float,
        atr_period: int,
    ) -> TPSLGene:
        """ボラティリティベースのTPSL遺伝子を生成"""
        return TPSLGene(
            method=TPSLMethod.VOLATILITY_BASED,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            atr_multiplier_sl=atr_sl_multiplier,
            atr_multiplier_tp=atr_tp_multiplier,
            atr_period=atr_period,
            enabled=True,
        )

    @classmethod
    def _create_risk_reward_tpsl(
        cls,
        sl_pct: float,
        tp_pct: float,
        risk_reward_ratio: float,
    ) -> TPSLGene:
        """リスクリワード比率ベースのTPSL遺伝子を生成"""
        return TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            risk_reward_ratio=risk_reward_ratio,
            enabled=True,
        )

    @classmethod
    def _create_simple_condition(
        cls,
        left_operand: dict[str, str] | str,
        operator: str,
        right_operand: float | dict[str, str],
        direction: EntryDirection,
    ) -> Condition:
        """単一条件を生成"""
        return Condition(
            left_operand=left_operand,
            operator=operator,
            right_operand=right_operand,
            direction=direction,
        )

    @classmethod
    def _create_and_condition_group(
        cls,
        conditions: list[Condition | ConditionGroup],
    ) -> ConditionGroup:
        """AND条件グループを生成"""
        return ConditionGroup(operator="AND", conditions=conditions)

    @classmethod
    def get_all_seeds(cls) -> List[StrategyGene]:
        """
        すべてのシード戦略を取得

        Returns:
            シード戦略のリスト
        """
        seeds = [
            getattr(cls, method_name)()
            for method_name in cls._SEED_MAPPING.values()
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
        method_name = cls._SEED_MAPPING.get(name.lower())
        if method_name:
            return getattr(cls, method_name)()
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
        params = cls.DMIExtremeParams

        # 指標: ADX (ADX, DMP, DMN を返す)
        adx_indicator = cls._create_indicator_gene(
            "ADX", {"length": params.ATR_LENGTH}
        )
        indicators = [adx_indicator]

        # Long: DMP > 45 AND ADX > 45
        long_conditions: list[Condition | ConditionGroup] = [
            ConditionGroup(
                operator="AND",
                conditions=[
                    Condition(
                        left_operand=cls._indicator_ref(adx_indicator, 1),
                        operator=">",
                        right_operand=params.DI_THRESHOLD,
                        direction="long",
                    ),
                    Condition(
                        left_operand=cls._indicator_ref(adx_indicator, 0),
                        operator=">",
                        right_operand=params.ADX_THRESHOLD,
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
                        right_operand=params.DI_THRESHOLD,
                        direction="short",
                    ),
                    Condition(
                        left_operand=cls._indicator_ref(adx_indicator, 0),
                        operator=">",
                        right_operand=params.ADX_THRESHOLD,
                        direction="short",
                    ),
                ],
            )
        ]

        # TP/SL: ATRベース (2.5x SL, 5x TP)
        tpsl_gene = cls._create_volatility_based_tpsl(
            sl_pct=params.SL_PCT,
            tp_pct=params.TP_PCT,
            atr_sl_multiplier=params.ATR_SL_MULTIPLIER,
            atr_tp_multiplier=params.ATR_TP_MULTIPLIER,
            atr_period=params.ATR_LENGTH,
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
        params = cls.RSIMomentumParams

        rsi_indicator = cls._create_indicator_gene(
            "RSI", {"period": params.RSI_PERIOD}
        )
        indicators = [rsi_indicator]

        # Long: RSI > 75 (強い上昇モメンタム)
        long_conditions: list[Condition | ConditionGroup] = [
            Condition(
                left_operand=cls._indicator_ref(rsi_indicator),
                operator=">",
                right_operand=params.RSI_LONG_THRESHOLD,
                direction="long",
            )
        ]

        # Short: RSI < 25 (強い下落モメンタム)
        short_conditions: list[Condition | ConditionGroup] = [
            Condition(
                left_operand=cls._indicator_ref(rsi_indicator),
                operator="<",
                right_operand=params.RSI_SHORT_THRESHOLD,
                direction="short",
            )
        ]

        tpsl_gene = cls._create_risk_reward_tpsl(
            sl_pct=params.SL_PCT,
            tp_pct=params.TP_PCT,
            risk_reward_ratio=params.RISK_REWARD_RATIO,
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
        params = cls.BollingerBreakoutParams

        bbands_indicator = cls._create_indicator_gene(
            "BBANDS", {"length": params.BB_LENGTH, "std": params.BB_STD}
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

        tpsl_gene = cls._create_risk_reward_tpsl(
            sl_pct=params.SL_PCT,
            tp_pct=params.TP_PCT,
            risk_reward_ratio=params.RISK_REWARD_RATIO,
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
        params = cls.KAMAADXHybridParams

        kama_indicator = cls._create_indicator_gene(
            "KAMA", {"length": params.KAMA_LENGTH}
        )
        adx_indicator = cls._create_indicator_gene(
            "ADX", {"length": params.ADX_LENGTH}
        )
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
                        right_operand=params.DI_THRESHOLD,
                        direction="long",
                    ),
                    Condition(
                        left_operand=cls._indicator_ref(adx_indicator, 0),
                        operator=">",
                        right_operand=params.ADX_THRESHOLD,
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
                        right_operand=params.DI_THRESHOLD,
                        direction="short",
                    ),
                    Condition(
                        left_operand=cls._indicator_ref(adx_indicator, 0),
                        operator=">",
                        right_operand=params.ADX_THRESHOLD,
                        direction="short",
                    ),
                ],
            )
        ]

        tpsl_gene = cls._create_risk_reward_tpsl(
            sl_pct=params.SL_PCT,
            tp_pct=params.TP_PCT,
            risk_reward_ratio=params.RISK_REWARD_RATIO,
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
        params = cls.WAEParams

        macd_indicator = cls._create_indicator_gene(
            "MACD",
            {
                "fast": params.MACD_FAST,
                "slow": params.MACD_SLOW,
                "signal": params.MACD_SIGNAL,
            },
        )
        bbands_indicator = cls._create_indicator_gene(
            "BBANDS", {"length": params.BB_LENGTH, "std": params.BB_STD}
        )
        atr_indicator = cls._create_indicator_gene(
            "ATR", {"length": params.ATR_LENGTH}
        )
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

        tpsl_gene = cls._create_volatility_based_tpsl(
            sl_pct=params.SL_PCT,
            tp_pct=params.TP_PCT,
            atr_sl_multiplier=params.ATR_SL_MULTIPLIER,
            atr_tp_multiplier=params.ATR_TP_MULTIPLIER,
            atr_period=params.ATR_PERIOD,
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
        params = cls.TrendiloParams

        t3_indicator = cls._create_indicator_gene(
            "T3", {"length": params.T3_LENGTH, "a": params.T3_A}
        )
        adx_indicator = cls._create_indicator_gene(
            "ADX", {"length": params.ADX_LENGTH}
        )
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
                        right_operand=params.ADX_THRESHOLD,
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
                        right_operand=params.ADX_THRESHOLD,
                        direction="short",
                    ),
                ],
            )
        ]

        tpsl_gene = cls._create_volatility_based_tpsl(
            sl_pct=params.SL_PCT,
            tp_pct=params.TP_PCT,
            atr_sl_multiplier=params.ATR_SL_MULTIPLIER,
            atr_tp_multiplier=params.ATR_TP_MULTIPLIER,
            atr_period=params.ATR_PERIOD,
        )

        return StrategyGene.assemble(
            indicators=indicators,
            long_entry_conditions=long_conditions,
            short_entry_conditions=short_conditions,
            tpsl_gene=tpsl_gene,
            metadata=cls._seed_metadata("trendilo"),
        )


def inject_seeds_into_population(
    population: List[StrategyGene],
    seed_injection_rate: float = 0.3,
) -> List[StrategyGene]:
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
