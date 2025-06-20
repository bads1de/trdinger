"""
ランダム遺伝子生成器

OI/FRデータを含む多様な戦略遺伝子をランダムに生成します。
"""

import random
from typing import List, Dict, Any
import logging

from ..models.strategy_gene import StrategyGene, IndicatorGene, Condition
from ...indicators.constants import ALL_INDICATORS
from ..utils.parameter_generators import generate_indicator_parameters

logger = logging.getLogger(__name__)


class RandomGeneGenerator:
    """
    ランダム戦略遺伝子生成器

    OI/FRデータソースを含む多様な戦略遺伝子を生成します。
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        """
        初期化

        Args:
            config: 生成設定
        """
        self.config = config or {}

        # デフォルト設定
        self.max_indicators = self.config.get("max_indicators", 5)
        self.min_indicators = self.config.get("min_indicators", 1)
        self.max_conditions = self.config.get("max_conditions", 3)
        self.min_conditions = self.config.get("min_conditions", 1)

        # 利用可能な指標タイプ（共通定数から取得）
        self.available_indicators = ALL_INDICATORS.copy()

        # 利用可能なデータソース
        self.available_data_sources = [
            "close",
            "open",
            "high",
            "low",
            "volume",
            "OpenInterest",
            "FundingRate",
        ]

        # 利用可能な演算子
        self.available_operators = [">", "<", ">=", "<=", "cross_above", "cross_below"]

    def generate_random_gene(self) -> StrategyGene:
        """
        ランダムな戦略遺伝子を生成

        Returns:
            生成された戦略遺伝子
        """
        try:
            # 指標を生成
            indicators = self._generate_random_indicators()

            # 条件を生成
            entry_conditions = self._generate_random_conditions(indicators, "entry")
            exit_conditions = self._generate_random_conditions(indicators, "exit")

            # リスク管理設定
            risk_management = self._generate_risk_management()

            gene = StrategyGene(
                indicators=indicators,
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
                risk_management=risk_management,
                metadata={"generated_by": "RandomGeneGenerator"},
            )

            logger.debug(f"Generated random gene with {len(indicators)} indicators")
            return gene

        except Exception as e:
            logger.error(f"Failed to generate random gene: {e}")
            # フォールバック: 最小限の遺伝子を生成
            return self._generate_fallback_gene()

    def _generate_random_indicators(self) -> List[IndicatorGene]:
        """ランダムな指標リストを生成"""
        num_indicators = random.randint(self.min_indicators, self.max_indicators)
        indicators = []

        for _ in range(num_indicators):
            indicator_type = random.choice(self.available_indicators)
            parameters = generate_indicator_parameters(indicator_type)

            indicators.append(
                IndicatorGene(type=indicator_type, parameters=parameters, enabled=True)
            )

        return indicators

    def _generate_random_conditions(
        self, indicators: List[IndicatorGene], condition_type: str
    ) -> List[Condition]:
        """ランダムな条件リストを生成"""
        num_conditions = random.randint(
            self.min_conditions, min(self.max_conditions, 2)
        )
        conditions = []

        for _ in range(num_conditions):
            condition = self._generate_single_condition(indicators, condition_type)
            if condition:
                conditions.append(condition)

        # 最低1つの条件は保証
        if not conditions:
            conditions.append(self._generate_fallback_condition(condition_type))

        return conditions

    def _generate_single_condition(
        self, indicators: List[IndicatorGene], condition_type: str
    ) -> Condition:
        """単一の条件を生成"""
        # 左オペランドの選択
        left_operand = self._choose_operand(indicators)

        # 演算子の選択
        operator = random.choice(self.available_operators)

        # 右オペランドの選択
        right_operand = self._choose_right_operand(
            left_operand, indicators, condition_type
        )

        return Condition(
            left_operand=left_operand, operator=operator, right_operand=right_operand
        )

    def _choose_operand(self, indicators: List[IndicatorGene]) -> str:
        """オペランドを選択（指標名またはデータソース）"""
        choices = []

        # テクニカル指標名を追加
        for indicator in indicators:
            # 通常のテクニカル指標の場合
            if indicator.type in ["MACD"]:
                # MACDは特別な命名
                choices.append(f"{indicator.type}")
            elif "period" in indicator.parameters:
                period = indicator.parameters.get("period", 20)
                choices.append(f"{indicator.type}_{int(period)}")
            else:
                choices.append(indicator.type)

        # 基本データソースを追加（価格データ）
        basic_sources = ["close", "open", "high", "low", "volume"]
        choices.extend(basic_sources * 2)  # 基本データソースの重みを増やす

        # OI/FRデータソースを判断材料として追加
        choices.extend(["OpenInterest", "FundingRate"])

        return random.choice(choices) if choices else "close"

    def _choose_right_operand(
        self, left_operand: str, indicators: List[IndicatorGene], condition_type: str
    ):
        """右オペランドを選択（指標名、データソース、または数値）"""
        # 30%の確率で数値を使用
        if random.random() < 0.3:
            return self._generate_threshold_value(left_operand, condition_type)

        # 70%の確率で別の指標またはデータソースを使用
        return self._choose_operand(indicators)

    def _generate_threshold_value(self, operand: str, condition_type: str) -> float:
        """オペランドに応じた実用的な閾値を生成"""
        if "RSI" in operand:
            # RSI: 買われすぎ・売られすぎの判定
            if condition_type == "entry":
                return random.uniform(25, 35)  # 売られすぎからの反発狙い
            else:
                return random.uniform(65, 75)  # 買われすぎでの利確

        elif operand == "FundingRate":
            # Funding Rate: 市場センチメントの判定
            # 正の値: ロングポジション過熱、負の値: ショートポジション過熱
            funding_thresholds = [
                0.0001,  # 0.01% - 軽微な偏り
                0.0005,  # 0.05% - 中程度の偏り
                0.001,  # 0.1% - 強い偏り
                -0.0001,  # -0.01% - 軽微な逆偏り
                -0.0005,  # -0.05% - 中程度の逆偏り
                -0.001,  # -0.1% - 強い逆偏り
            ]
            return random.choice(funding_thresholds)

        elif operand == "OpenInterest":
            # Open Interest: 絶対値での判定（市場規模に依存）
            # 実際の値は市場によって大きく異なるため、相対的な変化率で判定することが多い
            # ここでは仮の絶対値を設定（実際の運用では過去データの統計値を使用）
            oi_thresholds = [
                1000000,  # 100万 - 小規模
                5000000,  # 500万 - 中規模
                10000000,  # 1000万 - 大規模
                50000000,  # 5000万 - 非常に大規模
            ]
            return random.choice(oi_thresholds)

        elif "STOCH" in operand:
            # Stochastic: 買われすぎ・売られすぎ
            if condition_type == "entry":
                return random.uniform(15, 25)  # 売られすぎ
            else:
                return random.uniform(75, 85)  # 買われすぎ

        elif "CCI" in operand:
            # CCI: 買われすぎ・売られすぎ
            if condition_type == "entry":
                return random.uniform(-150, -100)  # 売られすぎ
            else:
                return random.uniform(100, 150)  # 買われすぎ

        # 新規追加指標の閾値生成
        elif "STOCHRSI" in operand:
            # Stochastic RSI: 0-100の範囲
            if condition_type == "entry":
                return random.uniform(15, 25)  # 売られすぎ
            else:
                return random.uniform(75, 85)  # 買われすぎ

        elif "CMO" in operand:
            # CMO: -100から100の範囲
            if condition_type == "entry":
                return random.uniform(-60, -40)  # 売られすぎ
            else:
                return random.uniform(40, 60)  # 買われすぎ

        elif "ULTOSC" in operand:
            # Ultimate Oscillator: 0-100の範囲
            if condition_type == "entry":
                return random.uniform(20, 30)  # 売られすぎ
            else:
                return random.uniform(70, 80)  # 買われすぎ

        elif "TRIX" in operand:
            # TRIX: 通常は小さな値（-0.01から0.01程度）
            if condition_type == "entry":
                return random.uniform(-0.005, 0)  # 下降トレンド
            else:
                return random.uniform(0, 0.005)  # 上昇トレンド

        elif "BOP" in operand:
            # BOP: -1から1の範囲
            if condition_type == "entry":
                return random.uniform(-0.5, 0)  # 売り圧力優勢
            else:
                return random.uniform(0, 0.5)  # 買い圧力優勢

        elif "APO" in operand or "PPO" in operand:
            # APO/PPO: ゼロライン周辺
            if condition_type == "entry":
                return random.uniform(-2, 0)  # 下降モメンタム
            else:
                return random.uniform(0, 2)  # 上昇モメンタム

        elif "AROONOSC" in operand:
            # AROONOSC: -100から100の範囲
            if condition_type == "entry":
                return random.uniform(-50, 0)  # 下降トレンド優勢
            else:
                return random.uniform(0, 50)  # 上昇トレンド優勢

        elif "DX" in operand:
            # DX: 0から100の範囲、25以上で強いトレンド
            if condition_type == "entry":
                return random.uniform(25, 40)  # 強いトレンド開始
            else:
                return random.uniform(15, 25)  # トレンド弱化

        elif "ADXR" in operand:
            # ADXR: 0から100の範囲、ADXの平滑化版
            if condition_type == "entry":
                return random.uniform(25, 35)  # 強いトレンド
            else:
                return random.uniform(15, 25)  # トレンド弱化

        elif any(
            price_type in operand
            for price_type in ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"]
        ):
            # Price Transform indicators: 価格レベルでの比較
            # 現在価格に対する相対的な閾値
            if condition_type == "entry":
                return random.uniform(0.98, 1.0)  # 価格下落時のエントリー
            else:
                return random.uniform(1.0, 1.02)  # 価格上昇時のエグジット

        elif any(
            indicator_type in operand for indicator_type in ["PLUS_DI", "MINUS_DI"]
        ):
            # PLUS_DI, MINUS_DI: 0-100の範囲
            if condition_type == "entry":
                return random.uniform(20, 40)  # エントリー閾値
            else:
                return random.uniform(10, 30)  # エグジット閾値

        elif "HMA" in operand or "ZLEMA" in operand:
            # HMA, ZLEMA: 価格レベルでの比較
            if condition_type == "entry":
                return random.uniform(0.98, 1.0)  # 価格下落時のエントリー
            else:
                return random.uniform(1.0, 1.02)  # 価格上昇時のエグジット

        elif "MOM" in operand and "MOMENTUM" not in operand:
            # MOM (Momentum): 価格差での比較
            if condition_type == "entry":
                return random.uniform(-10, 0)  # 下降モメンタム
            else:
                return random.uniform(0, 10)  # 上昇モメンタム

        elif "DONCHIAN" in operand:
            # DONCHIAN: 価格レベルでの比較
            if condition_type == "entry":
                return random.uniform(0.99, 1.0)  # 下限ブレイクアウト
            else:
                return random.uniform(1.0, 1.01)  # 上限ブレイクアウト

        elif "PVT" in operand:
            # PVT: 累積値での比較（相対的な変化を重視）
            if condition_type == "entry":
                return random.uniform(-1000, 0)  # 下降トレンド
            else:
                return random.uniform(0, 1000)  # 上昇トレンド

        elif "EMV" in operand:
            # EMV: 移動の容易さ（通常は小さな値）
            if condition_type == "entry":
                return random.uniform(-0.1, 0)  # 困難な移動
            else:
                return random.uniform(0, 0.1)  # 容易な移動

        elif any(indicator_type in operand for indicator_type in ["ROCP", "ROCR"]):
            # ROCP, ROCR: 変化率指標
            if condition_type == "entry":
                return random.uniform(-5, 5)  # エントリー閾値
            else:
                return random.uniform(-3, 3)  # エグジット閾値

        elif "STOCHF" in operand:
            # STOCHF: 0-100の範囲
            if condition_type == "entry":
                return random.uniform(20, 80)  # エントリー閾値
            else:
                return random.uniform(30, 70)  # エグジット閾値

        else:
            # その他の場合は汎用的な値
            return random.uniform(0.95, 1.05)

    def _generate_risk_management(self) -> Dict[str, float]:
        """リスク管理設定を生成"""
        return {
            "stop_loss": random.uniform(0.02, 0.05),  # 2-5%
            "take_profit": random.uniform(0.05, 0.15),  # 5-15%
            "position_size": random.uniform(0.1, 0.5),  # 10-50%
        }

    def _generate_fallback_condition(self, condition_type: str) -> Condition:
        """フォールバック用の基本条件を生成"""
        if condition_type == "entry":
            return Condition(left_operand="close", operator=">", right_operand="SMA_20")
        else:
            return Condition(left_operand="close", operator="<", right_operand="SMA_20")

    def _generate_fallback_gene(self) -> StrategyGene:
        """フォールバック用の最小限の遺伝子を生成"""
        indicators = [
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        ]

        entry_conditions = [
            Condition(left_operand="close", operator=">", right_operand="SMA_20")
        ]

        exit_conditions = [
            Condition(left_operand="close", operator="<", right_operand="SMA_20")
        ]

        return StrategyGene(
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            risk_management={"stop_loss": 0.03, "take_profit": 0.1},
            metadata={"generated_by": "RandomGeneGenerator_fallback"},
        )

    def generate_population(self, size: int) -> List[StrategyGene]:
        """
        ランダム個体群を生成

        Args:
            size: 個体群サイズ

        Returns:
            生成された戦略遺伝子のリスト
        """
        population = []

        for i in range(size):
            try:
                gene = self.generate_random_gene()
                population.append(gene)

                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{size} random genes")

            except Exception as e:
                logger.error(f"Failed to generate gene {i}: {e}")
                # フォールバックを追加
                population.append(self._generate_fallback_gene())

        logger.info(f"Generated population of {len(population)} genes")
        return population
