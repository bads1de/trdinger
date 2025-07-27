"""
ランダム遺伝子生成器

OI/FRデータを含む多様な戦略遺伝子をランダムに生成します。
スケール不一致問題を解決するため、オペランドグループ化システムを使用します。
"""

import random
from typing import List, Dict

import logging

from ..models.gene_strategy import StrategyGene, IndicatorGene, Condition
from ..models.ga_config import GAConfig
from ..models.gene_tpsl import TPSLGene, TPSLMethod, create_random_tpsl_gene
from ..models.gene_decoder import GeneDecoder
from app.services.indicators import TechnicalIndicatorService
from .smart_condition_generator import SmartConditionGenerator
from ...indicators.config import indicator_registry
from ...indicators.config.indicator_config import IndicatorScaleType
from ..utils.parameter_generators import (
    generate_indicator_parameters,
)
from ..utils.operand_grouping import operand_grouping_system

logger = logging.getLogger(__name__)


class RandomGeneGenerator:
    """
    ランダム戦略遺伝子生成器

    OI/FRデータソースを含む多様な戦略遺伝子を生成します。
    """

    def __init__(self, config: GAConfig, enable_smart_generation: bool = True):
        """
        初期化

        Args:
            config: GA設定オブジェクト
            enable_smart_generation: SmartConditionGeneratorを使用するか
        """
        self.config = config
        self.decoder = GeneDecoder(
            enable_smart_generation
        )  # GeneDecoderのインスタンスを作成
        self.smart_condition_generator = SmartConditionGenerator(
            enable_smart_generation
        )

        # 設定値を取得（型安全）
        self.max_indicators = config.max_indicators
        self.min_indicators = config.min_indicators
        self.max_conditions = config.max_conditions
        self.min_conditions = config.min_conditions
        self.threshold_ranges = config.threshold_ranges

        # 利用可能な指標タイプを指標モードに応じて設定
        self.indicator_service = TechnicalIndicatorService()
        self.available_indicators = self._setup_indicators_by_mode(config)

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
        self.available_operators = [">", "<", ">=", "<="]

    def generate_random_gene(self) -> StrategyGene:
        """
        ランダムな戦略遺伝子を生成

        Returns:
            生成された戦略遺伝子
        """
        try:
            # 指標を生成
            indicators = self._generate_random_indicators()

            # 条件を生成（後方互換性のため保持）
            entry_conditions = self._generate_random_conditions(indicators, "entry")

            # TP/SL遺伝子を先に生成してイグジット条件生成を調整
            tpsl_gene = self._generate_tpsl_gene()

            # MLオンリーモードの場合は常にTP/SL遺伝子を有効化
            indicator_mode = getattr(self.config, "indicator_mode", "mixed")
            if indicator_mode == "ml_only" and tpsl_gene:
                tpsl_gene.enabled = True

            # TP/SL遺伝子が有効な場合はイグジット条件を最小化
            if tpsl_gene and tpsl_gene.enabled:
                # TP/SL機能が有効な場合は空のイグジット条件を生成
                exit_conditions = []
            else:
                # TP/SL機能が無効な場合は従来通りイグジット条件を生成
                exit_conditions = self._generate_random_conditions(indicators, "exit")

            # ロング・ショート条件を生成（SmartConditionGeneratorを使用）
            long_entry_conditions, short_entry_conditions, _ = (
                self.smart_condition_generator.generate_balanced_conditions(indicators)
            )

            # リスク管理設定（従来方式、後方互換性のため保持）
            risk_management = self._generate_risk_management()

            # ポジションサイジング遺伝子を生成（GA最適化対象）
            position_sizing_gene = self._generate_position_sizing_gene()

            gene = StrategyGene(
                indicators=indicators,
                entry_conditions=entry_conditions,  # 後方互換性
                exit_conditions=exit_conditions,
                long_entry_conditions=long_entry_conditions,  # 新機能
                short_entry_conditions=short_entry_conditions,  # 新機能
                risk_management=risk_management,
                tpsl_gene=tpsl_gene,  # 新しいTP/SL遺伝子
                position_sizing_gene=position_sizing_gene,  # 新しいポジションサイジング遺伝子
                metadata={"generated_by": "RandomGeneGenerator"},
            )

            # logger.info(
            #     f"ランダム戦略遺伝子生成成功: 指標={len(indicators)}, エントリー={len(entry_conditions)}, エグジット={len(exit_conditions)}"
            # )
            return gene

        except Exception as e:
            logger.error(f"ランダム戦略遺伝子生成失敗: {e}", exc_info=True)
            # フォールバック: 最小限の遺伝子を生成
            # logger.info("フォールバック戦略遺伝子を生成")
            from ..utils.strategy_gene_utils import create_default_strategy_gene

            return create_default_strategy_gene(StrategyGene)

    def _setup_indicators_by_mode(self, config: GAConfig) -> List[str]:
        """
        指標モードに応じて利用可能な指標を設定

        Args:
            config: GA設定

        Returns:
            利用可能な指標のリスト
        """
        # テクニカル指標を取得
        technical_indicators = list(
            self.indicator_service.get_supported_indicators().keys()
        )

        # ML指標
        ml_indicators = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]

        # 指標モードに応じて選択
        indicator_mode = getattr(config, "indicator_mode", "mixed")

        if indicator_mode == "technical_only":
            # テクニカル指標のみ
            available_indicators = technical_indicators
            logger.info(
                f"指標モード: テクニカルオンリー ({len(available_indicators)}個の指標)"
            )

        elif indicator_mode == "ml_only":
            # ML指標のみ
            available_indicators = ml_indicators
            logger.info(f"指標モード: MLオンリー ({len(available_indicators)}個の指標)")

        else:  # mixed または未設定
            # 両方使用（デフォルト）
            available_indicators = technical_indicators + ml_indicators
            logger.info(
                f"指標モード: 混合 (テクニカル: {len(technical_indicators)}, ML: {len(ml_indicators)})"
            )

        return available_indicators

    def _generate_random_indicators(self) -> List[IndicatorGene]:
        """ランダムな指標リストを生成"""
        try:
            num_indicators = random.randint(self.min_indicators, self.max_indicators)
            indicators = []

            for i in range(num_indicators):
                try:
                    indicator_type = random.choice(self.available_indicators)

                    parameters = generate_indicator_parameters(indicator_type)

                    # JSON形式対応のIndicatorGene作成
                    indicator_gene = IndicatorGene(
                        type=indicator_type, parameters=parameters, enabled=True
                    )

                    # JSON設定を生成して保存
                    try:
                        json_config = indicator_gene.get_json_config()
                        indicator_gene.json_config = json_config
                    except Exception:
                        pass  # JSON設定生成エラーのログを削除

                    indicators.append(indicator_gene)

                except Exception as e:
                    logger.error(f"指標{i+1}生成エラー: {e}")
                    # エラーが発生した場合はSMAをフォールバックとして使用
                    indicators.append(
                        IndicatorGene(
                            type="SMA", parameters={"period": 20}, enabled=True
                        )
                    )

            return indicators

        except Exception as e:
            logger.error(f"指標リスト生成エラー: {e}")
            # 最低限の指標を返す
            return [IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)]

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
        """オペランドを選択（指標名またはデータソース）

        グループ化システムを考慮した重み付き選択を行います。
        """
        choices = []

        # テクニカル指標名を追加（JSON形式：パラメータなし）
        for indicator_gene in indicators:
            indicator_type = indicator_gene.type
            # JSON形式では指標名にパラメータを含めない
            choices.append(indicator_type)

        # 基本データソースを追加（価格データ）
        basic_sources = ["close", "open", "high", "low"]
        choices.extend(basic_sources * self.config.price_data_weight)

        # 出来高データを追加（重みを調整）
        choices.extend(["volume"] * self.config.volume_data_weight)

        # OI/FRデータソースを追加（重みを抑制）
        choices.extend(["OpenInterest", "FundingRate"] * self.config.oi_fr_data_weight)

        return random.choice(choices) if choices else "close"

    def _choose_right_operand(
        self, left_operand: str, indicators: List[IndicatorGene], condition_type: str
    ):
        """右オペランドを選択（指標名、データソース、または数値）

        グループ化システムを使用して、互換性の高いオペランドを優先的に選択します。
        """
        # 設定された確率で数値を使用（スケール不一致問題を回避）
        if random.random() < self.config.numeric_threshold_probability:
            return self._generate_threshold_value(left_operand, condition_type)

        # 20%の確率で別の指標またはデータソースを使用
        # グループ化システムを使用して厳密に互換性の高いオペランドのみを選択
        compatible_operand = self._choose_compatible_operand(left_operand, indicators)

        # 互換性チェック: 低い互換性の場合は数値にフォールバック
        if compatible_operand != left_operand:
            compatibility = operand_grouping_system.get_compatibility_score(
                left_operand, compatible_operand
            )
            if (
                compatibility < self.config.min_compatibility_score
            ):  # 設定された互換性チェック
                return self._generate_threshold_value(left_operand, condition_type)

        return compatible_operand

    def _choose_compatible_operand(
        self, left_operand: str, indicators: List[IndicatorGene]
    ) -> str:
        """左オペランドと互換性の高い右オペランドを選択

        Args:
            left_operand: 左オペランド
            indicators: 利用可能な指標リスト

        Returns:
            互換性の高い右オペランド
        """
        # 利用可能なオペランドリストを構築
        available_operands = []

        # テクニカル指標を追加
        for indicator_gene in indicators:
            available_operands.append(indicator_gene.type)

        # 基本データソースを追加
        available_operands.extend(["close", "open", "high", "low", "volume"])

        # OI/FRデータソースを追加
        available_operands.extend(["OpenInterest", "FundingRate"])

        # 厳密な互換性チェック（設定値以上のみ許可）
        strict_compatible = operand_grouping_system.get_compatible_operands(
            left_operand,
            available_operands,
            min_compatibility=self.config.strict_compatibility_score,
        )

        if strict_compatible:
            return random.choice(strict_compatible)

        # 厳密な互換性がない場合は高い互換性から選択
        high_compatible = operand_grouping_system.get_compatible_operands(
            left_operand,
            available_operands,
            min_compatibility=self.config.min_compatibility_score,
        )

        if high_compatible:
            return random.choice(high_compatible)

        # フォールバック: 利用可能なオペランドからランダム選択
        # ただし、左オペランドと同じものは除外
        fallback_operands = [op for op in available_operands if op != left_operand]
        if fallback_operands:
            selected = random.choice(fallback_operands)
            return selected

        # 最終フォールバック
        return "close"

    def _generate_threshold_value(self, operand: str, condition_type: str) -> float:
        """オペランドの型に応じて、データ駆動で閾値を生成"""

        # 特殊なデータソースの処理
        if "FundingRate" in operand:
            return self._get_safe_threshold(
                "funding_rate", [0.0001, 0.001], allow_choice=True
            )
        if "OpenInterest" in operand:
            return self._get_safe_threshold(
                "open_interest", [1000000, 50000000], allow_choice=True
            )
        if operand == "volume":
            return self._get_safe_threshold("volume", [1000, 100000])

        # 指標レジストリからスケールタイプを取得
        indicator_config = indicator_registry.get_indicator_config(operand)
        if indicator_config and indicator_config.scale_type:
            scale_type = indicator_config.scale_type
            if scale_type == IndicatorScaleType.OSCILLATOR_0_100:
                return self._get_safe_threshold("oscillator_0_100", [20, 80])
            if scale_type == IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100:
                return self._get_safe_threshold(
                    "oscillator_plus_minus_100", [-100, 100]
                )
            if scale_type == IndicatorScaleType.MOMENTUM_ZERO_CENTERED:
                return self._get_safe_threshold("momentum_zero_centered", [-0.5, 0.5])
            if scale_type == IndicatorScaleType.PRICE_RATIO:
                return self._get_safe_threshold("price_ratio", [0.95, 1.05])
            if scale_type == IndicatorScaleType.PRICE_ABSOLUTE:
                return self._get_safe_threshold("price_ratio", [0.95, 1.05])
            if scale_type == IndicatorScaleType.VOLUME:
                return self._get_safe_threshold("volume", [1000, 100000])

        # フォールバック: 価格ベースの指標として扱う
        return self._get_safe_threshold("price_ratio", [0.95, 1.05])

    def _get_safe_threshold(
        self, key: str, default_range: List[float], allow_choice: bool = False
    ) -> float:
        """設定から値を取得し、安全に閾値を生成する"""
        range_ = self.threshold_ranges.get(key, default_range)

        if isinstance(range_, list):
            if allow_choice and len(range_) > 2:
                # 離散値リストから選択
                try:
                    return float(random.choice(range_))
                except (ValueError, TypeError):
                    # 変換できない場合はフォールバック
                    pass
            if (
                len(range_) >= 2
                and isinstance(range_[0], (int, float))
                and isinstance(range_[1], (int, float))
            ):
                # 範囲から選択
                return random.uniform(range_[0], range_[1])
        # フォールバック
        return random.uniform(default_range[0], default_range[1])

    def _generate_risk_management(self) -> Dict[str, float]:
        """リスク管理設定を生成"""
        # Position Sizingシステムにより、position_sizeは自動最適化されるため固定値を使用
        return {
            "position_size": 0.1,  # デフォルト値（実際にはposition_sizing_geneが使用される）
        }

    def _generate_position_sizing_gene(self):
        """ポジションサイジング遺伝子を生成"""
        try:
            from ..models.gene_position_sizing import create_random_position_sizing_gene

            return create_random_position_sizing_gene(self.config)
        except Exception as e:
            logger.error(f"ポジションサイジング遺伝子生成失敗: {e}")
            # フォールバック: デフォルト遺伝子を返す
            from ..models.gene_position_sizing import (
                PositionSizingGene,
                PositionSizingMethod,
            )

            return PositionSizingGene(
                method=PositionSizingMethod.FIXED_RATIO,
                fixed_ratio=0.1,
                max_position_size=20.0,  # より大きなデフォルト値
                enabled=True,
            )

    def _generate_fallback_condition(self, condition_type: str) -> Condition:
        """フォールバック用の基本条件を生成（JSON形式の指標名）"""
        if condition_type == "entry":
            return Condition(left_operand="close", operator=">", right_operand="SMA")
        else:
            return Condition(left_operand="close", operator="<", right_operand="SMA")

    def generate_population(self, size: int) -> List[StrategyGene]:
        """
        ランダム個体群を生成

        Args:
            size: 個体群サイズ

        Returns:
            生成された戦略遺伝子のリスト
        """
        population = []

        from ..utils.strategy_gene_utils import create_default_strategy_gene

        for i in range(size):
            try:
                gene = self.generate_random_gene()
                population.append(gene)

                if (i + 1) % 10 == 0:
                    # logger.info(f"{i + 1}/{size}個のランダム遺伝子を生成しました")
                    pass

            except Exception as e:
                logger.error(f"遺伝子{i}の生成に失敗しました: {e}")
                # フォールバックを追加
                population.append(create_default_strategy_gene(StrategyGene))
                pass

        # logger.info(f"{len(population)}個の遺伝子の個体群を生成しました")
        return population

    def _generate_tpsl_gene(self) -> TPSLGene:
        """
        TP/SL遺伝子を生成（GA最適化対象）

        Returns:
            生成されたTP/SL遺伝子
        """
        try:
            # GAConfigの設定範囲内でランダムなTP/SL遺伝子を生成
            tpsl_gene = create_random_tpsl_gene()

            # GAConfigの制約を適用（設定されている場合）
            if hasattr(self.config, "tpsl_method_constraints"):
                # 許可されたメソッドのみを使用
                allowed_methods = self.config.tpsl_method_constraints
                if allowed_methods:
                    tpsl_gene.method = random.choice(
                        [TPSLMethod(m) for m in allowed_methods]
                    )

            if hasattr(self.config, "tpsl_sl_range"):
                # SL範囲制約
                sl_min, sl_max = self.config.tpsl_sl_range
                tpsl_gene.stop_loss_pct = random.uniform(sl_min, sl_max)
                tpsl_gene.base_stop_loss = random.uniform(sl_min, sl_max)

            if hasattr(self.config, "tpsl_tp_range"):
                # TP範囲制約
                tp_min, tp_max = self.config.tpsl_tp_range
                tpsl_gene.take_profit_pct = random.uniform(tp_min, tp_max)

            if hasattr(self.config, "tpsl_rr_range"):
                # リスクリワード比範囲制約
                rr_min, rr_max = self.config.tpsl_rr_range
                tpsl_gene.risk_reward_ratio = random.uniform(rr_min, rr_max)

            if hasattr(self.config, "tpsl_atr_multiplier_range"):
                # ATR倍率範囲制約
                atr_min, atr_max = self.config.tpsl_atr_multiplier_range
                tpsl_gene.atr_multiplier_sl = random.uniform(atr_min, atr_max)
                tpsl_gene.atr_multiplier_tp = random.uniform(
                    atr_min * 1.5, atr_max * 2.0
                )

            # logger.debug(
            #     f"TP/SL遺伝子生成: メソッド={tpsl_gene.method.value}, SL={tpsl_gene.stop_loss_pct:.3f}"
            # )
            return tpsl_gene

        except Exception as e:
            logger.error(f"TP/SL遺伝子生成エラー: {e}")
            # フォールバック: デフォルトのTP/SL遺伝子
            return TPSLGene(
                method=TPSLMethod.RISK_REWARD_RATIO,
                stop_loss_pct=0.03,
                take_profit_pct=0.06,
                risk_reward_ratio=2.0,
                base_stop_loss=0.03,
                enabled=True,  # 有効化を明示的に設定
            )
