"""
ランダム遺伝子生成器

OI/FRデータを含む多様な戦略遺伝子をランダムに生成します。
スケール不一致問題を解決するため、オペランドグループ化システムを使用します。
"""

import logging
import random
from typing import Dict, List

from app.services.indicators import TechnicalIndicatorService
from app.services.indicators.config import indicator_registry
from app.services.indicators.config.indicator_config import IndicatorScaleType
from ..models.ga_config import GAConfig
from ..models.gene_serialization import GeneSerializer
from ..models.gene_strategy import Condition, IndicatorGene, StrategyGene
from ..models.gene_tpsl import TPSLGene, TPSLMethod, create_random_tpsl_gene
from ..utils.operand_grouping import operand_grouping_system
from ..utils.parameter_generators import (
    generate_indicator_parameters,
)
from ..config.constants import OPERATORS, DATA_SOURCES
from .smart_condition_generator import SmartConditionGenerator

logger = logging.getLogger(__name__)


class RandomGeneGenerator:
    """
    ランダム戦略遺伝子生成器

    OI/FRデータソースを含む多様な戦略遺伝子を生成します。
    """

    def __init__(
        self,
        config: GAConfig,
        enable_smart_generation: bool = True,
        smart_context: dict | None = None,
    ):
        """
        初期化

        Args:
            config: GA設定オブジェクト
            enable_smart_generation: SmartConditionGeneratorを使用するか
            smart_context: スマート条件生成のコンテキスト（timeframe/symbol/threshold_profile/regime_gating）
        """
        self.config = config
        self.serializer = GeneSerializer(
            enable_smart_generation
        )  # GeneSerializerのインスタンスを作成
        self.smart_condition_generator = SmartConditionGenerator(
            enable_smart_generation
        )
        # コンテキストがあれば適用
        try:
            smart_context = smart_context or {}
            # デフォルトを強めに: テクニカルオンリー時は成功率改善のため aggressive を既定
            try:
                indicator_mode = getattr(config, "indicator_mode", "mixed")
                if (
                    "threshold_profile" not in smart_context
                    and indicator_mode == "technical_only"
                ):
                    smart_context["threshold_profile"] = "aggressive"
            except Exception:
                pass
            self.smart_condition_generator.set_context(
                timeframe=smart_context.get("timeframe"),
                symbol=smart_context.get("symbol"),
                regime_gating=smart_context.get("regime_gating"),
                threshold_profile=smart_context.get("threshold_profile"),
            )
        except Exception:
            pass

        # 設定値を取得（型安全）
        self.max_indicators = config.max_indicators
        self.min_indicators = config.min_indicators
        self.max_conditions = config.max_conditions
        self.min_conditions = config.min_conditions
        self.threshold_ranges = config.threshold_ranges

        # 利用可能な指標タイプを指標モードに応じて設定
        self.indicator_service = TechnicalIndicatorService()
        self.available_indicators = self._setup_indicators_by_mode(config)

        # カバレッジ向上: allowed_indicators が指定されている場合、
        # 生成呼び出し毎にリストを巡回して少なくとも1つは未登場指標を含める
        try:
            self._coverage_cycle = list(getattr(config, "allowed_indicators", []) or [])
            if self._coverage_cycle:
                random.shuffle(self._coverage_cycle)
        except Exception:
            self._coverage_cycle = []
        self._coverage_idx = 0

        # 利用可能なデータソース
        self.available_data_sources = DATA_SOURCES

        # 利用可能な演算子
        self.available_operators = OPERATORS

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

            # Auto-StrategyではTP/SLを常に有効化し、エグジット条件は冗長のため生成しない
            if tpsl_gene:
                tpsl_gene.enabled = True

            # TP/SL遺伝子が有効な場合はイグジット条件を最小化
            if tpsl_gene and tpsl_gene.enabled:
                exit_conditions = []
            else:
                exit_conditions = self._generate_random_conditions(indicators, "exit")

            # ロング・ショート条件を生成（SmartConditionGeneratorを使用）
            # geneに含まれる指標一覧を渡して、素名比較時のフォールバックを安定化
            try:
                self.smart_condition_generator.indicators = indicators
            except Exception:
                pass
            long_entry_conditions, short_entry_conditions, _ = (
                self.smart_condition_generator.generate_balanced_conditions(indicators)
            )

            # 条件の成立性を底上げ：OR 正規化と価格vsトレンド(or open)フォールバックをコアに委譲
            from app.services.auto_strategy.core.condition_assembly import (
                ConditionAssembly,
            )

            long_entry_conditions = ConditionAssembly.ensure_or_with_fallback(
                long_entry_conditions, "long", indicators
            )
            short_entry_conditions = ConditionAssembly.ensure_or_with_fallback(
                short_entry_conditions, "short", indicators
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
            from ..utils.auto_strategy_utils import AutoStrategyUtils

            return AutoStrategyUtils.create_default_strategy_gene(
                strategy_gene_class=StrategyGene
            )

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

        # allowed_indicators によりさらに絞り込み（安全性と一貫性のため）
        try:
            allowed = set(getattr(config, "allowed_indicators", []) or [])
            if allowed:
                available_indicators = [
                    ind for ind in available_indicators if ind in allowed
                ]
        except Exception:
            pass

        # 安定性のため、デフォルトでは実験的インジケータを除外（allowed_indicators 指定時は尊重）
        # 実験的インジケータはレジストリ定義に従う
        try:
            from app.services.indicators.config import indicator_registry

            experimental = indicator_registry.experimental_indicators
        except Exception:
            experimental = {"RMI", "DPO", "CHOP", "VORTEX", "EOM", "KVO", "PVT", "CMF"}
        try:
            allowed = set(getattr(self.config, "allowed_indicators", []) or [])
            if not allowed:
                available_indicators = [
                    ind for ind in available_indicators if ind not in experimental
                ]
        except Exception:
            available_indicators = [
                ind for ind in available_indicators if ind not in experimental
            ]
        # レジーム判定に利用するCHOPは有用だが、allowed 指定時は尊重する
        try:
            ind_mode = getattr(self.config, "indicator_mode", "mixed")
            allowed = set(getattr(self.config, "allowed_indicators", []) or [])
            if (
                ind_mode == "technical_only"
                and not allowed
                and "CHOP" not in available_indicators
            ):
                available_indicators.append("CHOP")
        except Exception:
            pass
        # テクニカルオンリー時のデフォルト候補を厳選して成立性を底上げ（allowed_indicators 指定時は尊重）
        if indicator_mode == "technical_only":
            try:
                allowed = set(getattr(config, "allowed_indicators", []) or [])
            except Exception:
                allowed = set()
            if not allowed:
                # VALID_INDICATOR_TYPESに含まれる安全な指標のみを使用
                from ..config.shared_constants import VALID_INDICATOR_TYPES

                curated = {
                    "SMA",
                    "EMA",
                    "WMA",
                    "MACD",
                    "MACDFIX",
                    "MACDEXT",
                    "RSI",
                    "STOCH",
                    "STOCHRSI",
                    "CCI",
                    "ADX",
                    "MFI",
                    "ATR",
                    "BBANDS",
                    "KAMA",
                    "T3",
                    "TRIMA",
                    "PPO",
                    "APO",
                    "ROC",
                    "TRIX",
                    "UO",
                    "CMO",
                    "DX",
                    "MINUS_DI",
                    "PLUS_DI",
                    "WILLR",
                    "AROON",
                    "AROONOSC",
                    "BOP",
                    "MOM",
                }
                # VALID_INDICATOR_TYPESに含まれる指標のみに絞り込み
                curated = {ind for ind in curated if ind in VALID_INDICATOR_TYPES}
            # カバレッジモード: allowed 指定時は1つは巡回候補を確実に含める
            coverage_pick = None
            try:
                if self._coverage_cycle:
                    coverage_pick = self._coverage_cycle[
                        self._coverage_idx % len(self._coverage_cycle)
                    ]
                    self._coverage_idx += 1
            except Exception:
                coverage_pick = None

            # curated での厳選は allowed 指定がない場合のみ適用
            try:
                allowed = set(getattr(config, "allowed_indicators", []) or [])
            except Exception:
                allowed = set()
            if not allowed and indicator_mode == "technical_only":
                available_indicators = [
                    ind for ind in available_indicators if ind in curated
                ]

        return available_indicators

    def _generate_random_indicators(self) -> List[IndicatorGene]:
        """ランダムな指標リストを生成"""
        try:
            # coverage_pick は上位で準備済み
            coverage_pick = locals().get("coverage_pick", None)

            num_indicators = random.randint(self.min_indicators, self.max_indicators)
            indicators = []

            # 1つ目は coverage_pick を優先
            if coverage_pick and coverage_pick in self.available_indicators:
                first_type = coverage_pick
            else:
                first_type = (
                    random.choice(self.available_indicators)
                    if self.available_indicators
                    else "SMA"
                )

            try:
                # 1本目
                indicator_type = first_type
                parameters = generate_indicator_parameters(indicator_type)
                indicator_gene = IndicatorGene(
                    type=indicator_type, parameters=parameters, enabled=True
                )
                try:
                    json_config = indicator_gene.get_json_config()
                    indicator_gene.json_config = json_config
                except Exception:
                    pass
                indicators.append(indicator_gene)
            except Exception as e:
                logger.error(f"指標1生成エラー: {e}")
                indicators.append(
                    IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
                )

            # 残り
            for i in range(1, num_indicators):
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

                # allowed_indicators が明示指定されている場合は、均一サンプリングを尊重し、
                # 成立性底上げの補助ロジック（トレンド強制/MA追加）はスキップする
                try:
                    if getattr(self.config, "allowed_indicators", None):
                        return indicators
                except Exception:
                    pass

            # 成立性の底上げ: 少なくとも1つはトレンド系（SMA/EMA/MAMA/MAのいずれか）を含める
            try:

                def _is_trend(name: str) -> bool:
                    cfg = indicator_registry.get_indicator_config(name)
                    return bool(cfg and getattr(cfg, "category", None) == "trend")

                has_trend = any(_is_trend(ind.type) for ind in indicators)
                if not has_trend:
                    # allowed_indicators/available_indicators を尊重してトレンド補完
                    trend_pool = [
                        name for name in self.available_indicators if _is_trend(name)
                    ]
                    if trend_pool:
                        chosen = random.choice(trend_pool)
                        # period が必要なものにのみデフォルトperiodを与える（SMA/EMA 等）
                        default_params = (
                            {"period": random.choice([10, 14, 20, 30, 50])}
                            if chosen in ("SMA", "EMA", "WMA", "TRIMA", "KAMA", "T3")
                            else {}
                        )
                        indicators.append(
                            IndicatorGene(
                                type=chosen, parameters=default_params, enabled=True
                            )
                        )
                        # 上限超過なら非トレンドを1つ削除
                        if len(indicators) > self.max_indicators:
                            for j, ind in enumerate(indicators):
                                if not _is_trend(ind.type):
                                    indicators.pop(j)
                                    break
                # MA系が2本未満ならクロス戦略を可能にするために追加
                try:

                    def _is_ma(name: str) -> bool:
                        # VALID_INDICATOR_TYPESに含まれる移動平均系指標のみ
                        return name in {
                            "SMA",
                            "EMA",
                            "WMA",
                            "TRIMA",
                            "KAMA",
                            "T3",
                        }

                    ma_count = sum(1 for ind in indicators if _is_ma(ind.type))
                    if ma_count < 2:
                        # 追加するMAを選択
                        ma_pool = [
                            name for name in self.available_indicators if _is_ma(name)
                        ]
                        if ma_pool:
                            # 既存のMAのperiodと被らないように
                            existing_periods = set(
                                ind.parameters.get("period")
                                for ind in indicators
                                if _is_ma(ind.type) and isinstance(ind.parameters, dict)
                            )
                            # テスト互換性のため SMA/EMA/MAMA/MA を優先
                            preferred = {"SMA", "EMA", "MAMA", "MA"}
                            pref_pool = [
                                n for n in ma_pool if n in preferred
                            ] or ma_pool
                            chosen = random.choice(pref_pool)
                            period_choices = [7, 10, 12, 14, 20, 30, 50]
                            period = random.choice(
                                [p for p in period_choices if p not in existing_periods]
                                or [14]
                            )
                            indicators.append(
                                IndicatorGene(
                                    type=chosen,
                                    parameters={"period": period},
                                    enabled=True,
                                )
                            )
                            # 上限超過なら非MAを1つ削除
                            if len(indicators) > self.max_indicators:
                                for j, ind in enumerate(indicators):
                                    if not _is_ma(ind.type):
                                        indicators.pop(j)
                                        break
                except Exception:
                    pass
                    # trend_pool が無ければ補完はスキップ（allowed_indicators を厳守）
            except Exception:
                # レジストリ取得が失敗しても安全にSMAを追加（ただしavailable_indicatorsにSMAが含まれる場合のみ）
                if (
                    any(ind.type in ("SMA", "EMA", "MAMA", "MA") for ind in indicators)
                    or "SMA" not in self.available_indicators
                ):
                    pass
                else:
                    indicators.append(
                        IndicatorGene(
                            type="SMA", parameters={"period": 20}, enabled=True
                        )
                    )
                    if len(indicators) > self.max_indicators:
                        indicators = indicators[: self.max_indicators]

            return indicators

        except Exception as e:
            logger.error(f"指標リスト生成エラー: {e}")
            # 最低限の指標を返す
            return [IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)]

    def _generate_random_conditions(
        self, indicators: List[IndicatorGene], condition_type: str
    ) -> List[Condition]:
        """ランダムな条件リストを生成"""
        # 条件数はプロファイルや生成器の方針により 1〜max_conditions に広げる
        # ここでは min_conditions〜max_conditions の範囲で選択（下限>上限にならないようにガード）
        low = int(self.min_conditions)
        high = int(self.max_conditions)
        if high < low:
            low, high = high, low
        num_conditions = random.randint(low, max(low, high))
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
        from ..config.shared_constants import VALID_INDICATOR_TYPES

        choices = []

        # テクニカル指標名を追加（JSON形式：パラメータなし）
        for indicator_gene in indicators:
            indicator_type = indicator_gene.type
            # VALID_INDICATOR_TYPESに含まれる指標のみを使用
            if indicator_type in VALID_INDICATOR_TYPES:
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

        from ..utils.auto_strategy_utils import AutoStrategyUtils

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
                population.append(
                    AutoStrategyUtils.create_default_strategy_gene(
                        strategy_gene_class=StrategyGene
                    )
                )

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

            # テクニカルオンリー時はクローズ成立性を高めるため、固定小幅TP/SLにバイアス
            try:
                if getattr(self.config, "indicator_mode", None) == "technical_only":
                    # 小さめの値にバイアスをかけつつ、メソッド自体は多様性を残す
                    if random.random() < 0.5:
                        tpsl_gene.method = TPSLMethod.FIXED_PERCENTAGE
                    # 小さめの値に再サンプル
                    tpsl_gene.stop_loss_pct = random.uniform(0.005, 0.02)
                    tpsl_gene.take_profit_pct = random.uniform(0.01, 0.05)
                    # RR連動のベースも縮小
                    tpsl_gene.base_stop_loss = max(
                        0.005, min(0.02, tpsl_gene.stop_loss_pct)
                    )
                    # RRは控えめ
                    tpsl_gene.risk_reward_ratio = 1.5
            except Exception:
                pass

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
                enabled=True,
            )
