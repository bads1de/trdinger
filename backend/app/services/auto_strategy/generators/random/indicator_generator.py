"""
指標生成器

ランダム戦略の指標部分を生成する専門ジェネレーター
"""

import logging
import random
from typing import List, Any

from app.services.indicators import TechnicalIndicatorService
from app.services.indicators.config import indicator_registry
from ...constants import CURATED_TECHNICAL_INDICATORS
from ...models.strategy_models import IndicatorGene
from ...utils.indicator_utils import get_all_indicators
from ..indicator_composition_service import IndicatorCompositionService

logger = logging.getLogger(__name__)


class IndicatorGenerator:
    """
    ランダム戦略の指標生成を担当するクラス
    """

    # トレンド系指標の優先順位
    TREND_PREF = (
        "SMA",
        "EMA",
        "WMA",
        "DEMA",
        "TEMA",
        "T3",
        "KAMA",
        "SAR",
    )

    def __init__(self, config: Any):
        """
        初期化

        Args:
            config: GA設定オブジェクト
        """
        self.config = config
        self.indicator_service = TechnicalIndicatorService()
        self._valid_indicator_names = self._initialize_valid_indicators()
        self.composition_service = IndicatorCompositionService(config)
        self.available_indicators = self._setup_indicators_by_mode(config)
        self._setup_coverage_tracking(config)

    def _initialize_valid_indicators(self) -> set:
        """有効な指標名を初期化"""
        try:
            return set(get_all_indicators())
        except Exception:
            return set()

    def _setup_coverage_tracking(self, config: Any):
        """カバレッジトラッキングを設定"""
        try:
            self._coverage_cycle = list(getattr(config, "allowed_indicators", []) or [])
            if self._coverage_cycle:
                random.shuffle(self._coverage_cycle)
            self._coverage_idx = 0
            self._coverage_pick = None
        except Exception:
            self._coverage_cycle = []
            self._coverage_idx = 0
            self._coverage_pick = None

    def _setup_indicators_by_mode(self, config: Any) -> List[str]:
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


        # テクニカル指標のみを使用
        available_indicators = technical_indicators
        logger.info(
            f"指標モード: テクニカルオンリー ({len(available_indicators)}個の指標)"
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
            experimental = {
                "CMF",
                "AROON", "AROONOSC", "BOP", "TRIX", "TSI", "ULTOSC", "CMO", "DX",
                "MINUS_DI", "PLUS_DI", "CFO", "CHOP", "CTI", "RVI", "RVGI", "SMI", "STC", "PVO",
                "TRIMA", "CWMA", "ALMA", "HMA", "RMA", "SWMA", "ZLMA", "VWMA", "FWMA", "HWMA",
                "JMA", "MCGD", "VIDYA", "LINREG", "LINREG_SLOPE", "LINREG_INTERCEPT", "LINREG_ANGLE",
                "NATR", "TRANGE", "HWC", "PDIST", "STOCHRSI", "MACDFIX", "MACDEXT"
            }
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
        # ### 成立性が高い指標のみを使用し、可読性を向上
        curated = CURATED_TECHNICAL_INDICATORS
        # 動的に有効な指標のみに絞り込み
        try:
            curated = {ind for ind in curated if ind in self._valid_indicator_names}
        except Exception:
            curated = set(curated)

        try:
            allowed = set(getattr(config, "allowed_indicators", []) or [])
        except Exception:
            allowed = set()

        # カバレッジモード: allowed 指定時は1つは巡回候補を確実に含める
        # store coverage pick on the instance so other methods can access it
        try:
            if self._coverage_cycle:
                self._coverage_pick = self._coverage_cycle[
                    self._coverage_idx % len(self._coverage_cycle)
                ]
                self._coverage_idx += 1
        except Exception:
            self._coverage_pick = None

        # curated での厳選は allowed 指定がない場合のみ適用
        if not allowed:
            available_indicators = [
                ind for ind in available_indicators if ind in curated
            ]

        return available_indicators

    def generate_random_indicators(self) -> List[IndicatorGene]:
        """ランダムな指標リストを生成"""
        try:
            # coverage_pick は上位で準備済み（インスタンス属性から取得）
            coverage_pick = getattr(self, "_coverage_pick", None)

            num_indicators = random.randint(
                self.config.min_indicators, self.config.max_indicators
            )
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
                parameters = indicator_registry.generate_parameters_for_indicator(
                    indicator_type
                )
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
                    if not self.available_indicators:
                        # 利用可能な指標がない場合はSMAをフォールバックとして使い続ける
                        indicator_type = "SMA"
                    else:
                        indicator_type = random.choice(self.available_indicators)

                    parameters = indicator_registry.generate_parameters_for_indicator(
                        indicator_type
                    )

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

                # allowed_indicators が明示指定されている場合は、均一サンプリングを尊重し、,
                # 成立性底上げの補助ロジック（トレンド強制/MA追加）はスキップする
                try:
                    if getattr(self.config, "allowed_indicators", None):
                        return indicators
                except Exception:
                    pass

            # 指標構成サービスを使用して成立性を底上げ
            try:
                # トレンド指標の強制追加（産業性底上げ）
                indicators = self.composition_service.enhance_with_trend_indicators(
                    indicators, self.available_indicators
                )

                # MAクロス戦略を可能にするための追加
                indicators = self.composition_service.enhance_with_ma_cross_strategy(
                    indicators, self.available_indicators
                )

            except Exception as e:
                logger.error(f"指標構成サービス使用エラー: {e}")
                # フォールバック: SMAを追加（安全策）
                if (
                    any(
                        ind.type in ("SMA", "EMA", "WMA", "DEMA", "TEMA", "T3", "KAMA")
                        for ind in indicators
                    )
                    or "SMA" not in self.available_indicators
                ):
                    pass
                else:
                    indicators.append(
                        IndicatorGene(
                            type="SMA", parameters={"period": 20}, enabled=True
                        )
                    )
                    if len(indicators) > self.config.max_indicators:
                        indicators = indicators[: self.config.max_indicators]

            return indicators

        except Exception as e:
            logger.error(f"指標リスト生成エラー: {e}")
            # 最低限の指標を返す
            return [IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)]
