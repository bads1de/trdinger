"""
IndicatorCompositionService
ロジックをRandomGeneGeneratorから分離し、指標の構成を責任とするサービス。
"""

import logging
import random
from typing import List

from app.services.indicators import TechnicalIndicatorService

from ..config.constants import (
    MA_INDICATORS_NEEDING_PERIOD,
    MOVING_AVERAGE_INDICATORS,
    PREFERRED_MA_INDICATORS,
)
from ..models import IndicatorGene

logger = logging.getLogger(__name__)


class IndicatorCompositionService:
    """
    指標構成のロジックを担当するサービス。
    トレンド指標強制追加やMAクロス戦略などの複雑な構成を管理。
    """

    def __init__(self, config):
        self.config = config
        self.indicator_service = TechnicalIndicatorService()

    def enhance_with_ma_cross_strategy(
        self, indicators: List[IndicatorGene], available_indicators: List[str]
    ) -> List[IndicatorGene]:
        """
        MAクロス戦略を可能にするために複数のMA指標を追加（確率的アプローチ）。

        Args:
            indicators: 現在の指標リスト
            available_indicators: 利用可能な指標リスト

        Returns:
            MAクロス対応の指標リスト
        """
        try:
            # 現在のMA指標数をカウント
            ma_count = sum(
                1 for ind in indicators if ind.type in MOVING_AVERAGE_INDICATORS
            )
            print(f"DEBUG: ma_count={ma_count}")

            # 確率的にMAクロスを導入（強制的ではない）
            rnd_val = random.random()
            print(f"DEBUG: random.random()={rnd_val}")

            if ma_count < 2 and rnd_val < 0.25:  # 25%の確率で導入
                # MA指標プールを準備
                ma_pool = [
                    name
                    for name in available_indicators
                    if name in MOVING_AVERAGE_INDICATORS
                ]
                print(f"DEBUG: ma_pool={ma_pool}")

                if ma_pool:
                    # 既存のMA指標のパラメータと重複しないように選択
                    existing_periods = self._get_existing_periods(indicators)
                    print(f"DEBUG: existing_periods={existing_periods}")

                    chosen_ma = self._choose_ma_with_unique_period(
                        ma_pool, existing_periods
                    )
                    print(f"DEBUG: chosen_ma={chosen_ma}")

                    if chosen_ma:
                        indicators.append(
                            IndicatorGene(
                                type=chosen_ma,
                                parameters=self._get_default_params_for_indicator(
                                    chosen_ma
                                ),
                                enabled=True,
                            )
                        )

                        # 上限を超えた場合は非MA指標を削除
                        if len(indicators) > self.config.max_indicators:
                            self._remove_non_ma_indicator(indicators)

        except Exception as e:
            logger.error(f"MAクロス戦略追加エラー: {e}")
            print(f"DEBUG: EXCEPTION={e}")

        return indicators

    def _get_default_params_for_indicator(self, indicator_type: str) -> dict:
        """指標のデフォルトパラメータを取得"""
        try:
            if indicator_type in MA_INDICATORS_NEEDING_PERIOD:
                return {"period": random.choice([10, 14, 20, 30, 50])}
            else:
                return {}
        except Exception:
            return {"period": 20}  # SMA用フォールバック

    def _get_existing_periods(self, indicators: List[IndicatorGene]) -> set:
        """既存の指標のパラメータを取得"""
        periods = set()
        for ind in indicators:
            if hasattr(ind, "parameters") and isinstance(ind.parameters, dict):
                period = ind.parameters.get("period")
                if period:
                    periods.add(period)
        return periods

    def _choose_ma_with_unique_period(
        self, ma_pool: List[str], existing_periods: set
    ) -> str | None:
        """重複しないperiodのMAを選択"""
        try:
            # 優先MAから選択
            preferred = [name for name in ma_pool if name in PREFERRED_MA_INDICATORS]
            candidates = preferred or ma_pool

            # ランダムに選択
            for ma_type in random.sample(candidates, len(candidates)):
                params = self._get_default_params_for_indicator(ma_type)
                if params.get("period") not in existing_periods:
                    return ma_type

            # 見つからない場合はとりあえず返す
            return candidates[0] if candidates else None

        except Exception as e:
            logger.error(f"MA選択エラー: {e}")
            return None

    def _remove_non_ma_indicator(self, indicators: List[IndicatorGene]):
        """非MA指標を1つ削除"""
        for i, ind in enumerate(indicators):
            if ind.type not in MOVING_AVERAGE_INDICATORS:
                indicators.pop(i)
                break
