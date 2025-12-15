"""
マルチタイムフレーム戦略生成

上位足のトレンド方向と下位足のエントリートリガーを組み合わせた
高勝率なMTFロジックを生成します。
"""

import logging
import random
import copy
from typing import List, Union, Tuple, Dict

from ...models import Condition, ConditionGroup, IndicatorGene
from ...config.constants import IndicatorType
from .base_strategy import ConditionStrategy

logger = logging.getLogger(__name__)


class MTFStrategy(ConditionStrategy):
    """
    マルチタイムフレーム（MTF）戦略

    ロジック:
    1. 実行足（Current TF）の1-2段階上の「上位足（Higher TF）」を決定。
    2. トレンド系指標を上位足に適用し、相場の大局的な方向（バイアス）を決定。
    3. オシレーター/モメンタム系指標を実行足に適用し、バイアス方向へのエントリータイミングを決定。
    4. これらをAND条件で結合。
    """

    def generate_conditions(self, indicators: List[IndicatorGene]) -> Tuple[
        List[Union[Condition, ConditionGroup]],
        List[Union[Condition, ConditionGroup]],
        List[Condition],
    ]:
        long_conditions = []
        short_conditions = []
        exit_conditions = []

        # 現在のコンテキストからタイムフレームを取得
        current_tf = self.condition_generator.context.get("timeframe", "1h")
        if not current_tf:
            current_tf = "1h"

        # 上位足を決定
        higher_tf = self._determine_higher_timeframe(current_tf)
        logger.debug(f"MTF戦略生成: Current={current_tf}, Higher={higher_tf}")

        # 指標を分類
        trend_indicators = []
        trigger_indicators = []

        for ind in indicators:
            if not ind.enabled:
                continue

            ind_type = self.condition_generator._get_indicator_type(ind)

            if ind_type == IndicatorType.TREND:
                trend_indicators.append(ind)
            else:
                # MOMENTUM, VOLATILITY, その他はトリガーとして使用
                trigger_indicators.append(ind)

        # トレンド指標がない場合は生成不可
        if not trend_indicators:
            return long_conditions, short_conditions, exit_conditions

        # トレンド指標のコピーを作成し、上位足を設定
        mtf_trend_indicators = self._create_mtf_indicators(trend_indicators, higher_tf)

        # 組み合わせ生成
        # トレンド指標（上位足） x トリガー指標（下位足）
        for trend_ind in mtf_trend_indicators:
            # トレンド方向の条件生成
            try:
                # トレンド判定: Close > SMA (Long bias), Close < SMA (Short bias)
                # 注: _generic_long_conditions は通常 "Indicator > Threshold" を返す
                # トレンド系の場合は "Close > Indicator" のような形式が望ましい場合もあるが、
                # ConditionGeneratorの実装に依存する。
                # ここでは汎用メソッドを使って条件を取得し、必要なら修正する。

                # トレンド系指標の条件生成（上位足）
                trend_longs = self.condition_generator._generic_long_conditions(
                    trend_ind
                )
                trend_shorts = self.condition_generator._generic_short_conditions(
                    trend_ind
                )
            except Exception as e:
                logger.warning(f"MTFトレンド条件生成エラー: {e}")
                continue

            if not trend_longs or not trend_shorts:
                continue

            # トリガー指標との組み合わせ
            targets = trigger_indicators if trigger_indicators else trend_indicators
            # トリガーがない場合は、別のトレンド指標を下位足で使うこともあり得る

            for trigger_ind in targets:
                # 同じ指標の組み合わせは避ける（意味がないため）
                if (
                    trigger_ind.type == trend_ind.type
                    and trigger_ind.parameters == trend_ind.parameters
                ):
                    continue

                try:
                    trigger_longs = self.condition_generator._generic_long_conditions(
                        trigger_ind
                    )
                    trigger_shorts = self.condition_generator._generic_short_conditions(
                        trigger_ind
                    )

                    # 組み合わせ: (上位足Long) AND (下位足Long)
                    for tl in trend_longs:
                        for trl in trigger_longs:
                            group = ConditionGroup(operator="AND", conditions=[tl, trl])
                            long_conditions.append(group)

                    # 組み合わせ: (上位足Short) AND (下位足Short)
                    for ts in trend_shorts:
                        for trs in trigger_shorts:
                            group = ConditionGroup(operator="AND", conditions=[ts, trs])
                            short_conditions.append(group)

                except Exception as e:
                    logger.warning(f"MTFトリガー条件生成エラー: {e}")
                    continue

        # 生成数が多すぎる場合はランダムに間引く
        max_conditions = 5
        if len(long_conditions) > max_conditions:
            long_conditions = random.sample(long_conditions, max_conditions)
        if len(short_conditions) > max_conditions:
            short_conditions = random.sample(short_conditions, max_conditions)

        return long_conditions, short_conditions, exit_conditions

    def _determine_higher_timeframe(self, current_tf: str) -> str:
        """実行足に基づいて適切な上位足を決定"""
        # 単純なマッピング
        mapping = {
            "1m": ["5m", "15m"],
            "5m": ["30m", "1h"],
            "15m": ["1h", "4h"],
            "30m": ["1h", "4h"],
            "1h": ["4h", "1d"],
            "4h": "1d",
            "1d": "1w",
        }

        candidates = mapping.get(current_tf, "1d")

        if isinstance(candidates, list):
            return random.choice(candidates)
        return candidates

    def _create_mtf_indicators(
        self, indicators: List[IndicatorGene], timeframe: str
    ) -> List[IndicatorGene]:
        """指標のディープコピーを作成し、timeframeを設定"""
        mtf_list = []
        for ind in indicators:
            new_ind = copy.deepcopy(ind)
            new_ind.timeframe = timeframe
            # IDを変更して区別できるようにする（オプション）
            if new_ind.id:
                new_ind.id = f"{new_ind.id}_{timeframe}"
            mtf_list.append(new_ind)
        return mtf_list
