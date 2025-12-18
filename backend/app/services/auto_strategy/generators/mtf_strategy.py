"""
マルチタイムフレーム戦略生成

上位足のトレンド方向と下位足のエントリートリガーを組み合わせた
高勝率なMTFロジックを生成します。
"""

import logging
import random
import copy
from typing import List, Union, Tuple, Dict, Any

from ..genes import Condition, ConditionGroup, IndicatorGene
from ..config.constants import IndicatorType

logger = logging.getLogger(__name__)


class MTFStrategy:
    """
    マルチタイムフレーム（MTF）戦略
    """

    def __init__(self, condition_generator: Any) -> None:
        self.gen = condition_generator

    def generate_conditions(self, indicators: List[IndicatorGene]) -> Tuple[
        List[Union[Condition, ConditionGroup]],
        List[Union[Condition, ConditionGroup]],
        List[Condition],
    ]:
        long_conds, short_conds = [], []
        current_tf = self.gen.context.get("timeframe", "1h") or "1h"
        higher_tf = self._determine_higher_tf(current_tf)

        # 指標を分類
        classified = self.gen._classify_indicators(indicators)
        trends, triggers = classified[IndicatorType.TREND], classified[IndicatorType.MOMENTUM]
        if not trends: return [], [], []

        # 上位足トレンド指標の作成
        mtf_trends = self._create_mtf_indicators(trends, higher_tf)
        targets = triggers if triggers else trends

        # 組み合わせ生成: (上位足トレンド) AND (下位足トリガー)
        for trend_ind in mtf_trends:
            t_name = self.gen._get_indicator_name(trend_ind)
            t_long = self.gen._create_side_condition(trend_ind, "long", t_name)
            t_short = self.gen._create_side_condition(trend_ind, "short", t_name)

            for trig_ind in targets:
                if trig_ind.type == trend_ind.type and trig_ind.parameters == trend_ind.parameters:
                    continue
                
                m_name = self.gen._get_indicator_name(trig_ind)
                m_long = self.gen._create_side_condition(trig_ind, "long", m_name)
                m_short = self.gen._create_side_condition(trig_ind, "short", m_name)

                long_conds.append(ConditionGroup(operator="AND", conditions=[t_long, m_long]))
                short_conds.append(ConditionGroup(operator="AND", conditions=[t_short, m_short]))

        # 多すぎる場合は間引く
        def _sample(lst): return random.sample(lst, 5) if len(lst) > 5 else lst
        return _sample(long_conds), _sample(short_conds), []

    def _determine_higher_tf(self, current_tf: str) -> str:
        """実行足に基づいて適切な上位足を決定"""
        mapping = {
            "1m": ["5m", "15m"], "5m": ["30m", "1h"], "15m": ["1h", "4h"],
            "30m": ["1h", "4h"], "1h": ["4h", "1d"], "4h": "1d", "1d": "1w",
        }
        res = mapping.get(current_tf, "1d")
        return random.choice(res) if isinstance(res, list) else res

    def _create_mtf_indicators(self, indicators: List[IndicatorGene], timeframe: str) -> List[IndicatorGene]:
        """指標のディープコピーを作成し、timeframeを設定"""
        res = []
        for ind in indicators:
            new_ind = copy.deepcopy(ind)
            new_ind.timeframe = timeframe
            if new_ind.id: new_ind.id = f"{new_ind.id}_{timeframe}"
            res.append(new_ind)
        return res

    # テスト互換用エイリアス
    def _determine_higher_timeframe(self, tf): return self._determine_higher_tf(tf)
