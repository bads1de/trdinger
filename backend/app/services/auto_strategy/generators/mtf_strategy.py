"""
マルチタイムフレーム戦略生成

上位足のトレンド方向と下位足のエントリートリガーを組み合わせた
高勝率なMTFロジックを生成します。
"""

import copy
import logging
import random
from typing import Any

from ..config.constants import IndicatorType
from ..genes import Condition, ConditionGroup, IndicatorGene

logger = logging.getLogger(__name__)


class MTFStrategy:
    """
    マルチタイムフレーム（MTF）戦略
    """

    # 定数
    MAX_CONDITIONS_SAMPLE = 5
    # サポート対象の最大足は 1d（SUPPORTED_TIMEFRAMES 準拠）のため、
    # 1d をベース足とする実行には上位足が存在しない
    DEFAULT_HIGHER_TF = "1d"
    TIMEFRAME_MAPPING = {
        "1m": ["5m", "15m"],
        "5m": ["30m", "1h"],
        "15m": ["1h", "4h"],
        "30m": ["1h", "4h"],
        "1h": ["4h", "1d"],
        "4h": ["1d"],
    }

    def __init__(self, condition_generator: Any) -> None:
        self.gen = condition_generator

    def generate_conditions(
        self, indicators: list[IndicatorGene]
    ) -> tuple[
        list[Condition | ConditionGroup],
        list[Condition | ConditionGroup],
        list[IndicatorGene],
    ]:
        """
        上位足のトレンドと下位足のトリガーを組み合わせたMTF条件を生成

        上位足のトレンド方向に沿って、下位足のオシレーター等がシグナルを
        出している状態（例: 4時間足SMAが上昇中 AND 1時間足RSIが売られすぎ）を
        構築します。

        生成した条件が参照する上位足指標コピーは第三戻り値として返すため、
        呼び出し側で遺伝子の指標リストへ追加する必要があります
        （追加しないと条件が参照整合性検査で除去される）。

        Args:
            indicators (List[IndicatorGene]): 生成済みの指標遺伝子のリスト。

        Returns:
            (ロング条件, ショート条件, 条件が参照する上位足指標コピー) のタプル。
        """
        # MTF生成は明示的に有効化された場合のみ行う。評価側のウォームアップ
        # 期間延長も同フラグ依存のため、フラグなしで上位足指標を作ると
        # 計算結果が欠損する。
        ga_config = getattr(self.gen, "ga_config_obj", None)
        if not getattr(ga_config, "enable_multi_timeframe", False):
            return [], [], []

        long_conds, short_conds = [], []
        current_tf = self.gen.context.get("timeframe", "1h") or "1h"
        higher_tf = self._determine_higher_tf(current_tf)
        if higher_tf == current_tf:
            # ベース足が最大足（1d）の場合は上位足が存在しないため
            # MTF条件を生成しない（同足の重複指標は無意味）
            return [], [], []

        # 指標を分類
        classified = self.gen._dynamic_classify(indicators)
        trends, triggers = (
            classified[IndicatorType.TREND],
            classified[IndicatorType.MOMENTUM],
        )
        if not trends:
            return [], [], []

        # 上位足トレンド指標の作成（指標数上限と重複を考慮）
        mtf_trends, new_copies = self._create_mtf_indicators(
            indicators, trends, higher_tf
        )
        if not mtf_trends:
            return [], [], []
        targets = triggers if triggers else trends

        # 組み合わせ生成: (上位足トレンド) AND (下位足トリガー)
        for trend_ind in mtf_trends:
            t_name = self.gen._get_indicator_name(trend_ind)
            t_long = self.gen._create_side_condition(trend_ind, "long", t_name)
            t_short = self.gen._create_side_condition(trend_ind, "short", t_name)

            for trig_ind in targets:
                if (
                    trig_ind.type == trend_ind.type
                    and trig_ind.parameters == trend_ind.parameters
                ):
                    continue

                m_name = self.gen._get_indicator_name(trig_ind)
                m_long = self.gen._create_side_condition(trig_ind, "long", m_name)
                m_short = self.gen._create_side_condition(trig_ind, "short", m_name)

                build_and_groups = getattr(type(self.gen), "_build_and_groups", None)
                if build_and_groups is None:
                    long_group = ConditionGroup(
                        operator="AND",
                        conditions=[t_long, m_long],
                    )
                    short_group = ConditionGroup(
                        operator="AND",
                        conditions=[t_short, m_short],
                    )
                else:
                    long_group, short_group = build_and_groups(
                        self.gen,
                        [t_long, m_long],
                        [t_short, m_short],
                    )
                long_conds.append(long_group)
                short_conds.append(short_group)

        # 多すぎる場合は間引く
        def _sample(lst: list[Any]) -> list[Any]:
            return (
                random.sample(lst, self.MAX_CONDITIONS_SAMPLE)
                if len(lst) > self.MAX_CONDITIONS_SAMPLE
                else lst
            )

        sampled_longs = _sample(long_conds)
        sampled_shorts = _sample(short_conds)

        # サンプリング後の条件が実際に参照する新規コピーのみ返す
        # （既存指標の再利用分は遺伝子に登録済みのため返さない）
        used_names = self._collect_operand_names(sampled_longs + sampled_shorts)
        used_copies = [
            ind for ind in new_copies if self.gen._get_indicator_name(ind) in used_names
        ]
        return sampled_longs, sampled_shorts, used_copies

    def _determine_higher_tf(self, current_tf: str) -> str:
        """実行足に基づいて適切な上位足を決定"""
        candidates = self.TIMEFRAME_MAPPING.get(current_tf, [self.DEFAULT_HIGHER_TF])
        return str(random.choice(candidates))

    def _create_mtf_indicators(
        self,
        indicators: list[IndicatorGene],
        trend_indicators: list[IndicatorGene],
        timeframe: str,
    ) -> tuple[list[IndicatorGene], list[IndicatorGene]]:
        """条件生成に使う上位足トレンド指標の集合を返す。

        既に上位足の指標が遺伝子内に存在する場合はそれを再利用し、
        無い分だけディープコピーを作る。コピーは (type, timeframe) の
        重複を作らず、遺伝子の指標数上限（max_indicators）を超えない
        範囲に収める。

        Returns:
            (条件生成に使う指標リスト, 新規作成したコピーのリスト) のタプル。
            新規コピーのみ呼び出し側で遺伝子に登録する。
        """
        ga_config = getattr(self.gen, "ga_config_obj", None)
        max_indicators = getattr(ga_config, "max_indicators", 10)
        room = max(0, max_indicators - len(indicators))

        existing_pairs = {
            (ind.type, getattr(ind, "timeframe", None)) for ind in indicators
        }
        res: list[IndicatorGene] = []
        new_copies: list[IndicatorGene] = []
        res_types: set[str] = set()

        # 1. 既存の上位足トレンド指標を再利用（枠は消費しない）
        for ind in trend_indicators:
            if getattr(ind, "timeframe", None) == timeframe and (
                ind.type not in res_types
            ):
                res.append(ind)
                res_types.add(ind.type)

        # 2. 不足分はコピーを作成（枠の範囲内）
        for ind in trend_indicators:
            if len(new_copies) >= room:
                break
            if ind.type in res_types or (ind.type, timeframe) in existing_pairs:
                continue
            new_ind = copy.deepcopy(ind)
            new_ind.timeframe = timeframe
            res.append(new_ind)
            new_copies.append(new_ind)
            res_types.add(ind.type)
        return res, new_copies

    @staticmethod
    def _collect_operand_names(conditions: list[Any]) -> set[str]:
        """条件リスト内の全オペランド名（文字列のみ）を収集する。"""
        names: set[str] = set()

        def scan(item: Any) -> None:
            if isinstance(item, ConditionGroup):
                for sub in item.conditions:
                    scan(sub)
                return
            for operand in (
                getattr(item, "left_operand", None),
                getattr(item, "right_operand", None),
            ):
                if isinstance(operand, str):
                    names.add(operand)

        for item in conditions:
            scan(item)
        return names
