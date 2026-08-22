"""
StrategyGene の突然変異（mutation）演算ロジック。

遺伝的アルゴリズムにおける指標・条件・リスクパラメータ・サブ遺伝子の
突然変異処理を提供します。
"""

import inspect
import logging
import random
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np

from ...config.constants import PositionSizingMethod, TPSLMethod
from ...config.ga_config import GAConfig
from ...utils.indicator_references import build_indicator_reference_name
from ...utils.scale_compat import (
    default_threshold_for,
    is_close_comparable_type,
)
from ..conditions import Condition, ConditionGroup, StatefulCondition
from ..entry import EntryGene, create_random_entry_gene
from ..exit import ExitGene, create_random_exit_gene
from ..position_sizing import (
    PositionSizingGene,
    create_random_position_sizing_gene,
)
from ..strategy import StrategyGene
from ..tpsl import TPSLGene, create_random_tpsl_gene

logger = logging.getLogger(__name__)

WEEKEND_FILTER_NAME = "weekend_filter"


def _ensure_weekend_filter(tool_genes: list[Any]) -> list[Any]:
    """weekend_filter が常時ONで存在することを保証する。"""
    from ..tool import ToolGene

    existing = next(
        (t for t in tool_genes if getattr(t, "tool_name", None) == WEEKEND_FILTER_NAME),
        None,
    )
    if existing is not None:
        existing.enabled = True
    else:
        tool_genes.append(
            ToolGene(tool_name=WEEKEND_FILTER_NAME, enabled=True, params={})
        )
    return tool_genes


_SUB_GENE_MUTATION_RULES = {
    TPSLGene: (
        create_random_tpsl_gene,
        "tpsl_gene_creation_multiplier",
    ),
    PositionSizingGene: (
        create_random_position_sizing_gene,
        "position_sizing_gene_creation_multiplier",
    ),
    EntryGene: (
        create_random_entry_gene,
        "entry_gene_creation_multiplier",
    ),
    ExitGene: (
        create_random_exit_gene,
        "exit_gene_creation_multiplier",
    ),
}


def _create_sub_gene(creator_func: Callable[..., object], config: object) -> object:
    """生成関数のシグネチャに応じて設定を渡し分ける。"""
    try:
        signature = inspect.signature(creator_func)
    except (TypeError, ValueError):
        return creator_func()

    if len(signature.parameters) == 0:
        return creator_func()
    return creator_func(config)


def _iter_mutable_sub_gene_specs(
    config: GAConfig,
) -> list[tuple[str, object, float]]:
    """StrategyGene の定義を基準に、突然変異対象のサブ遺伝子を列挙する。"""
    class_map = StrategyGene.sub_gene_class_map()
    specs: list[tuple[str, object, float]] = []

    for field_name in StrategyGene.sub_gene_field_names():
        gene_class = class_map.get(field_name)
        rule = _SUB_GENE_MUTATION_RULES.get(gene_class)  # type: ignore[arg-type]

        if rule is None:
            continue

        creator_func, creation_prob_attr = rule
        try:
            creation_prob_mult = float(
                getattr(config.mutation_config, creation_prob_attr) or 0.0
            )
        except (AttributeError, TypeError, ValueError):
            creation_prob_mult = 0.0

        specs.append((field_name, creator_func, creation_prob_mult))

    return specs


def _mutate_threshold(val: int | float) -> int | float:
    """数値オペランド（閾値）の突然変異"""
    if isinstance(val, bool):
        return val
    # オシレーター等 0-100 の範囲
    if 0.0 <= val <= 100.0 and val > 1.0:
        factor = random.uniform(0.85, 1.15)
        new_val = val * factor
        new_val = max(0.0, min(100.0, new_val))
    # 0.0 - 1.0 の割合（例: 0.02, 0.05 等）
    elif 0.0 < val <= 1.0:
        factor = random.uniform(0.85, 1.15)
        new_val = max(0.0001, min(1.0, val * factor))
    else:
        factor = random.uniform(0.85, 1.15)
        new_val = val * factor

    if isinstance(val, int):
        return int(round(new_val))
    return round(new_val, 4)


def _mutate_operands(
    condition: Condition,
    strategy: StrategyGene,
    valid_names: set[str],
) -> None:
    """条件オペランドのスワップ変異"""
    price_operands = ["close", "open", "high", "low"]

    # valid_names と strategy.indicators を突き合わせて価格系と非価格系を正確に分類
    price_candidates: list[str] = list(price_operands)
    non_price_candidates: list[str] = []

    for ind in strategy.indicators:
        ref_name = build_indicator_reference_name(ind)
        cand_names = [ref_name, ind.type]
        timeframe = getattr(ind, "timeframe", None)
        if timeframe:
            cand_names.append(f"{ind.type}_{timeframe}")

        # valid_names に登録されている有効な名前のみを採用
        for name in cand_names:
            if name in valid_names:
                if is_close_comparable_type(ind.type):
                    if name not in price_candidates:
                        price_candidates.append(name)
                else:
                    if name not in non_price_candidates:
                        non_price_candidates.append(name)

    for attr in ("left_operand", "right_operand"):
        current = getattr(condition, attr)
        if not isinstance(current, str):
            continue

        # 価格系オペランド（価格列または価格系指標）に該当する場合
        if current.lower() in price_operands or current in price_candidates:
            alt_candidates = [c for c in price_candidates if c != current]
            if alt_candidates:
                setattr(condition, attr, random.choice(alt_candidates))
        # 非価格系指標に該当する場合
        elif current in non_price_candidates:
            alt_candidates = [c for c in non_price_candidates if c != current]
            if alt_candidates:
                setattr(condition, attr, random.choice(alt_candidates))


def _mutate_condition_leaf(
    condition: Condition,
    strategy: StrategyGene,
    config: GAConfig,
    valid_names: set[str],
) -> None:
    """単一条件の突然変異（演算子・閾値・オペランド）"""
    # 1. 演算子の変異
    if random.random() < config.mutation_config.condition_operator_switch_probability:
        try:
            condition.operator = random.choice(
                config.mutation_config.valid_condition_operators
            )
        except Exception as e:
            logger.debug(f"演算子変異スキップ: {e}")

    # 2. 閾値（数値オペランド）の変異
    if (
        random.random()
        < config.mutation_config.condition_threshold_mutation_probability
    ):
        for attr in ("right_operand", "left_operand"):
            val = getattr(condition, attr)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                setattr(condition, attr, _mutate_threshold(val))

    # 3. オペランドのスワップ変異
    if random.random() < config.mutation_config.condition_operand_swap_probability:
        _mutate_operands(condition, strategy, valid_names)


def _mutate_condition_topology(
    conditions: list[Condition | ConditionGroup],
    strategy: StrategyGene,
    config: GAConfig,
    side: str = "long",
) -> None:
    """条件リストのトポロジー変異（条件追加・枝刈り）"""
    max_conds = getattr(config, "max_conditions", 3)
    min_conds = getattr(config, "min_conditions", 1)

    # 削除（枝刈り）: 条件が min_conditions より多ければ 1 つ削除
    if len(conditions) > min_conds and random.random() < 0.5:
        idx = random.randint(0, len(conditions) - 1)
        conditions.pop(idx)
    # 追加: 有効な指標から条件を生成して追加（max_conditions 未満の場合のみ）
    elif len(conditions) < max_conds and strategy.indicators:
        # 20% の確率で ConditionGroup（AND/OR + 2条件）を生成してネスト構造を導入
        if (
            random.random() < 0.2
            and len(conditions) + 1 <= max_conds
            and len(strategy.indicators) >= 1
        ):
            try:
                conds: list[Condition | ConditionGroup] = []
                for _ in range(2):
                    ind = random.choice(strategy.indicators)
                    ref_name = build_indicator_reference_name(ind)
                    if is_close_comparable_type(ind.type):
                        op = ">" if side == "long" else "<"
                        conds.append(
                            Condition(
                                left_operand="close",
                                operator=op,
                                right_operand=ref_name,
                            )
                        )
                    else:
                        thresh = default_threshold_for(
                            ind.type, bullish=(side == "long")
                        )
                        op = "<" if side == "long" else ">"
                        conds.append(
                            Condition(
                                left_operand=ref_name,
                                operator=op,
                                right_operand=thresh,
                            )
                        )
                group = ConditionGroup(
                    operator=random.choice(["AND", "OR"]), conditions=conds
                )
                conditions.append(group)
                return
            except Exception as e:
                logger.debug(f"ConditionGroup生成スキップ: {e}")
        ind = random.choice(strategy.indicators)
        ref_name = build_indicator_reference_name(ind)
        if is_close_comparable_type(ind.type):
            op = ">" if side == "long" else "<"
            new_cond = Condition(
                left_operand="close", operator=op, right_operand=ref_name
            )
        else:
            thresh = default_threshold_for(ind.type, bullish=(side == "long"))
            op = "<" if side == "long" else ">"
            new_cond = Condition(
                left_operand=ref_name, operator=op, right_operand=thresh
            )
        conditions.append(new_cond)


# レジストリ照会失敗時のフォールバック用 整数パラメータ名（実パラメータ名を含む）
_LEGACY_INTEGER_PARAM_NAMES = frozenset(
    {
        "period",
        "signal_period",
        "lookback",
        "length",
        "fast_period",
        "slow_period",
        "fast",
        "slow",
        "medium",
        "signal",
        "cycle",
        "d1",
        "d2",
        "drift",
        "max_lookback",
        "min_lookback",
        "roc_period",
        "atr_period",
        "std_dev_period",
        "mom_period",
        "cci_period",
        "willr_period",
        "stoch_k_period",
        "stoch_d_period",
        "stoch_slowk",
        "stoch_slowd",
        "obv_period",
        "ad_period",
        "adx_period",
        "aroon_period",
        "bop_period",
    }
)


def _is_integer_parameter(indicator_type: str, param_name: str) -> bool:
    """
    パラメータが整数かどうかをレジストリ定義から判定する。

    従来の名称リスト（fast_period 等）と異なり、実際のパラメータ名
    （MACD の fast/slow など）にも追従でき、実数化した期間の
    小数変異を防げる。レジストリ照会に失敗した場合は名称リストへ
    フォールバックする。
    """
    try:
        from app.services.indicators.config import indicator_registry

        config = indicator_registry.get_indicator_config(indicator_type)
        if config and param_name in config.parameters:
            default_value = config.parameters[param_name].default_value
            return isinstance(default_value, int) and not isinstance(
                default_value, bool
            )
    except Exception:
        pass
    return param_name in _LEGACY_INTEGER_PARAM_NAMES


def mutate_indicators(
    mutated: StrategyGene, mutation_rate: float, config: GAConfig
) -> None:
    """指標遺伝子の突然変異処理。"""
    min_multiplier, max_multiplier = config.mutation_config.indicator_param_range

    for i, indicator in enumerate(mutated.indicators):
        # パラメータ変異: 各パラメータを mutation_rate で変異（二重ゲートを解消）
        for param_name, param_value in list(indicator.parameters.items()):
            if (
                isinstance(param_value, (int, float))
                and random.random() < mutation_rate
            ):
                was_integer = isinstance(param_value, int)

                if (
                    param_name == "period"
                    and hasattr(config, "parameter_ranges")
                    and "period" in config.parameter_ranges
                ):
                    min_p, max_p = config.parameter_ranges["period"]
                    mutated.indicators[i].parameters[param_name] = max(
                        min_p,
                        min(
                            max_p,
                            int(
                                param_value
                                * random.uniform(min_multiplier, max_multiplier)
                            ),
                        ),
                    )
                elif was_integer or _is_integer_parameter(indicator.type, param_name):
                    new_value = int(
                        param_value * random.uniform(min_multiplier, max_multiplier)
                    )
                    mutated.indicators[i].parameters[param_name] = max(1, new_value)
                else:
                    mutated.indicators[i].parameters[param_name] = (
                        param_value * random.uniform(min_multiplier, max_multiplier)
                    )

        # 指標タイプの変異: 別タイプへスイッチ（パラメータは新タイプ用に再生成）
        if random.random() < mutation_rate * 0.15:
            try:
                from ...config.indicator_universe import get_indicator_universe_names
                from ..indicator import create_random_indicator_gene

                available = list(get_indicator_universe_names(config))  # type: ignore[arg-type]
                if available:
                    new_type = random.choice(available)
                    if new_type != indicator.type:
                        new_gene = create_random_indicator_gene(
                            new_type, config, timeframe=indicator.timeframe
                        )
                        # id と enabled は旧指標を引き継ぐ
                        new_gene.id = indicator.id
                        new_gene.enabled = indicator.enabled
                        mutated.indicators[i] = new_gene
                        indicator = new_gene
            except Exception as e:
                logger.debug(f"指標タイプ変異スキップ: {e}")

        # タイムフレーム変異（MTF有効時のみ）
        if random.random() < mutation_rate * 0.15:
            try:
                if getattr(config, "enable_multi_timeframe", False):
                    from app.config.constants import SUPPORTED_TIMEFRAMES

                    available_tfs = (
                        getattr(config, "available_timeframes", None)
                        or SUPPORTED_TIMEFRAMES
                    )
                    # 30% で None（デフォルトTF）、それ以外でランダムTF
                    if indicator.timeframe is None:
                        if random.random() < 0.5:
                            mutated.indicators[i].timeframe = random.choice(
                                available_tfs
                            )
                    else:
                        if random.random() < 0.3:
                            mutated.indicators[i].timeframe = None
                        else:
                            mutated.indicators[i].timeframe = random.choice(
                                available_tfs
                            )
                else:
                    mutated.indicators[i].timeframe = None
            except Exception as e:
                logger.debug(f"タイムフレーム変異スキップ: {e}")

        # 有効/無効フラグの変異
        if random.random() < mutation_rate * 0.1:
            mutated.indicators[i].enabled = not mutated.indicators[i].enabled

        # json_config のリフレッシュ（パラメータ/type/timeframe 変異後）
        try:
            mutated.indicators[i].json_config = mutated.indicators[i].get_json_config()
        except Exception as e:
            logger.debug(f"json_config更新スキップ: {e}")

    if (
        random.random()
        < mutation_rate * config.mutation_config.indicator_add_delete_probability
    ):
        max_indicators = config.max_indicators
        if (
            len(mutated.indicators) < max_indicators
            and random.random()
            < config.mutation_config.indicator_add_vs_delete_probability
        ):
            from ..indicator import generate_random_indicators
            from ..validator import GeneValidator

            validator = GeneValidator()
            new_indicators = generate_random_indicators(config)
            allowed_indicators = {indicator.type for indicator in new_indicators}
            valid_new_indicators = [
                indicator
                for indicator in new_indicators
                if validator.validate_indicator_gene_for_generation(
                    indicator,
                    indicator_universe_mode=getattr(
                        config, "indicator_universe_mode", "curated"
                    ),
                    allowed_indicators=allowed_indicators,
                )
            ]
            if valid_new_indicators:
                new_ind = random.choice(valid_new_indicators)
                mutated.indicators.append(new_ind)

                # 新指標を利用する条件をエントリー条件にスマート注入（max_conditions を上限として遵守）
                if (
                    random.random()
                    < config.mutation_config.indicator_condition_injection_probability
                ):
                    max_conds = getattr(config, "max_conditions", 3)
                    ref_name = build_indicator_reference_name(new_ind)
                    if is_close_comparable_type(new_ind.type):
                        if len(mutated.long_entry_conditions) < max_conds:
                            mutated.long_entry_conditions.append(
                                Condition(
                                    left_operand="close",
                                    operator=">",
                                    right_operand=ref_name,
                                )
                            )
                        if len(mutated.short_entry_conditions) < max_conds:
                            mutated.short_entry_conditions.append(
                                Condition(
                                    left_operand="close",
                                    operator="<",
                                    right_operand=ref_name,
                                )
                            )
                    else:
                        long_thresh = default_threshold_for(new_ind.type, bullish=True)
                        short_thresh = default_threshold_for(
                            new_ind.type, bullish=False
                        )
                        if len(mutated.long_entry_conditions) < max_conds:
                            mutated.long_entry_conditions.append(
                                Condition(
                                    left_operand=ref_name,
                                    operator="<",
                                    right_operand=long_thresh,
                                )
                            )
                        if len(mutated.short_entry_conditions) < max_conds:
                            mutated.short_entry_conditions.append(
                                Condition(
                                    left_operand=ref_name,
                                    operator=">",
                                    right_operand=short_thresh,
                                )
                            )
        elif len(mutated.indicators) > config.min_indicators and random.random() < (
            1 - config.mutation_config.indicator_add_vs_delete_probability
        ):
            mutated.indicators.pop(random.randint(0, len(mutated.indicators) - 1))


def mutate_conditions(
    mutated: StrategyGene, mutation_rate: float, config: GAConfig
) -> None:
    """条件の突然変異処理。"""
    valid_names = mutated.valid_operand_names()

    def mutate_item(condition: Condition | ConditionGroup) -> None:
        if isinstance(condition, ConditionGroup):
            if (
                random.random()
                < config.mutation_config.condition_operator_switch_probability
            ):
                condition.operator = "AND" if condition.operator == "OR" else "OR"
            elif condition.conditions:
                idx = random.randint(0, len(condition.conditions) - 1)
                mutate_item(condition.conditions[idx])
        elif isinstance(condition, Condition):
            _mutate_condition_leaf(condition, mutated, config, valid_names)

    mutation_threshold = (
        mutation_rate * config.mutation_config.condition_change_multiplier
    )

    def maybe_mutate_branch(
        conditions: list[Condition | ConditionGroup], side: str
    ) -> None:
        # 1. トポロジー変異（条件追加・枝刈り）
        if random.random() < config.mutation_config.condition_add_delete_probability:
            _mutate_condition_topology(conditions, mutated, config, side=side)

        # 2. 条件アイテムの変異
        if (
            conditions
            and random.random() < config.mutation_config.condition_selection_probability
        ):
            idx = random.randint(0, len(conditions) - 1)
            mutate_item(conditions[idx])

    for conditions, side in (
        (mutated.long_entry_conditions, "long"),
        (mutated.short_entry_conditions, "short"),
        (mutated.long_exit_conditions, "long"),
        (mutated.short_exit_conditions, "short"),
    ):
        if random.random() < mutation_threshold:
            maybe_mutate_branch(conditions, side)


def mutate_stateful_conditions(
    mutated: StrategyGene, mutation_rate: float, config: GAConfig
) -> None:
    """ステートフル条件の突然変異処理。"""
    valid_names = mutated.valid_operand_names()
    max_stateful = 3  # 設定に上限がないため固定

    # 既存の stateful 条件の変異
    for sc in list(mutated.stateful_conditions):
        if random.random() < mutation_rate:
            # lookback_bars の摂動
            if random.random() < 0.3:
                sc.lookback_bars = max(
                    1, min(20, int(sc.lookback_bars * random.uniform(0.8, 1.2)) or 1)
                )
            # cooldown_bars の摂動
            if random.random() < 0.3:
                sc.cooldown_bars = max(
                    0, min(20, int(sc.cooldown_bars * random.uniform(0.8, 1.2)))
                )
            # direction の反転
            if random.random() < 0.2:
                sc.direction = "short" if sc.direction == "long" else "long"
            # enabled トグル
            if random.random() < 0.2:
                sc.enabled = not sc.enabled
            # trigger / follow 条件の変異
            if random.random() < 0.5:
                _mutate_condition_leaf(
                    sc.trigger_condition, mutated, config, valid_names
                )
            if random.random() < 0.5:
                _mutate_condition_leaf(
                    sc.follow_condition, mutated, config, valid_names
                )
            # オペランドスワップは _mutate_condition_leaf 内で確率的に実行されるが、追加で単独スワップも
            if (
                random.random()
                < config.mutation_config.condition_operand_swap_probability
            ):
                _mutate_operands(sc.trigger_condition, mutated, valid_names)
            if (
                random.random()
                < config.mutation_config.condition_operand_swap_probability
            ):
                _mutate_operands(sc.follow_condition, mutated, valid_names)

    # トポロジー変異: 追加・削除
    if random.random() < mutation_rate * 0.2:
        # 削除
        if (
            mutated.stateful_conditions
            and len(mutated.stateful_conditions) > 0
            and random.random() < 0.5
        ):
            # 1件以上あればランダムに削除（少なくとも1件は削除可能）
            idx = random.randint(0, len(mutated.stateful_conditions) - 1)
            mutated.stateful_conditions.pop(idx)
        # 追加（指標がある場合のみ）
        elif len(mutated.stateful_conditions) < max_stateful and mutated.indicators:
            try:
                ind_trigger = random.choice(mutated.indicators)
                ind_follow = random.choice(mutated.indicators)
                ref_trigger = build_indicator_reference_name(ind_trigger)
                ref_follow = build_indicator_reference_name(ind_follow)

                # トリガー条件の生成
                if is_close_comparable_type(ind_trigger.type):
                    trigger = Condition(
                        left_operand="close", operator=">", right_operand=ref_trigger
                    )
                else:
                    thresh = default_threshold_for(ind_trigger.type, bullish=True)
                    trigger = Condition(
                        left_operand=ref_trigger, operator="<", right_operand=thresh
                    )
                # フォロー条件の生成
                if is_close_comparable_type(ind_follow.type):
                    follow = Condition(
                        left_operand="close", operator=">", right_operand=ref_follow
                    )
                else:
                    thresh = default_threshold_for(ind_follow.type, bullish=False)
                    follow = Condition(
                        left_operand=ref_follow, operator=">", right_operand=thresh
                    )

                new_sc = StatefulCondition(
                    trigger_condition=trigger,
                    follow_condition=follow,
                    lookback_bars=random.randint(3, 10),
                    cooldown_bars=random.randint(0, 5),
                    direction=random.choice(["long", "short"]),
                    enabled=True,
                )
                mutated.stateful_conditions.append(new_sc)
            except Exception as e:
                logger.debug(f"ステートフル条件追加スキップ: {e}")


def mutate_risk_management_modes(
    mutated: StrategyGene, mutation_rate: float, config: GAConfig
) -> None:
    """リスク管理の「方式」を切り替える突然変異。

    TPSLGene / PositionSizingGene の汎用ENUM変異は二重ゲート
    （サブ遺伝子呼び出しゲート × 内部再選択ゲート）で実効確率が低く、
    さらに現在値と同じ方式の再選択（無効変異）が混ざるため、
    方式自体の探索がほぼ機能しない。本関数は確率が当たれば
    「必ず別方式へ」切り替える専用経路を提供する。
    """
    switch_probability = mutation_rate * 0.3

    # TPSL 遺伝子（共通 + 方向別）の method 切替
    for field_name in ("tpsl_gene", "long_tpsl_gene", "short_tpsl_gene"):
        tpsl_gene = getattr(mutated, field_name)
        if tpsl_gene is None or random.random() >= switch_probability:
            continue
        methods = [m for m in TPSLMethod if m != tpsl_gene.method]
        if methods:
            tpsl_gene.method = random.choice(methods)

    # トレーリングストップ / TP のトグル（共通 + 方向別 TPSL）
    for field_name in ("tpsl_gene", "long_tpsl_gene", "short_tpsl_gene"):
        tpsl_gene = getattr(mutated, field_name)
        if tpsl_gene is None:
            continue
        if random.random() < mutation_rate * 0.2:
            tpsl_gene.trailing_stop = not tpsl_gene.trailing_stop
        if random.random() < mutation_rate * 0.2:
            tpsl_gene.trailing_take_profit = not tpsl_gene.trailing_take_profit

    # ポジションサイジング遺伝子の method 切替
    sizing_gene = mutated.position_sizing_gene
    if sizing_gene is not None and random.random() < switch_probability:
        sizing_methods = [m for m in PositionSizingMethod if m != sizing_gene.method]
        if sizing_methods:
            sizing_gene.method = random.choice(sizing_methods)


def mutate_strategy_gene(
    gene: StrategyGene, config: GAConfig, mutation_rate: float = 0.1
) -> StrategyGene:
    """戦略遺伝子の「突然変異（Mutation）」を実行し、新しい個体を生成する。"""
    try:
        mutated = gene.clone()

        try:
            mutate_indicators(mutated, mutation_rate, config)
        except Exception as e:
            logger.warning(f"指標変異に失敗しました（スキップ）: {e}")

        try:
            mutate_conditions(mutated, mutation_rate, config)
        except Exception as e:
            logger.warning(f"条件変異に失敗しました（スキップ）: {e}")

        try:
            mutate_stateful_conditions(mutated, mutation_rate, config)
        except Exception as e:
            logger.warning(f"ステートフル条件変異に失敗しました（スキップ）: {e}")

        try:
            mutate_risk_management_modes(mutated, mutation_rate, config)
        except Exception as e:
            logger.warning(f"リスク管理方式変異に失敗しました（スキップ）: {e}")

        try:
            min_risk_multiplier, max_risk_multiplier = (
                config.mutation_config.risk_param_range
            )
            for key, value in mutated.risk_management.items():
                if isinstance(value, (int, float)) and random.random() < mutation_rate:
                    if key == "position_size":
                        mutated.risk_management[key] = max(
                            0.01,
                            min(
                                1.0,
                                value
                                * random.uniform(
                                    min_risk_multiplier,
                                    max_risk_multiplier,
                                ),
                            ),
                        )
                    else:
                        mutated.risk_management[key] = value * random.uniform(
                            min_risk_multiplier,
                            max_risk_multiplier,
                        )
        except Exception as e:
            logger.debug(f"リスクパラメータ変異スキップ: {e}")

        try:
            for (
                field_name,
                creator_func,
                creation_prob_mult,
            ) in _iter_mutable_sub_gene_specs(config):
                sub_gene = getattr(mutated, field_name)
                if sub_gene:
                    if random.random() < mutation_rate:
                        setattr(mutated, field_name, sub_gene.mutate(mutation_rate))
                elif random.random() < mutation_rate * creation_prob_mult:
                    setattr(
                        mutated,
                        field_name,
                        _create_sub_gene(creator_func, config),  # type: ignore[arg-type]
                    )
        except Exception as e:
            logger.debug(f"サブ遺伝子変異スキップ: {e}")

        try:
            from ...tools import tool_registry

            tool_add_prob = (
                config.mutation_config.tool_gene_add_delete_probability * mutation_rate
            )

            if not mutated.tool_genes:
                if random.random() < tool_add_prob:
                    available_tools = [t.name for t in tool_registry.get_all()]
                    if available_tools:
                        from ..tool import ToolGene

                        mutated.tool_genes.append(
                            ToolGene(
                                tool_name=random.choice(available_tools),
                                enabled=True,
                                params={},
                            )
                        )
                mutated.tool_genes = _ensure_weekend_filter(mutated.tool_genes)

            if mutated.tool_genes:
                for tool_gene in mutated.tool_genes:
                    if tool_gene.tool_name == WEEKEND_FILTER_NAME:
                        continue
                    if random.random() < mutation_rate:
                        if random.random() < 0.3:
                            tool_gene.enabled = not tool_gene.enabled

                        tool = tool_registry.get(tool_gene.tool_name)
                        if tool:
                            tool_gene.params = tool.mutate_params(tool_gene.params)

                # ツール削除（weekend_filter以外をランダムに除去）
                if (
                    len(mutated.tool_genes) > 1
                    and random.random() < tool_add_prob * 0.5
                ):
                    removable = [
                        t
                        for t in mutated.tool_genes
                        if t.tool_name != WEEKEND_FILTER_NAME
                    ]
                    if removable:
                        to_remove = random.choice(removable)
                        mutated.tool_genes.remove(to_remove)

                # ツール新規追加
                if (
                    len(mutated.tool_genes) < getattr(config, "max_enabled_filters", 3)
                    and random.random()
                    < config.mutation_config.tool_gene_add_delete_probability
                    * mutation_rate
                ):
                    existing_names = {t.tool_name for t in mutated.tool_genes}
                    candidates = [
                        t.name
                        for t in tool_registry.get_all()
                        if t.name not in existing_names
                    ]
                    if candidates:
                        from ..tool import ToolGene

                        mutated.tool_genes.append(
                            ToolGene(
                                tool_name=random.choice(candidates),
                                enabled=True,
                                params={},
                            )
                        )

                # フィルター数制限を強制
                from ...generators.random_gene_generator import (
                    RandomGeneGenerator,
                )

                mutated.tool_genes = RandomGeneGenerator.enforce_filter_limit(
                    config, mutated.tool_genes
                )
                mutated.tool_genes = _ensure_weekend_filter(mutated.tool_genes)
        except Exception as e:
            logger.debug(f"ToolGene変異スキップ: {e}")

        mutated.metadata["mutated"] = True
        mutated.metadata["mutation_rate"] = mutation_rate
        mutated.id = str(uuid.uuid4())

        # 非価格指標（OI/FR/LSR由来）の最低数を確保
        mutated.repair_non_price_indicators(config)

        # 独立に変異されたパラメータの依存関係（fast<slow等）と整数性を修復する
        mutated.repair_indicator_parameters()

        # 指標の削除・交換で条件参照が切れるため、整合性を修復する
        mutated.repair_condition_references()

        # 旧シード由来の価格×非価格スケール比較（恒真/恒偽）を閾値比較へ書き直す
        mutated.repair_condition_scales()

        # 参照除去で空になったエントリー条件を修復（無効個体の無駄打ちを削減）
        mutated.ensure_min_entry_conditions(
            min_conditions=getattr(config, "min_conditions", 1) or 1
        )

        return mutated

    except Exception as e:
        logger.error(f"戦略遺伝子突然変異エラー: {e}")
        try:
            mutated = gene.clone()
        except AttributeError:
            mutated = gene
        try:
            del mutated.fitness.values  # type: ignore[attr-defined]
        except AttributeError:
            pass
        return mutated


def adaptive_mutate_strategy_gene(
    gene: StrategyGene,
    population: list[Any],
    config: GAConfig,
    base_mutation_rate: float = 0.1,
) -> StrategyGene:
    """集団分散に基づく適応的突然変異。"""
    try:
        fitnesses = []
        for ind in population:
            try:
                if ind.fitness and ind.fitness.values:
                    fitnesses.append(ind.fitness.values[0])
            except AttributeError:
                pass

        # ±inf（制約違反ペナルティ）の fitness は分散計算の対象外とする。
        # そのまま np.var に渡すと RuntimeWarning（invalid value encountered
        # in subtract）を引き起こすため除外する。
        finite_fitnesses = [f for f in fitnesses if np.isfinite(f)]
        if not finite_fitnesses:
            adaptive_rate = base_mutation_rate
        else:
            variance = float(np.var(finite_fitnesses))
            variance_threshold = config.mutation_config.adaptive_variance_threshold

            if variance > variance_threshold:
                adaptive_rate = (
                    base_mutation_rate
                    * config.mutation_config.adaptive_decrease_multiplier
                )
            else:
                adaptive_rate = (
                    base_mutation_rate
                    * config.mutation_config.adaptive_increase_multiplier
                )

            adaptive_rate = max(0.01, min(1.0, adaptive_rate))

        mutated = mutate_strategy_gene(gene, config, mutation_rate=adaptive_rate)
        mutated.metadata["adaptive_mutation_rate"] = adaptive_rate
        return mutated

    except Exception as e:
        logger.error(f"適応的戦略遺伝子突然変異エラー: {e}")
        return mutate_strategy_gene(gene, config, mutation_rate=base_mutation_rate)
