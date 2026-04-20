"""
フィットネス共有の特徴ベクトル化ユーティリティ
"""

from typing import Any, Mapping, Sequence

import numpy as np

from app.services.auto_strategy.genes import ConditionGroup, StrategyGene

# 定数
DEFAULT_POSITION_SIZE = 0.1
DEFAULT_STOP_LOSS_PCT = 0.05
DEFAULT_TAKE_PROFIT_PCT = 0.1
DEFAULT_RISK_PER_TRADE = 0.01
DEFAULT_PERIOD_MEAN = 0.0
DEFAULT_PERIOD_MAX = 0.0
SAFE_FLOAT_FALLBACK = 0.0
EPSILON = 1e-6

BEHAVIOR_FEATURE_NAMES: tuple[str, ...] = (
    "objective_count",
    "fitness_primary",
    "fitness_mean",
    "fitness_std",
    "pass_rate",
    "scenario_count",
    "aggregated_primary",
    "worst_case_primary",
    "mean_total_return",
    "std_total_return",
    "mean_sharpe_ratio",
    "std_sharpe_ratio",
    "mean_max_drawdown",
    "std_max_drawdown",
    "mean_total_trades",
    "std_total_trades",
)


def build_behavior_profile(
    fitness_values: Sequence[float] | None = None,
    evaluation_report: object | None = None,
) -> dict[str, float]:
    """
    評価済み個体の挙動を固定長の数値特徴へ要約する。

    フィットネス値のみ利用可能な場合はその統計量を、
    EvaluationReport が利用可能な場合は pass_rate やシナリオ別の
    パフォーマンス安定性も取り込む。
    """
    profile = {name: 0.0 for name in BEHAVIOR_FEATURE_NAMES}

    sanitized_fitness = _sanitize_numeric_sequence(fitness_values)
    if sanitized_fitness:
        profile["objective_count"] = float(len(sanitized_fitness))
        profile["fitness_primary"] = float(sanitized_fitness[0])
        profile["fitness_mean"] = float(np.mean(sanitized_fitness))
        profile["fitness_std"] = float(np.std(sanitized_fitness))
        profile["aggregated_primary"] = float(sanitized_fitness[0])
        profile["worst_case_primary"] = float(sanitized_fitness[0])
        profile["scenario_count"] = 1.0
        profile["pass_rate"] = 1.0

    if evaluation_report is None:
        return profile

    if isinstance(evaluation_report, Mapping) and any(
        name in evaluation_report for name in BEHAVIOR_FEATURE_NAMES
    ):
        for name in BEHAVIOR_FEATURE_NAMES:
            if name in evaluation_report:
                profile[name] = _safe_float(
                    evaluation_report.get(name),
                    fallback=profile[name],
                )
        return profile

    pass_rate = _safe_float(getattr(evaluation_report, "pass_rate", 0.0))
    scenario_list = list(getattr(evaluation_report, "scenarios", []) or [])
    scenario_count = float(len(scenario_list))
    aggregated_primary = _safe_float(
        getattr(evaluation_report, "primary_aggregated_fitness", None),
        fallback=profile["aggregated_primary"],
    )
    worst_case_primary = _safe_float(
        getattr(evaluation_report, "primary_worst_case_fitness", None),
        fallback=aggregated_primary,
    )

    profile["pass_rate"] = pass_rate
    profile["scenario_count"] = scenario_count
    profile["aggregated_primary"] = aggregated_primary
    profile["worst_case_primary"] = worst_case_primary

    metric_names = (
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "total_trades",
    )
    for metric_name in metric_names:
        metric_values = _extract_scenario_metric_values(scenario_list, metric_name)
        if not metric_values:
            continue

        metric_prefix = f"mean_{metric_name}"
        std_prefix = f"std_{metric_name}"
        if metric_prefix in profile:
            profile[metric_prefix] = float(np.mean(metric_values))
        if std_prefix in profile:
            profile[std_prefix] = float(np.std(metric_values))

    return profile


def behavior_profile_to_vector(
    behavior_profile: Mapping[str, float] | None,
) -> list[float]:
    """behavior profile を固定順のベクトルへ変換する。"""
    if not behavior_profile:
        return [0.0] * len(BEHAVIOR_FEATURE_NAMES)
    return [float(behavior_profile.get(name, 0.0)) for name in BEHAVIOR_FEATURE_NAMES]


def vectorize_gene(
    gene: StrategyGene,
    indicator_types: Sequence[str],
    indicator_map: Mapping[str, int],
    operator_types: Sequence[str],
    operator_map: Mapping[str, int],
    behavior_profile: Mapping[str, float] | None = None,
) -> np.ndarray:
    """
    戦略遺伝子を固定長の特徴ベクトルに変換する。
    """
    features: list[float] = []

    # 指標数
    features.append(float(len(gene.indicators)))

    # 条件数
    features.append(float(len(gene.long_entry_conditions)))
    features.append(float(len(gene.short_entry_conditions)))

    # リスク管理パラメータ
    if gene.risk_management:
        features.append(float(gene.risk_management.get("position_size", DEFAULT_POSITION_SIZE)))
    else:
        features.append(DEFAULT_POSITION_SIZE)

    # TP/SLパラメータ
    if gene.tpsl_gene:
        features.append(float(gene.tpsl_gene.stop_loss_pct or DEFAULT_STOP_LOSS_PCT))
        features.append(float(gene.tpsl_gene.take_profit_pct or DEFAULT_TAKE_PROFIT_PCT))
    else:
        features.append(DEFAULT_STOP_LOSS_PCT)
        features.append(DEFAULT_TAKE_PROFIT_PCT)

    # ポジションサイジングパラメータ
    if gene.position_sizing_gene and hasattr(
        gene.position_sizing_gene, "risk_per_trade"
    ):
        features.append(float(gene.position_sizing_gene.risk_per_trade or DEFAULT_RISK_PER_TRADE))
    else:
        features.append(DEFAULT_RISK_PER_TRADE)

    # 指標タイプベクトル（Bag of Words）
    if indicator_types:
        indicator_vector = np.zeros(len(indicator_types))
        for ind in gene.indicators:
            if ind.type in indicator_map:
                idx = indicator_map[ind.type]
                indicator_vector[idx] += 1.0

        features.extend(indicator_vector.tolist())

    # オペレータタイプベクトル（Bag of Words）
    all_conditions = _collect_conditions(gene)
    if operator_types:
        operator_vector = np.zeros(len(operator_types))

        _count_operators(all_conditions, operator_map, operator_vector)

        features.extend(operator_vector.tolist())

    # 時間軸特性（指標パラメータから推定）
    period_values = []
    period_keys = [
        "period",
        "fast_period",
        "slow_period",
        "signal_period",
        "timeperiod",
        "k_period",
        "d_period",
    ]

    for ind in gene.indicators:
        parameters = getattr(ind, "parameters", {}) or {}
        for key in period_keys:
            if key in parameters and isinstance(parameters[key], (int, float)):
                period_values.append(float(parameters[key]))

    if period_values:
        features.append(float(np.mean(period_values)))
        features.append(float(np.max(period_values)))
    else:
        features.append(DEFAULT_PERIOD_MEAN)
        features.append(DEFAULT_PERIOD_MAX)

    # オペランド特性（定数比較 vs 動的比較）
    numeric_operands, dynamic_operands = _count_operand_types(all_conditions)

    features.append(numeric_operands)
    features.append(dynamic_operands)
    features.extend(behavior_profile_to_vector(behavior_profile))

    return np.array(features)


def _safe_float(value: object, fallback: float = SAFE_FLOAT_FALLBACK) -> float:
    """非数値や NaN/Inf を安全に float へ落とす。"""
    if value is None or not isinstance(value, (int, float)):
        return fallback
    value_float = float(value)
    if not np.isfinite(value_float):
        return fallback
    return value_float


def _sanitize_numeric_sequence(values: Sequence[float] | None) -> list[float]:
    """数値列から有限な float のみを抽出する。"""
    if not values:
        return []
    sanitized: list[float] = []
    for value in values:
        if not isinstance(value, (int, float)):
            continue
        value_float = float(value)
        if np.isfinite(value_float):
            sanitized.append(value_float)
    return sanitized


def _extract_scenario_metric_values(
    scenarios: Sequence[Any],
    metric_name: str,
) -> list[float]:
    """ScenarioEvaluation 群から metric を抽出する。"""
    metric_values: list[float] = []
    for scenario in scenarios:
        metrics = getattr(scenario, "performance_metrics", None)
        if not isinstance(metrics, dict):
            continue
        value = _safe_float(metrics.get(metric_name, None), fallback=np.nan)
        if np.isfinite(value):
            metric_values.append(value)
    return metric_values


def _collect_conditions(gene: StrategyGene) -> list[Any]:
    """全条件を1つのリストにまとめる。"""
    all_conditions: list[Any] = []
    if gene.long_entry_conditions:
        all_conditions.extend(gene.long_entry_conditions)
    if gene.short_entry_conditions:
        all_conditions.extend(gene.short_entry_conditions)
    return all_conditions


def _count_operators(
    conditions: list[Any],
    operator_map: Mapping[str, int],
    vector: np.ndarray,
) -> None:
    """条件リスト内のオペレータを再帰的にカウントする。"""
    for cond in conditions:
        if isinstance(cond, ConditionGroup):
            if cond.operator and cond.operator in operator_map:
                idx = operator_map[cond.operator]
                vector[idx] += 1.0

            if cond.conditions:
                _count_operators(cond.conditions, operator_map, vector)
        elif hasattr(cond, "operator"):
            op = cond.operator
            if op in operator_map:
                idx = operator_map[op]
                vector[idx] += 1.0


def _count_operand_types(conditions: list[Any]) -> tuple[float, float]:
    """
    オペランドのタイプ（数値/動的）をカウントする。
    """
    numeric = 0.0
    dynamic = 0.0

    for cond in conditions:
        if isinstance(cond, ConditionGroup):
            if cond.conditions:
                n, d = _count_operand_types(cond.conditions)
                numeric += n
                dynamic += d
        elif hasattr(cond, "right_operand"):
            op_val = cond.right_operand

            is_numeric = False
            if isinstance(op_val, (int, float)):
                is_numeric = True
            elif isinstance(op_val, str):
                try:
                    float(op_val)
                    is_numeric = True
                except ValueError:
                    is_numeric = False

            if is_numeric:
                numeric += 1.0
            else:
                dynamic += 1.0

    return numeric, dynamic
