"""
GA固有定数

遺伝的アルゴリズムに関連する定数を定義します。
"""

# === GA基本設定 ===
GA_DEFAULT_CONFIG = {
    "population_size": 100,
    "generations": 50,
    "crossover_rate": 0.8,
    "mutation_rate": 0.1,
    "elite_size": 10,
    "max_indicators": 10,
    "zero_trades_penalty": 0.1,
    "constraint_violation_penalty": 0.0,  # 0.0に戻して制約違反が原因か確認
    "max_enabled_filters": 3,  # 最大有効フィルター数
}

# === フィットネス重み設定 ===
DEFAULT_FITNESS_WEIGHTS = {
    "total_return": 0.1,
    "excess_return": 0.1,
    "sharpe_ratio": 0.25,
    "max_drawdown": 0.15,
    "win_rate": 0.1,
    "balance_score": 0.1,
    "ulcer_index_penalty": 0.15,
    "trade_frequency_penalty": 0.05,
}

# === フィットネス制約設定 ===
DEFAULT_FITNESS_CONSTRAINTS = {
    "min_trades": 10,  # 最低保証回数（50→10に一時的に緩和）
    "max_drawdown_limit": 0.2,  # 20%以上のドローダウンでペナルティ（0.3→0.2に強化）
    "min_sharpe_ratio": 0.5,  # 最低シャープレシオ（1.0→0.5に変更）
}

# === GA目的設定 ===
DEFAULT_GA_OBJECTIVES = ["weighted_score"]
DEFAULT_GA_OBJECTIVE_WEIGHTS = [1.0]

# === GAフィットネス共有設定 ===
GA_DEFAULT_FITNESS_SHARING = {
    "enable_fitness_sharing": True,
    "sharing_radius": 0.1,
    "sharing_alpha": 1.0,
    "sampling_threshold": 200,
    "sampling_ratio": 0.3,
}

# === GAパラメータ範囲定義 ===
GA_PARAMETER_RANGES = {
    # 基本パラメータ
    "period": [5, 200],
    "fast_period": [5, 20],
    "slow_period": [20, 50],
    "signal_period": [5, 15],
    # 特殊パラメータ
    "std_dev": [1.5, 2.5],
    "k_period": [10, 20],
    "d_period": [3, 7],
    "slowing": [1, 5],
    # 閾値パラメータ
    "overbought": [70, 90],
    "oversold": [10, 30],
}

# === GA閾値範囲定義 ===
# 閾値の生成はインジケーター設定（IndicatorConfig.thresholds）から
# スケール型に応じて解決されるため、ここでの範囲テーブルは持たない。

# === GA突然変異設定 ===
GA_MUTATION_SETTINGS = {
    "indicator_param_mutation_range": (0.8, 1.2),
    "indicator_add_delete_probability": 0.3,
    "indicator_add_vs_delete_probability": 0.5,
    "crossover_field_selection_probability": 0.5,
    "condition_operator_switch_probability": 0.2,
    "condition_change_probability_multiplier": 1.0,
    "condition_selection_probability": 0.5,
    "risk_param_mutation_range": (0.9, 1.1),
    "tpsl_gene_creation_probability_multiplier": 0.2,
    "position_sizing_gene_creation_probability_multiplier": 0.2,
    "entry_gene_creation_probability_multiplier": 0.2,
    "exit_gene_creation_probability_multiplier": 0.2,
    "adaptive_mutation_variance_threshold": 0.001,
    "adaptive_mutation_rate_decrease_multiplier": 0.8,
    "adaptive_mutation_rate_increase_multiplier": 1.2,
}

# === GA TPSL関連定数 ===
# TPSLの生成範囲・検証範囲は genes/gene_ranges.py の
# TPSL_GENERATION_RANGES / TPSL_VALIDATION_RANGES を単一ソースとする。
