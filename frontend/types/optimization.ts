export interface OptimizationConfig {
  base_config: {
    strategy_name: string;
    symbol: string;
    timeframe: string;
    start_date: string;
    end_date: string;
    initial_capital: number;
    commission_rate: number;
    strategy_config: {
      strategy_type: string;
      parameters: Record<string, number>;
    };
  };
  optimization_params: {
    method: "grid" | "sambo";
    max_tries?: number;
    maximize: string;
    return_optimization?: boolean;
    random_state?: number;
    constraint?: string;
    parameters: Record<string, number[]>;
  };
}

export interface MultiObjectiveConfig {
  base_config: {
    strategy_name: string;
    symbol: string;
    timeframe: string;
    start_date: string;
    end_date: string;
    initial_capital: number;
    commission_rate: number;
    strategy_config: {
      strategy_type: string;
      parameters: Record<string, number>;
    };
  };
  optimization_params: {
    objectives: string[];
    weights: number[];
    method: "grid" | "sambo";
    max_tries?: number;
    parameters: Record<string, number[]>;
  };
}

export interface RobustnessConfig {
  base_config: {
    strategy_name: string;
    symbol: string;
    timeframe: string;
    initial_capital: number;
    commission_rate: number;
    strategy_config: {
      strategy_type: string;
      parameters: Record<string, number>;
    };
  };
  test_periods: string[][];
  optimization_params: {
    method: "grid" | "sambo";
    max_tries?: number;
    maximize: string;
    parameters: Record<string, number[]>;
  };
}

export interface GAConfig {
  experiment_name: string;
  base_config: {
    strategy_name: string;
    symbol: string;
    timeframe: string;
    start_date: string;
    end_date: string;
    initial_capital: number;
    commission_rate: number;
    strategy_config: {
      strategy_type: string;
      parameters: Record<string, number>;
    };
  };
  ga_config: {
    population_size: number;
    generations: number;
    crossover_rate: number;
    mutation_rate: number;
    elite_size: number;
    max_indicators: number;
    allowed_indicators: string[];
    fitness_weights: {
      total_return: number;
      sharpe_ratio: number;
      max_drawdown: number;
      win_rate: number;
    };
    fitness_constraints: {
      min_trades: number;
      max_drawdown_limit: number;
      min_sharpe_ratio: number;
    };
    ga_objective: string;
    indicator_mode: "technical_only" | "ml_only" | "mixed";

    // 高度な設定
    enable_fitness_sharing?: boolean;
    sharing_radius?: number;
    sharing_alpha?: number;
    enable_short_bias_mutation?: boolean;
    short_bias_rate?: number;

    // 従来のリスク管理パラメータ（Position Sizingシステムにより廃止）
    stop_loss_range: [number, number];
    take_profit_range: [number, number];

    // 新しいTP/SL自動決定設定
    tpsl_strategy?: TPSLStrategy;
    max_risk_per_trade?: number;
    preferred_risk_reward_ratio?: number;
    volatility_sensitivity?: VolatilitySensitivity;
    enable_advanced_tpsl?: boolean;

    // 統計的TP/SL設定
    statistical_lookback_days?: number;
    statistical_min_samples?: number;
    statistical_confidence_threshold?: number;

    // ボラティリティベース設定
    atr_period?: number;
    atr_multiplier_sl?: number;
    atr_multiplier_tp?: number;
    adaptive_multiplier?: boolean;
  };
}

// TP/SL戦略の種類
export type TPSLStrategy =
  | "legacy"
  | "random"
  | "risk_reward"
  | "volatility_adaptive"
  | "statistical"
  | "auto_optimal";

// ボラティリティ感度
export type VolatilitySensitivity = "low" | "medium" | "high";

// TP/SL自動決定結果
export interface TPSLResult {
  stop_loss_pct: number;
  take_profit_pct: number;
  risk_reward_ratio: number;
  strategy_used: string;
  confidence_score: number;
  metadata?: Record<string, any>;
}

// TP/SL設定プリセット
export interface TPSLPreset {
  name: string;
  description: string;
  strategy: TPSLStrategy;
  max_risk_per_trade: number;
  preferred_risk_reward_ratio: number;
  volatility_sensitivity: VolatilitySensitivity;
}

// 簡素化されたGA設定（新しいUI用）
export interface SimplifiedGAConfig {
  experiment_name: string;
  base_config: {
    strategy_name: string;
    symbol: string;
    timeframe: string;
    start_date: string;
    end_date: string;
    initial_capital: number;
    commission_rate: number;
  };

  // 簡素化された設定
  tpsl_preset: string; // "conservative", "balanced", "aggressive", "custom"
  custom_tpsl_config?: {
    strategy: TPSLStrategy;
    max_risk_per_trade: number;
    preferred_risk_reward_ratio: number;
    volatility_sensitivity: VolatilitySensitivity;
  };

  // GA基本設定
  population_size: number;
  generations: number;
  max_indicators: number;
  allowed_indicators: string[];
}
