/**
 * バックテスト関連の型定義
 */

import { BacktestConfig } from "./backtest";

/**
 * GA戦略生成の設定
 */
export interface GAConfig {
  experiment_name: string;
  base_config: BacktestConfig;
  ga_config: {
    population_size: number;
    generations: number;
    mutation_rate: number;
    crossover_rate: number;
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

    // 指標モード設定
    indicator_mode: "technical_only" | "ml_only" | "mixed";

    // 高度な設定
    enable_fitness_sharing?: boolean;
    sharing_radius?: number;

    // 多目的最適化設定
    enable_multi_objective?: boolean;
    objectives?: string[];
    objective_weights?: number[];
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

// 多目的最適化結果
export interface MultiObjectiveGAResult {
  success: boolean;
  result: {
    best_strategy: {
      gene_data: Record<string, any>;
      fitness_score: number;
      fitness_values: number[];
    };
    pareto_front: Array<{
      strategy: Record<string, any>;
      fitness_values: number[];
    }>;
    objectives: string[];
    total_strategies: number;
    execution_time: number;
  };
  message: string;
}

// パレート最適解
export interface ParetoSolution {
  strategy: Record<string, any>;
  fitness_values: number[];
  objectives: string[];
}

// 多目的最適化の目的関数定義
export interface ObjectiveDefinition {
  name: string;
  display_name: string;
  description: string;
  type: "maximize" | "minimize";
  weight: number;
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
    strategy_config: {
      strategy_type: string;
      parameters: Record<string, any>;
    };
  };
}


