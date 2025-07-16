/**
 * バックテスト関連の型定義
 */

/**
 * 最適化結果の詳細な型定義
 */
export interface OptimizationResult {
  strategy_name: string;
  symbol: string;
  timeframe: string;
  initial_capital: number;
  performance_metrics: {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    profit_factor: number;
    total_trades: number;
  };
  optimized_parameters?: Record<string, any>;
  heatmap_summary?: {
    best_combination: any;
    best_value: number;
    worst_combination: any;
    worst_value: number;
    mean_value: number;
    std_value: number;
    total_combinations: number;
  };
  optimization_details?: {
    method: string;
    n_calls: number;
    best_value: number;
    convergence?: {
      initial_value: number;
      final_value: number;
      improvement: number;
      convergence_rate: number;
      plateau_detection: boolean;
    };
  };
  optimization_metadata?: {
    method: string;
    maximize: string;
    parameter_space_size: number;
    optimization_timestamp: string;
  };
  multi_objective_details?: {
    objectives: string[];
    weights: number[];
    individual_scores: Record<string, number>;
  };
  robustness_analysis?: {
    robustness_score: number;
    successful_periods: number;
    failed_periods: number;
    performance_statistics: Record<
      string,
      {
        mean: number;
        std: number;
        min: number;
        max: number;
        consistency_score: number;
      }
    >;
    parameter_stability: Record<
      string,
      {
        mean: number;
        std: number;
        coefficient_of_variation: number;
      }
    >;
  };
  individual_results?: Record<string, any>;
  total_periods?: number;
}

/**
 * 最適化モーダルのプロパティ
 */
export interface OptimizationModalProps {
  /** モーダルの表示状態 */
  isOpen: boolean;
  /** モーダルを閉じる関数 */
  onClose: () => void;
  /** 拡張最適化実行時のコールバック */
  onEnhancedOptimization: (config: OptimizationConfig) => void;
  /** マルチ目的最適化実行時のコールバック */
  onMultiObjectiveOptimization: (config: MultiObjectiveConfig) => void;
  /** ロバストネステスト実行時のコールバック */
  onRobustnessTest: (config: RobustnessConfig) => void;
  /** GA戦略生成実行時のコールバック */
  onGAGeneration?: (config: GAConfig) => void;
  /** 最適化実行中かどうか */
  isLoading?: boolean;
  /** 選択されたバックテスト結果（設定を引き継ぐため） */
  selectedResult?: BacktestResult | null;
  /** 現在のバックテスト設定（基本設定を引き継ぐため） */
  currentBacktestConfig?: BacktestConfig | null;
}

/**
 * 拡張最適化の設定
 */
export interface OptimizationConfig {
  base_config: BacktestConfig;
  optimization_params: {
    method: "grid" | "sambo";
    max_tries?: number;
    maximize: string;
    return_optimization: boolean;
    random_state: number;
    constraint: string;
    parameters: Record<string, [number, number, number]>;
  };
}

/**
 * マルチ目的最適化の設定
 */
export interface MultiObjectiveConfig {
  base_config: BacktestConfig;
  optimization_params: {
    objectives: string[];
    weights: number[];
    method: "grid" | "sambo";
    max_tries?: number;
    parameters: Record<string, [number, number, number]>;
  };
}

/**
 * ロバストネステストの設定
 */
export interface RobustnessConfig {
  base_config: BacktestConfig;
  test_periods: [string, string][];
  optimization_params: {
    method: "grid" | "sambo";
    max_tries?: number;
    maximize: string;
    parameters: Record<string, [number, number, number]>;
  };
}

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
/**
 * バックテストの設定
 */
export interface BacktestConfig {
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
}

/**
 * バックテストの結果
 */
export interface BacktestResult {
  id?: string;
  strategy_name: string;
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  commission_rate: number;
  config_json: any;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  winning_trades?: number;
  losing_trades?: number;
  avg_win?: number;
  avg_loss?: number;
  equity_curve: EquityPoint[];
  trade_history: Trade[];
  execution_time?: number;
  status?: string;
  error_message?: string;
  created_at: Date | string;
  updated_at?: Date;
  performance_metrics?: any; // 互換性のため
}

/**
 * 損益曲線のポイント
 */
export interface EquityPoint {
  timestamp: string;
  equity: number;
  drawdown: number;
}

/**
 * 取引履歴
 */
export interface Trade {
  id: string;
  timestamp: string;
  type: "buy" | "sell";
  price: number;
  quantity: number;
  commission: number;
  pnl?: number;
}
