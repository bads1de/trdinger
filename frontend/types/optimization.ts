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
  };
}
