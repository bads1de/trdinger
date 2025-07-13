/**
 * ベイジアン最適化関連の型定義
 */

export interface ParameterSpace {
  type: "real" | "integer" | "categorical";
  low?: number;
  high?: number;
  categories?: any[];
}

export interface BayesianOptimizationConfig {
  optimization_type: "ga" | "ml";
  experiment_name?: string;
  model_type?: string;
  base_config?: any;
  parameter_space?: Record<string, ParameterSpace>;
  n_calls: number;
  optimization_config?: {
    acq_func?: string;
    n_initial_points?: number;
    random_state?: number;
  };
}

export interface OptimizationHistory {
  iteration: number;
  params: Record<string, any>;
  score: number;
}

export interface BayesianOptimizationResult {
  optimization_type: string;
  experiment_name?: string;
  model_type?: string;
  best_params: Record<string, any>;
  best_score: number;
  total_evaluations: number;
  optimization_time: number;
  convergence_info: {
    converged: boolean;
    best_iteration: number;
  };
  optimization_history: OptimizationHistory[];
}

export interface BayesianOptimizationResponse {
  success: boolean;
  result?: BayesianOptimizationResult;
  error?: string;
  message: string;
  timestamp: string;
}

export interface DefaultParameterSpaceResponse {
  success: boolean;
  parameter_space?: Record<string, ParameterSpace>;
  message: string;
}
