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
  // プロファイル保存用パラメータ
  save_as_profile?: boolean;
  profile_name?: string;
  profile_description?: string;
}

export interface OptimizationHistory {
  iteration: number;
  params: Record<string, any>;
  score: number;
}

export interface BayesianOptimizationResponse {
  success: boolean;
  result?: any;
  error?: string;
  message: string;
  timestamp: string;
}

export interface DefaultParameterSpaceResponse {
  success: boolean;
  parameter_space?: Record<string, ParameterSpace>;
  message: string;
}

export interface MLOptimizationRequest {
  model_type: string;
  parameter_space?: Record<string, ParameterSpace>;
  n_calls?: number;
  optimization_config?: Record<string, any>;
  save_as_profile?: boolean;
  profile_name?: string;
  profile_description?: string;
}
