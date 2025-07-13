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

// プロファイル関連の型定義
export interface OptimizationProfile {
  id: number;
  profile_name: string;
  description?: string;
  optimization_result_id: number;
  is_default: boolean;
  is_active: boolean;
  target_model_type?: string;
  created_at: string;
  updated_at: string;
  optimization_result?: BayesianOptimizationResult;
}

export interface ProfileCreateRequest {
  name: string;
  optimization_result_id: number;
  description?: string;
  is_default?: boolean;
  target_model_type?: string;
}

export interface ProfileUpdateRequest {
  name?: string;
  description?: string;
  is_default?: boolean;
  is_active?: boolean;
  target_model_type?: string;
}

export interface ProfileResponse {
  success: boolean;
  profile?: OptimizationProfile;
  profiles?: OptimizationProfile[];
  count?: number;
  message: string;
  timestamp: string;
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
