/**
 * オートストラテジー生成戦略の型定義
 *
 * GET /api/strategies のレスポンス形式に対応します。
 */

export interface GeneratedStrategy {
  id: string;
  name: string;
  description: string;
  category: string;
  indicators: string[];
  parameters: Record<string, any>;
  expected_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  risk_level: string;
  recommended_timeframe: string;
  source: string;
  experiment_id: number;
  generation: number;
  fitness_score: number | null;
  evaluation_summary: Record<string, any> | null;
  validation_summary: Record<string, any> | null;
  validation_passed: boolean | null;
  robustness_pass_rate: number | null;
  selection_rank: number | null;
  created_at: string | null;
  updated_at: string | null;
}
