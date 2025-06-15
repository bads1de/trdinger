/**
 * 戦略ショーケース関連の型定義
 *
 */

/**
 * 戦略カテゴリ
 */
export type StrategyCategory =
  | "trend_following"
  | "mean_reversion"
  | "breakout"
  | "range_trading"
  | "momentum";

/**
 * リスクレベル
 */
export type RiskLevel = "low" | "medium" | "high";

/**
 * ソート項目
 */
export type SortBy =
  | "expected_return"
  | "sharpe_ratio"
  | "max_drawdown"
  | "win_rate"
  | "created_at";

/**
 * ソート順序
 */
export type SortOrder = "asc" | "desc";

/**
 * ショーケース戦略データ
 */
export interface StrategyShowcase {
  id: number;
  name: string;
  description: string;
  category: StrategyCategory;
  indicators: string[];
  parameters: Record<string, any>;
  expected_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  gene_data: Record<string, any>;
  backtest_result_id?: number;
  risk_level: RiskLevel;
  recommended_timeframe: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

/**
 * 戦略フィルター
 */
export interface StrategyFilter {
  category?: StrategyCategory;
  risk_level?: RiskLevel;
  min_return?: number;
  max_return?: number;
  min_sharpe?: number;
  max_sharpe?: number;
  min_drawdown?: number;
  max_drawdown?: number;
  search_query?: string;
}

/**
 * ページネーション
 */
export interface Pagination {
  limit?: number;
  offset: number;
  sort_by: SortBy;
  sort_order: SortOrder;
}

/**
 * 戦略一覧レスポンス
 */
export interface StrategyListResponse {
  success: boolean;
  strategies: StrategyShowcase[];
  total_count: number;
  message: string;
}

/**
 * 戦略詳細レスポンス
 */
export interface StrategyDetailResponse {
  success: boolean;
  strategy?: StrategyShowcase;
  message: string;
}

/**
 * ショーケース統計
 */
export interface ShowcaseStatistics {
  total_strategies: number;
  avg_return: number;
  avg_sharpe_ratio: number;
  avg_max_drawdown: number;
  category_distribution: Record<StrategyCategory, number>;
  risk_distribution: Record<RiskLevel, number>;
}

/**
 * 統計レスポンス
 */
export interface ShowcaseStatsResponse {
  success: boolean;
  statistics: ShowcaseStatistics;
  message: string;
}

/**
 * 戦略生成リクエスト
 */
export interface GenerateShowcaseRequest {
  count: number;
  base_config?: {
    symbol: string;
    timeframe: string;
    start_date: string;
    end_date: string;
    initial_capital: number;
    commission_rate: number;
  };
}

/**
 * 戦略生成レスポンス
 */
export interface GenerateShowcaseResponse {
  success: boolean;
  message: string;
  generated_count: number;
  saved_count: number;
}

/**
 * カテゴリ一覧レスポンス
 */
export interface CategoriesResponse {
  success: boolean;
  categories: Record<StrategyCategory, string>;
  message: string;
}

/**
 * リスクレベル一覧レスポンス
 */
export interface RiskLevelsResponse {
  success: boolean;
  risk_levels: Record<RiskLevel, string>;
  message: string;
}

/**
 * 戦略カードプロパティ
 */
export interface StrategyCardProps {
  strategy: StrategyShowcase;
  onViewDetail: (strategy: StrategyShowcase) => void;
  className?: string;
}

/**
 * 戦略フィルタープロパティ
 */
export interface StrategyFiltersProps {
  filter: StrategyFilter;
  pagination: Pagination;
  onFilterChange: (filter: StrategyFilter) => void;
  onPaginationChange: (pagination: Pagination) => void;
  categories: Record<StrategyCategory, string>;
  riskLevels: Record<RiskLevel, string>;
}

/**
 * 戦略モーダルプロパティ
 */
export interface StrategyModalProps {
  strategy?: StrategyShowcase;
  isOpen: boolean;
  onClose: () => void;
}

/**
 * パフォーマンス指標の色分け
 */
export interface PerformanceColors {
  return: "positive" | "negative" | "neutral";
  sharpe: "excellent" | "good" | "fair" | "poor";
  drawdown: "low" | "medium" | "high";
}

/**
 * 戦略統計サマリー
 */
export interface StrategySummary {
  best_return: StrategyShowcase;
  best_sharpe: StrategyShowcase;
  lowest_drawdown: StrategyShowcase;
  highest_win_rate: StrategyShowcase;
}

/**
 * API エラーレスポンス
 */
export interface ApiErrorResponse {
  success: false;
  message: string;
  detail?: string;
}

/**
 * ローディング状態
 */
export interface LoadingState {
  strategies: boolean;
  statistics: boolean;
  generation: boolean;
  detail: boolean;
}

/**
 * エラー状態
 */
export interface ErrorState {
  strategies?: string;
  statistics?: string;
  generation?: string;
  detail?: string;
}
