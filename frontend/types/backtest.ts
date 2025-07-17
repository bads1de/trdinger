/**
 * バックテスト関連の型定義
 */

/**
 * 資産曲線の単一データポイント
 */
export interface EquityPoint {
  timestamp: string;
  equity: number;
  drawdown_pct?: number;
}

/**
 * 取引履歴の単一データ
 * backtesting.pyの出力に合わせて定義
 */
export interface Trade {
  size: number;
  entry_price: number;
  exit_price: number;
  pnl: number;
  return_pct: number;
  entry_time: string;
  exit_time: string;
}

/**
 * チャート表示用の資産曲線データポイント
 */
export interface ChartEquityPoint {
  date: number;
  equity: number;
  drawdown: number;
  formattedDate: string;
  buyHold?: number;
}

/**
 * チャート表示用の取引データポイント
 */
export interface ChartTradePoint {
  entryDate: number;
  exitDate: number;
  pnl: number;
  returnPct: number;
  size: number;
  type: "long" | "short";
  isWin: boolean;
}

/**
 * リターン分布のデータ
 */
export interface ReturnDistribution {
  rangeStart: number;
  rangeEnd: number;
  count: number;
  frequency: number;
}

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
 * チャートコンテナのProps
 */
export interface ChartContainerProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  actions?: React.ReactNode;
  data: any[] | null | undefined;
  loading?: boolean;
  error?: string | null;
  height?: number;
  className?: string;
  theme?: "dark" | "light";
}
