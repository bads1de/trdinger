// テクニカル指標の定義
export interface TechnicalIndicator {
  name: string;
  params: Record<string, number | string>;
}

// 売買条件の定義
export interface TradingCondition {
  condition: string;
  description?: string;
}

// 戦略の定義
export interface TradingStrategy {
  id?: string;
  strategy_name: string;
  target_pair: string;
  timeframe: string;
  indicators: TechnicalIndicator[];
  entry_rules: TradingCondition[];
  exit_rules: TradingCondition[];
  created_at?: Date;
  updated_at?: Date;
}

// バックテストの設定
export interface BacktestConfig {
  strategy: TradingStrategy;
  start_date: string;
  end_date: string;
  initial_capital: number;
  commission_rate?: number;
  slippage_rate?: number;
}

// バックテストの結果
export interface BacktestResult {
  id?: string;
  strategy_id: string;
  config: BacktestConfig;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  avg_win: number;
  avg_loss: number;
  equity_curve: EquityPoint[];
  trade_history: Trade[];
  created_at: Date;
}

// 損益曲線のポイント
export interface EquityPoint {
  timestamp: string;
  equity: number;
  drawdown: number;
}

// 取引履歴
export interface Trade {
  id: string;
  timestamp: string;
  type: 'buy' | 'sell';
  price: number;
  quantity: number;
  commission: number;
  pnl?: number;
}

// 価格データ
export interface PriceData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}
