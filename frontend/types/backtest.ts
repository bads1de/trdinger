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
