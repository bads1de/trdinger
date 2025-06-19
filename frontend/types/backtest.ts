/**
 * バックテスト関連の統一型定義
 *
 * 全てのバックテスト関連コンポーネントで使用する統一された型定義
 */

/**
 * 資産曲線の各ポイント
 */
export interface EquityPoint {
  /** タイムスタンプ（ISO文字列） */
  timestamp: string;
  /** 資産額 */
  equity: number;
  /** ドローダウン率（0-1の範囲、計算で追加される） */
  drawdown_pct?: number;
}

/**
 * 取引履歴の各取引
 */
export interface Trade {
  /** 取引サイズ（正=ロング、負=ショート） */
  size: number;
  /** エントリー価格 */
  entry_price: number;
  /** イグジット価格 */
  exit_price: number;
  /** 損益（PnL） */
  pnl: number;
  /** リターン率 */
  return_pct: number;
  /** エントリー時刻 */
  entry_time: string;
  /** イグジット時刻 */
  exit_time: string;
}

/**
 * パフォーマンス指標
 */
export interface PerformanceMetrics {
  /** 総リターン */
  total_return: number | null;
  /** シャープレシオ */
  sharpe_ratio: number | null;
  /** 最大ドローダウン */
  max_drawdown: number | null;
  /** 勝率 */
  win_rate: number | null;
  /** プロフィットファクター */
  profit_factor: number | null;
  /** 総取引数 */
  total_trades: number | null;
  /** 勝ち取引数 */
  winning_trades: number | null;
  /** 負け取引数 */
  losing_trades: number | null;
  /** 平均利益 */
  avg_win: number | null;
  /** 平均損失 */
  avg_loss: number | null;
  /** 最終資産額（計算で追加される） */
  equity_final?: number | null;
  /** Buy & Hold リターン（オプション） */
  buy_hold_return?: number | null;
  /** エクスポージャー時間（オプション） */
  exposure_time?: number | null;
  /** ソルティーノレシオ（オプション） */
  sortino_ratio?: number | null;
  /** カルマーレシオ（オプション） */
  calmar_ratio?: number | null;
}

/**
 * バックテスト設定
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
    parameters: Record<string, number>;
  };
}

/**
 * バックテスト結果（統一版）
 */
export interface BacktestResult {
  /** 結果ID */
  id: string;
  /** 戦略名 */
  strategy_name: string;
  /** 取引ペア */
  symbol: string;
  /** 時間軸 */
  timeframe: string;
  /** 開始日 */
  start_date: string;
  /** 終了日 */
  end_date: string;
  /** 初期資金 */
  initial_capital: number;
  /** 手数料率 */
  commission_rate: number;
  /** 戦略設定（JSON形式） */
  config_json?: Record<string, any>;
  /** パフォーマンス指標 */
  performance_metrics: PerformanceMetrics;
  /** 資産曲線データ（オプション） */
  equity_curve?: Array<{
    timestamp: string;
    equity: number;
  }>;
  /** 取引履歴（オプション） */
  trade_history?: Array<{
    size: number;
    entry_price: number;
    exit_price: number;
    pnl: number;
    return_pct: number;
    entry_time: string;
    exit_time: string;
  }>;
  /** 作成日時 */
  created_at: string;
}

/**
 * チャート表示用の変換されたデータ型
 */
export interface ChartEquityPoint {
  /** 日付（数値タイムスタンプ） */
  date: number;
  /** 資産額 */
  equity: number;
  /** ドローダウン率（パーセンテージ） */
  drawdown: number;
  /** フォーマットされた日付文字列 */
  formattedDate: string;
  /** Buy & Hold 資産額（オプション） */
  buyHold?: number;
}

/**
 * チャート表示用の取引データ
 */
export interface ChartTradePoint {
  /** エントリー日（数値タイムスタンプ） */
  entryDate: number;
  /** イグジット日（数値タイムスタンプ） */
  exitDate: number;
  /** 損益 */
  pnl: number;
  /** リターン率（パーセンテージ） */
  returnPct: number;
  /** 取引サイズ（絶対値） */
  size: number;
  /** 取引タイプ */
  type: "long" | "short";
  /** 勝敗 */
  isWin: boolean;
}

/**
 * 月次リターンデータ
 */
export interface MonthlyReturn {
  /** 年 */
  year: number;
  /** 月（1-12） */
  month: number;
  /** 月次リターン率 */
  return: number;
  /** 月名 */
  monthName: string;
}

/**
 * リターン分布データ
 */
export interface ReturnDistribution {
  /** リターン範囲の下限 */
  rangeStart: number;
  /** リターン範囲の上限 */
  rangeEnd: number;
  /** 該当する取引数 */
  count: number;
  /** 頻度（パーセンテージ） */
  frequency: number;
}

/**
 * チャート共通プロパティ
 */
export interface BaseChartProps {
  /** データ配列 */
  data: any[];
  /** ローディング状態 */
  loading?: boolean;
  /** エラーメッセージ */
  error?: string;
  /** チャートの高さ */
  height?: number;
  /** 追加のCSSクラス */
  className?: string;
  /** テーマ */
  theme?: "light" | "dark";
}

/**
 * チャートコンテナのプロパティ
 */
export interface ChartContainerProps extends BaseChartProps {
  /** チャートタイトル */
  title: string;
  /** サブタイトル（オプション） */
  subtitle?: string;
  /** 子要素 */
  children: React.ReactNode;
  /** アクションボタン（オプション） */
  actions?: React.ReactNode;
}
