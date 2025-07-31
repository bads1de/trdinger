/**
 * バックテスト関連の型定義
 */

/**
 * 資産曲線の単一データポイント
 */
export interface EquityPoint {
  /** ISO形式の時刻 */
  timestamp: string;
  /** 資産評価額 */
  equity: number;
  /** ドローダウン（0-1の割合。UIで%表示推奨） */
  drawdown_pct?: number;
}

/**
 * 取引履歴の単一データ
 * backtesting.pyの出力に合わせて定義
 */
export interface Trade {
  /** ポジションサイズ（単位やレバレッジは戦略設定に依存） */
  size: number;
  /** エントリー価格 */
  entry_price: number;
  /** エグジット価格 */
  exit_price: number;
  /** 損益（通貨建て） */
  pnl: number;
  /** 取引リターン（0-1の割合） */
  return_pct: number;
  /** エントリー時刻（ISO） */
  entry_time: string;
  /** エグジット時刻（ISO） */
  exit_time: string;
}

/**
 * チャート表示用の資産曲線データポイント
 */
export interface ChartEquityPoint {
  /** Unix epoch milliseconds */
  date: number;
  /** 資産評価額 */
  equity: number;
  /** ドローダウン（0-1の割合） */
  drawdown: number;
  /** フォーマット済み日時文字列 */
  formattedDate: string;
  /** 買い持ち指数曲線（比較用、任意） */
  buyHold?: number;
}

/**
 * チャート表示用の取引データポイント
 */
export interface ChartTradePoint {
  /** エントリー時刻（ms） */
  entryDate: number;
  /** エグジット時刻（ms） */
  exitDate: number;
  /** 損益（通貨建て） */
  pnl: number;
  /** 取引リターン（0-1の割合） */
  returnPct: number;
  /** ポジションサイズ */
  size: number;
  /** 建玉種別 */
  type: "long" | "short";
  /** 勝ち判定（pnl > 0） */
  isWin: boolean;
}

/**
 * リターン分布のデータ
 */
export interface ReturnDistribution {
  /** ビンの開始値（含む） */
  rangeStart: number;
  /** ビンの終了値（含まない想定） */
  rangeEnd: number;
  /** 件数 */
  count: number;
  /** 相対頻度（0-1） */
  frequency: number;
}

export interface BacktestResult {
  /** 任意のID（保存時に付与される場合あり） */
  id?: string;
  /** 戦略名 */
  strategy_name: string;
  /** シンボル */
  symbol: string;
  /** 時間軸 */
  timeframe: string;
  /** 期間開始 */
  start_date: string;
  /** 期間終了 */
  end_date: string;
  /** 初期資金 */
  initial_capital: number;
  /** 片道手数料率（0-1） */
  commission_rate: number;
  /** 実行時の設定（後方互換のため any） */
  config_json: any;
  /** 総リターン（0-1の割合） */
  total_return: number;
  /** シャープレシオ */
  sharpe_ratio: number;
  /** 最大ドローダウン（0-1） */
  max_drawdown: number;
  /** 勝率（0-1） */
  win_rate: number;
  /** プロフィットファクター */
  profit_factor: number;
  /** 総取引数 */
  total_trades: number;
  /** 勝ち数 */
  winning_trades?: number;
  /** 負け数 */
  losing_trades?: number;
  /** 平均勝ち額 */
  avg_win?: number;
  /** 平均負け額 */
  avg_loss?: number;
  /** 資産曲線 */
  equity_curve: EquityPoint[];
  /** 取引履歴 */
  trade_history: Trade[];
  /** 実行時間（秒） */
  execution_time?: number;
  /** 実行状態（例: "pending" | "running" | "completed" | "error"） */
  status?: string;
  /** エラーメッセージ（失敗時） */
  error_message?: string;
  /** 作成時刻 */
  created_at: Date | string;
  /** 更新時刻 */
  updated_at?: Date;
  /** 追加メトリクス（後方互換のため any） */
  performance_metrics?: any; // 互換性のため
}

/**
 * バックテストの設定
 */
export interface BacktestConfig {
  /** 戦略名 */
  strategy_name: string;
  /** シンボル */
  symbol: string;
  /** 時間軸 */
  timeframe: string;
  /** 期間開始 */
  start_date: string;
  /** 期間終了 */
  end_date: string;
  /** 初期資金 */
  initial_capital: number;
  /** 片道手数料率（0-1） */
  commission_rate: number;
  /** 戦略の型とパラメータ（型に応じて parameters のキーが変化） */
  strategy_config: {
    /** 戦略タイプ（例: "rule_based", "ml_based" など） */
    strategy_type: string;
    /** パラメータ辞書 */
    parameters: Record<string, any>;
  };
}

/**
 * チャートコンテナのProps
 */
export interface ChartContainerProps {
  /** 見出し */
  title: string;
  /** サブタイトル */
  subtitle?: string;
  /** チャイルド要素 */
  children: React.ReactNode;
  /** 右上操作ボタン群 */
  actions?: React.ReactNode;
  /** チャート/テーブルに与えるデータ（null/undefinedは未取得扱い） */
  data: any[] | null | undefined;
  /** ローディングフラグ */
  loading?: boolean;
  /** エラーメッセージ */
  error?: string | null;
  /** 高さ（px） */
  height?: number;
  /** 追加クラス */
  className?: string;
  /** テーマ */
  theme?: "dark" | "light";
}
