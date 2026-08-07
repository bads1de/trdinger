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
  /** 取引リターン（%値。例: 2.5 = 2.5%） */
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
  /** 実行時の設定 */
  config_json: Record<string, unknown>;
  /** 総リターン（%値。例: 15.0 = 15%） */
  total_return: number;
  /** シャープレシオ */
  sharpe_ratio: number;
  /** 最大ドローダウン（%値。例: 20.0 = 20%） */
  max_drawdown: number;
  /** 勝率（%値。例: 55.0 = 55%） */
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
  /** 追加メトリクス */
  performance_metrics?: Record<string, unknown>;
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
 * 戦略遺伝子（GA戦略）の単一条件
 */
export interface StrategyCondition {
  /** 左辺オペランド（例: "rsi"） */
  left_operand: string;
  /** 比較演算子（例: ">", "<", "crosses_above"） */
  operator: string;
  /** 右辺オペランド（数値または文字列） */
  right_operand: string | number;
}

/**
 * OR条件で結合された条件グループ
 */
export interface StrategyConditionGroup {
  /** グループ内の条件群（OR結合） */
  conditions: StrategyCondition[];
}

/**
 * 単一条件または条件グループ
 */
export type StrategyConditionOrGroup =
  | StrategyCondition
  | StrategyConditionGroup;

/**
 * 戦略遺伝子で使用するインジケーター
 */
export interface StrategyIndicator {
  /** インジケータータイプ（例: "rsi", "ema"） */
  type: string;
  /** パラメータ辞書 */
  parameters: Record<string, any>;
  /** 有効フラグ */
  enabled: boolean;
}

/**
 * GAで生成された戦略遺伝子
 */
export interface StrategyGene {
  /** 遺伝子ID（任意） */
  id?: string;
  /** 使用インジケーター群 */
  indicators?: StrategyIndicator[];
  /** エントリー条件（全方向共通。旧形式） */
  entry_conditions?: StrategyConditionOrGroup[];
  /** ロングエントリー条件 */
  long_entry_conditions?: StrategyConditionOrGroup[];
  /** ショートエントリー条件 */
  short_entry_conditions?: StrategyConditionOrGroup[];
  /** エグジット条件 */
  exit_conditions?: StrategyConditionOrGroup[];
  /** TP/SL設定 */
  tpsl_gene?: Record<string, any>;
  /** 資金管理設定 */
  position_sizing_gene?: Record<string, any>;
  /** メタデータ */
  metadata?: Record<string, any>;
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
