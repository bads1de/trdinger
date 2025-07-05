/**
 * トレーディング戦略関連の型定義
 *
 * このファイルには、バックテストシステムで使用される
 * 全ての型定義が含まれています。
 *
 */

/**
 * 一括OHLCVデータ収集結果
 */
export interface BulkOHLCVCollectionResult {
  /** 処理成功フラグ */
  success: boolean;
  /** 結果メッセージ */
  message: string;
  /** 処理開始時刻 */
  started_at: string;
  /** 処理ステータス */
  status: "started" | "in_progress" | "completed" | "error";
  /** 総組み合わせ数 */
  total_combinations?: number;
  /** 実際に実行されるタスク数 */
  actual_tasks?: number;
  /** スキップされたタスク数 */
  skipped_tasks?: number;
  /** 失敗したタスク数 */
  failed_tasks?: number;
  /** 処理対象の総数（後方互換性のため） */
  total_tasks?: number;
  /** 処理済みの数（後方互換性のため） */
  completed_tasks?: number;
  /** 成功した数（後方互換性のため） */
  successful_tasks?: number;
  /** 対象シンボル一覧 */
  symbols?: string[];
  /** 対象時間軸一覧 */
  timeframes?: string[];
  /** タスクの詳細情報 */
  task_details?: {
    /** 実行中のタスク */
    executing: Array<{
      symbol: string;
      original_symbol: string;
      timeframe: string;
    }>;
    /** スキップされたタスク */
    skipped: Array<{
      symbol: string;
      original_symbol: string;
      timeframe: string;
      reason: string;
    }>;
    /** 失敗したタスク */
    failed: Array<{
      symbol: string;
      timeframe: string;
      error: string;
    }>;
  };
  /** 個別タスクの結果（後方互換性のため） */
  task_results?: Array<{
    symbol: string;
    timeframe: string;
    success: boolean;
    message: string;
    saved_count?: number;
    skipped_count?: number;
  }>;
}

/**
 * 売買条件の定義
 *
 * エントリーやエグジットの条件を文字列で表現します。
 *
 * @example
 * ```typescript
 * const entryCondition: TradingCondition = {
 *   condition: 'close > SMA(close, 20)',
 *   description: '終値が20期間移動平均を上回る'
 * }
 * ```
 */
export interface TradingCondition {
  /** 条件式（例: "close > SMA(close, 20)"） */
  condition: string;
  /** 条件の説明（オプション） */
  description?: string;
}

/**
 * トレーディング戦略の定義
 *
 * 完全なトレーディング戦略の設定を表現します。
 * テクニカル指標、エントリー・エグジット条件を含みます。
 */
export interface TradingStrategy {
  /** 戦略の一意識別子（オプション） */
  id?: string;
  /** 戦略名 */
  strategy_name: string;
  /** 対象通貨ペア（例: "BTC/USD"） */
  target_pair: string;
  /** 時間足（例: "1h", "1d"） */
  timeframe: string;
  /** 使用するテクニカル指標のリスト（削除済み） */
  // indicators: TechnicalIndicator[];
  /** エントリー条件のリスト（AND条件） */
  entry_rules: TradingCondition[];
  /** エグジット条件のリスト（OR条件） */
  exit_rules: TradingCondition[];
  /** 作成日時（オプション） */
  created_at?: Date;
  /** 更新日時（オプション） */
  updated_at?: Date;
}

/**
 * バックテストの設定
 *
 * バックテスト実行時のパラメータを定義します。
 * 戦略、期間、初期資金、手数料等を含みます。
 */
export interface BacktestConfig {
  /** バックテストする戦略 */
  strategy: TradingStrategy;
  /** バックテスト開始日（ISO形式） */
  start_date: string;
  /** バックテスト終了日（ISO形式） */
  end_date: string;
  /** 初期資金（USD） */
  initial_capital: number;
  /** 手数料率（デフォルト: 0.00055 = 0.055%） */
  commission_rate?: number;
  /** スリッパージ率（オプション） */
  slippage_rate?: number;
}

/**
 * バックテストの結果
 *
 * バックテスト実行後の詳細な結果とパフォーマンス指標を含みます。
 */
export interface BacktestResult {
  /** 結果の一意識別子（オプション） */
  id?: string;
  /** 戦略ID */
  strategy_id: string;
  /** バックテスト設定 */
  config: BacktestConfig;
  /** 総リターン（投資期間全体の収益率） */
  total_return: number;
  /** シャープレシオ（リスク調整後リターン） */
  sharpe_ratio: number;
  /** 最大ドローダウン（最大下落率） */
  max_drawdown: number;
  /** 勝率（勝ち取引の割合） */
  win_rate: number;
  /** プロフィットファクター（総利益/総损失） */
  profit_factor: number;
  /** 総取引数 */
  total_trades: number;
  /** 勝ち取引数 */
  winning_trades: number;
  /** 負け取引数 */
  losing_trades: number;
  /** 平均利益（勝ち取引あたり） */
  avg_win: number;
  /** 平均损失（負け取引あたり） */
  avg_loss: number;
  /** 損益曲線データ */
  equity_curve: EquityPoint[];
  /** 取引履歴 */
  trade_history: Trade[];
  /** 結果作成日時 */
  created_at: Date;
}

/**
 * 損益曲線のポイント
 *
 * 特定の時点での資産価値とドローダウンを表現します。
 * グラフ表示やパフォーマンス分析に使用されます。
 */
export interface EquityPoint {
  /** タイムスタンプ（ISO形式） */
  timestamp: string;
  /** 総資産価値（USD） */
  equity: number;
  /** ドローダウン率（ピークからの下落率） */
  drawdown: number;
}

/**
 * 取引履歴
 *
 * 個々の売買取引の詳細情報を表現します。
 * バックテスト結果の分析やデバッグに使用されます。
 */
export interface Trade {
  /** 取引の一意識別子 */
  id: string;
  /** 取引実行時刻（ISO形式） */
  timestamp: string;
  /** 取引タイプ（買いまたは売り） */
  type: "buy" | "sell";
  /** 取引価格（USD） */
  price: number;
  /** 取引数量 */
  quantity: number;
  /** 手数料（USD） */
  commission: number;
  /** 損益（USD、売り取引のみ） */
  pnl?: number;
}

/**
 * 価格データ（OHLCV）
 *
 * 仮想通貨の価格データを表現します。
 * バックテストの入力データとして使用されます。
 */
export interface PriceData {
  /** タイムスタンプ（ISO形式） */
  timestamp: string;
  /** 始値（USD） */
  open: number;
  /** 高値（USD） */
  high: number;
  /** 安値（USD） */
  low: number;
  /** 終値（USD） */
  close: number;
  /** 出来高 */
  volume: number;
}

/**
 * 時間軸の定義
 *
 * データ表示で使用可能な時間軸を定義します。
 */
export type TimeFrame = "15m" | "30m" | "1h" | "4h" | "1d";

/**
 * 時間軸の表示情報
 *
 * 時間軸の表示名と説明を含みます。
 */
export interface TimeFrameInfo {
  /** 時間軸の値 */
  value: TimeFrame;
  /** 表示名 */
  label: string;
  /** 説明 */
  description: string;
}

/**
 * OHLCVデータのAPIレスポンス
 *
 * APIから返されるOHLCVデータの形式を定義します。
 */
export interface OHLCVResponse {
  /** 成功フラグ */
  success: boolean;
  /** データ */
  data: {
    /** 通貨ペア */
    symbol: string;
    /** 時間軸 */
    timeframe: TimeFrame;
    /** OHLCVデータの配列 */
    ohlcv: PriceData[];
  };
  /** メッセージ */
  message?: string;
  /** タイムスタンプ */
  timestamp: string;
}

/**
 * 利用可能な通貨ペア
 *
 * システムで取引可能な通貨ペアの情報を定義します。
 */
export interface TradingPair {
  /** 通貨ペアのシンボル（例: "BTC/USD"） */
  symbol: string;
  /** 表示名（例: "Bitcoin / US Dollar"） */
  name: string;
  /** ベース通貨（例: "BTC"） */
  base: string;
  /** クォート通貨（例: "USD"） */
  quote: string;
}

/**
 * FRデータ
 *
 * 無期限契約のFR情報を表現します。
 */
export interface FundingRateData {
  /** 通貨ペアシンボル（例: "BTC/USDT:USDT"） */
  symbol: string;
  /** FR（例: -0.00015708） */
  funding_rate: number;
  /** ファンディング時刻（ISO形式） */
  funding_timestamp: string;
  /** データ取得時刻（ISO形式） */
  timestamp: string;
  /** 次回ファンディング時刻（ISO形式、オプション） */
  next_funding_timestamp?: string | null;
  /** マーク価格（オプション） */
  mark_price?: number | null;
  /** インデックス価格（オプション） */
  index_price?: number | null;
}

/**
 * 現在のFRデータ
 *
 * リアルタイムのFR情報を表現します。
 */
export interface CurrentFundingRateData {
  /** 通貨ペアシンボル（例: "BTC/USDT:USDT"） */
  symbol: string;
  /** FR（例: -0.00015708） */
  funding_rate: number;
  /** ファンディング時刻（ISO形式） */
  funding_timestamp?: string | null;
  /** 次回ファンディング時刻（ISO形式） */
  next_funding_timestamp?: string | null;
  /** マーク価格 */
  mark_price?: number | null;
  /** インデックス価格 */
  index_price?: number | null;
  /** データ取得時刻（ISO形式） */
  timestamp?: string | null;
}

/**
 * FRAPIレスポンス
 *
 * APIから返されるFRデータの形式を定義します。
 */
export interface FundingRateResponse {
  /** 成功フラグ */
  success: boolean;
  /** データ */
  data: {
    /** 通貨ペア */
    symbol: string;
    /** データ件数 */
    count: number;
    /** FRデータの配列 */
    funding_rates: FundingRateData[];
  };
  /** メッセージ */
  message?: string;
}

/**
 * FR収集結果
 *
 * FRデータ収集の結果を表現します。
 */
export interface FundingRateCollectionResult {
  /** 通貨ペア */
  symbol: string;
  /** 取得件数 */
  fetched_count: number;
  /** 保存件数 */
  saved_count: number;
  /** 成功フラグ */
  success: boolean;
}

/**
 * FR収集APIレスポンス
 *
 * FR収集APIのレスポンス形式を定義します。
 */
export interface FundingRateCollectionResponse {
  /** 成功フラグ */
  success: boolean;
  /** データ */
  data: FundingRateCollectionResult;
  /** メッセージ */
  message?: string;
}

/**
 * 一括FR収集結果
 *
 * 複数シンボルのFRデータ一括収集の結果を表現します。
 */
export interface BulkFundingRateCollectionResult {
  /** 処理成功フラグ */
  success: boolean;
  /** 結果メッセージ */
  message: string;
  /** 処理開始時刻 */
  started_at?: string;
  /** 処理ステータス */
  status?: "started" | "in_progress" | "completed" | "error";
  /** 総シンボル数 */
  total_symbols: number;
  /** 成功したシンボル数 */
  successful_symbols: number;
  /** 失敗したシンボル数 */
  failed_symbols: number;
  /** 総保存レコード数 */
  total_saved_records: number;
  /** 個別結果 */
  results: FundingRateCollectionResult[];
  /** 失敗したシンボルの詳細 */
  failures: Array<{
    symbol: string;
    error: string;
  }>;
}

/**
 * OIデータ
 *
 * 無期限契約のOI（建玉残高）情報を表現します。
 */
export interface OpenInterestData {
  /** 通貨ペアシンボル（例: "BTC/USDT:USDT"） */
  symbol: string;
  /** OI値（USD建て） */
  open_interest_value: number;
  /** データ時刻（ISO形式） */
  data_timestamp: string;
  /** データ取得時刻（ISO形式） */
  timestamp: string;
}

/**
 * OIAPIレスポンス
 *
 * APIから返されるOIデータの形式を定義します。
 */
export interface OpenInterestResponse {
  /** 成功フラグ */
  success: boolean;
  /** データ */
  data: {
    /** 通貨ペア */
    symbol: string;
    /** データ件数 */
    count: number;
    /** OIデータの配列 */
    open_interest: OpenInterestData[];
  };
  /** メッセージ */
  message?: string;
}

/**
 * OI収集結果
 *
 * OIデータ収集の結果を表現します。
 */
export interface OpenInterestCollectionResult {
  /** 通貨ペア */
  symbol: string;
  /** 取得件数 */
  fetched_count: number;
  /** 保存件数 */
  saved_count: number;
  /** 成功フラグ */
  success: boolean;
}

/**
 * OI収集APIレスポンス
 *
 * OI収集APIのレスポンス形式を定義します。
 */
export interface OpenInterestCollectionResponse {
  /** 成功フラグ */
  success: boolean;
  /** データ */
  data: OpenInterestCollectionResult;
  /** メッセージ */
  message?: string;
}

/**
 * 一括OI収集結果
 *
 * 複数シンボルのOIデータ一括収集の結果を表現します。
 */
export interface BulkOpenInterestCollectionResult {
  /** 処理成功フラグ */
  success: boolean;
  /** 結果メッセージ */
  message: string;
  /** 処理開始時刻 */
  started_at?: string;
  /** 処理ステータス */
  status?: "started" | "in_progress" | "completed" | "error";
  /** 総シンボル数 */
  total_symbols: number;
  /** 成功したシンボル数 */
  successful_symbols: number;
  /** 失敗したシンボル数 */
  failed_symbols: number;
  /** 総保存レコード数 */
  total_saved_records: number;
  /** 個別結果 */
  results: OpenInterestCollectionResult[];
  /** 失敗したシンボルの詳細 */
  failures: Array<{
    symbol: string;
    error: string;
  }>;
}
