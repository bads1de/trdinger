/**
 * ファンディングレート関連の型定義
 */

/**
 * FRデータ
 *
 * 無期限契約のFR情報を表現します。
 */
export interface FundingRateData {
  /** 通貨ペアシンボル（例: "BTC/USDT:USDT"） */
  symbol: string;
  /** FR（例: -0.00015708。通常は8時間毎などの率） */
  funding_rate: number;
  /** ファンディング時刻（ISO形式） */
  funding_timestamp: string;
  /** データ取得時刻（ISO形式） */
  timestamp: string;
  /** 次回ファンディング時刻（ISO形式、オプション） */
  next_funding_timestamp?: string | null;
  /** マーク価格（任意。USD建て想定） */
  mark_price?: number | null;
  /** インデックス価格（任意。USD建て想定） */
  index_price?: number | null;
}

/**
 * FR収集結果
 *
 * FRデータ収集の結果を表現します。
 */
export interface FundingRateCollectionResult {
  /** 通貨ペア */
  symbol: string;
  /** 取得件数（上流から取得した件数） */
  fetched_count: number;
  /** 保存件数（DBへ実際に保存できた件数） */
  saved_count: number;
  /** 成功フラグ */
  success: boolean;
}

/**
 * 一括FR収集結果
 *
 * 複数シンボルのFRデータ一括収集の結果を表現します。
 */
export interface BulkFundingRateCollectionResult {
  /** 処理成功フラグ */
  success: boolean;
  /** 結果メッセージ（概要） */
  message: string;
  /** 処理開始時刻（ISO文字列） */
  started_at?: string;
  /** 処理ステータス */
  status?: "started" | "in_progress" | "completed" | "error";
  /** 総シンボル数 */
  total_symbols: number;
  /** 成功したシンボル数 */
  successful_symbols: number;
  /** 失敗したシンボル数 */
  failed_symbols: number;
  /** 総保存レコード数（全シンボル合算） */
  total_saved_records: number;
  /** 個別結果 */
  results: FundingRateCollectionResult[];
  /** 失敗したシンボルの詳細（エラー要約） */
  failures: Array<{
    symbol: string;
    error: string;
  }>;
}
