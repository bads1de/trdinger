/**
 * オープンインタレスト関連の型定義
 */

/**
 * OIデータ
 *
 * 無期限契約のOI（建玉残高）情報を表現します。
 */
export interface OpenInterestData {
  /** 通貨ペアシンボル（例: "BTC/USDT:USDT"） */
  symbol: string;
  /** OI値（USD建て相当。取引所により算出式が異なる場合あり） */
  open_interest_value: number;
  /** データ時刻（ISO形式。上流が示す対象時刻） */
  data_timestamp: string;
  /** データ取得時刻（ISO形式。保存した時刻） */
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
    /** データ件数（返却した open_interest の件数） */
    count: number;
    /** OIデータの配列 */
    open_interest: OpenInterestData[];
  };
  /** メッセージ（警告/補足） */
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
  /** 取得件数（上流から取得した件数） */
  fetched_count: number;
  /** 保存件数（DBへ保存できた件数） */
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
  /** メッセージ（警告/補足） */
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
  results: OpenInterestCollectionResult[];
  /** 失敗したシンボルの詳細（エラー要約） */
  failures: Array<{
    symbol: string;
    error: string;
  }>;
}
