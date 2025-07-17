/**
 * データ収集関連の型定義
 */

import { BulkFundingRateCollectionResult } from "./funding-rate";
import { BulkOpenInterestCollectionResult } from "./open-interest";

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
 * 全データ一括収集結果
 *
 * OHLCV、ファンディングレート、オープンインタレストの
 * 全てのデータ収集結果をまとめた型です。
 */
export interface AllDataCollectionResult {
  /** OHLCVデータ収集結果 */
  ohlcv_result: BulkOHLCVCollectionResult;
  /** ファンディングレートデータ収集結果 */
  funding_rate_result: BulkFundingRateCollectionResult;
  /** オープンインタレストデータ収集結果 */
  open_interest_result: BulkOpenInterestCollectionResult;
}

/**
 * 一括差分更新結果
 *
 * OHLCV、ファンディングレート、オープンインタレストの
 * 差分データを一括で更新した結果を表現します。
 */
export interface BulkIncrementalUpdateResult {
  /** 処理成功フラグ */
  success: boolean;
  /** 結果メッセージ */
  message: string;
  /** データ詳細 */
  data: {
    /** OHLCVデータ更新結果 */
    ohlcv: {
      /** 通貨ペア */
      symbol: string;
      /** 時間軸（"all"の場合は全時間足） */
      timeframe: string;
      /** 保存件数 */
      saved_count: number;
      /** 成功フラグ */
      success: boolean;
      /** エラーメッセージ（失敗時） */
      error?: string;
      /** 時間足別の結果（全時間足の場合） */
      timeframe_results?: {
        [timeframe: string]: {
          symbol: string;
          timeframe: string;
          saved_count: number;
          success: boolean;
          error?: string;
        };
      };
    };
    /** ファンディングレートデータ更新結果 */
    funding_rate: {
      /** 通貨ペア */
      symbol: string;
      /** 保存件数 */
      saved_count: number;
      /** 成功フラグ */
      success: boolean;
      /** 最新タイムスタンプ */
      latest_timestamp?: number | null;
      /** エラーメッセージ（失敗時） */
      error?: string;
    };
    /** オープンインタレストデータ更新結果 */
    open_interest: {
      /** 通貨ペア */
      symbol: string;
      /** 保存件数 */
      saved_count: number;
      /** 成功フラグ */
      success: boolean;
      /** 最新タイムスタンプ */
      latest_timestamp?: number | null;
      /** エラーメッセージ（失敗時） */
      error?: string;
    };
  };
  /** 総保存件数 */
  total_saved_count: number;
  /** エラーリスト */
  errors?: string[];
  /** タイムスタンプ */
  timestamp: string;
}

/**
 * 一括差分更新APIレスポンス
 *
 * APIから返される一括差分更新データの形式を定義します。
 */
export interface BulkIncrementalUpdateResponse {
  /** 成功フラグ */
  success: boolean;
  /** データ */
  data: BulkIncrementalUpdateResult;
  /** メッセージ */
  message?: string;
  /** タイムスタンプ */
  timestamp: string;
}
