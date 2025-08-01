/**
 * データ収集メッセージハンドラー用カスタムフック
 *
 * 各種データ収集処理の結果に応じたメッセージ生成とハンドリング機能を提供します。
 * OHLCV、ファンディングレート、オープンインタレスト、Fear & Greed Indexなどの
 * データ収集結果を統一的に処理します。
 */

import { useCallback } from "react";
import {
  BulkOHLCVCollectionResult,
  AllDataCollectionResult,
} from "@/types/data-collection";
import {
  BulkFundingRateCollectionResult,
  FundingRateCollectionResult,
} from "@/types/funding-rate";
import {
  BulkOpenInterestCollectionResult,
  OpenInterestCollectionResult,
} from "@/types/open-interest";
import { FearGreedCollectionResult } from "@/hooks/useFearGreedData";

/**
 * データ収集メッセージハンドラーの依存関係の型
 */
export interface UseCollectionMessageHandlersDeps {
  /** メッセージを設定する関数 */
  setMessage: (key: string, message: string, duration?: number) => void;
  /** Fear & Greedデータを取得する関数 */
  fetchFearGreedData: () => Promise<void> | void;
  /** データステータスを取得する関数 */
  fetchDataStatus: () => void;
  /** OHLCVデータを取得する関数 */
  fetchOHLCVData: () => Promise<void> | void;
  /** ファンディングレートデータを取得する関数 */
  fetchFundingRateData: () => Promise<void> | void;
  /** オープンインタレストデータを取得する関数 */
  fetchOpenInterestData: () => Promise<void> | void;
  /** メッセージキーの定義 */
  MESSAGE_KEYS: Record<string, string>;
  /** メッセージ表示期間の定義 */
  MESSAGE_DURATION: Record<"SHORT" | "MEDIUM" | "LONG", number>;
}

/**
 * データ収集メッセージハンドラーフック
 *
 * 各種データ収集処理の結果に応じたメッセージ生成とハンドリング機能を提供します。
 * OHLCV、ファンディングレート、オープンインタレスト、Fear & Greed Indexなどの
 * データ収集結果を統一的に処理します。
 *
 * @example
 * ```tsx
 * const {
 *   handleCollectionStart,
 *   handleCollectionError,
 *   collectionHandlers
 * } = useCollectionMessageHandlers({
 *   setMessage,
 *   fetchFearGreedData,
 *   fetchDataStatus,
 *   fetchOHLCVData,
 *   fetchFundingRateData,
 *   fetchOpenInterestData,
 *   MESSAGE_KEYS,
 *   MESSAGE_DURATION
 * });
 *
 * // データ収集開始時の処理
 * handleCollectionStart('BULK_COLLECTION', 'bulk', result, 5000);
 *
 * // エラー時の処理
 * handleCollectionError('BULK_COLLECTION', '収集に失敗しました');
 * ```
 *
 * @param {UseCollectionMessageHandlersDeps} deps - 依存関係オブジェクト
 * @returns {{
 *   handleCollectionStart: (messageKey: string, messageType: string, result: any, duration?: number, onSuccess?: (result: any) => void) => void,
 *   handleCollectionError: (messageKey: string, errorMessage: string, duration?: number) => void,
 *   collectionHandlers: Record<string, any>
 * }} メッセージハンドリング関連の関数
 */
export const useCollectionMessageHandlers = ({
  setMessage,
  fetchFearGreedData,
  fetchDataStatus,
  fetchOHLCVData,
  fetchFundingRateData,
  fetchOpenInterestData,
  MESSAGE_KEYS,
  MESSAGE_DURATION,
}: UseCollectionMessageHandlersDeps) => {
  const messageGenerators: Record<string, (result: any) => string> = {
    bulk: (result: BulkOHLCVCollectionResult) =>
      `🚀 ${result.message} (${result.total_tasks}タスク)`,
    funding: (
      result: BulkFundingRateCollectionResult | FundingRateCollectionResult
    ) => {
      if ("total_symbols" in result) {
        return `🚀 ${result.message} (${result.successful_symbols}/${result.total_symbols}シンボル成功)`;
      }
      return `🚀 ${result.symbol}のFRデータ収集完了 (${result.saved_count}件保存)`;
    },
    openinterest: (
      result: BulkOpenInterestCollectionResult | OpenInterestCollectionResult
    ) => {
      if ("total_symbols" in result) {
        return `🚀 ${result.message} (${result.successful_symbols}/${result.total_symbols}シンボル成功)`;
      }
      return `🚀 ${result.symbol}のOIデータ収集完了 (${result.saved_count}件保存)`;
    },
    feargreed: (result: FearGreedCollectionResult) =>
      result.success
        ? `🚀 Fear & Greed Index収集完了 (取得:${result.fetched_count}件, 挿入:${result.inserted_count}件)`
        : `❌ ${result.message}`,
    alldata: (result: AllDataCollectionResult) => {
      if (result.ohlcv_result?.status === "completed") {
        const ohlcvCount = result.ohlcv_result?.total_tasks || 0;
        const fundingCount =
          result.funding_rate_result?.total_saved_records || 0;
        const openInterestCount =
          result.open_interest_result?.total_saved_records || 0;
        return `🚀 全データ収集完了！ OHLCV:${ohlcvCount}タスク, FR:${fundingCount}件, OI:${openInterestCount}件, TI:自動計算済み`;
      }
      return `🔄 ${result.ohlcv_result?.message || "処理中..."} (実行中...)`;
    },
    default: (result: any) => `🚀 ${result.message || "処理完了"}`,
  };

  const generateCollectionMessage = useCallback(
    (type: string, result: any): string => {
      const generator = messageGenerators[type] || messageGenerators.default;
      return generator(result);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  const handleCollectionStart = useCallback(
    (
      messageKey: string,
      messageType: string,
      result: any,
      duration?: number,
      onSuccess?: (result: any) => void
    ) => {
      setMessage(
        messageKey,
        generateCollectionMessage(messageType, result),
        duration
      );
      if (onSuccess) {
        onSuccess(result);
      }
    },
    [setMessage, generateCollectionMessage]
  );

  const handleCollectionError = useCallback(
    (messageKey: string, errorMessage: string, duration?: number) => {
      setMessage(messageKey, `❌ ${errorMessage}`, duration);
    },
    [setMessage]
  );

  const collectionHandlers = {
    bulk: {
      key: MESSAGE_KEYS.BULK_COLLECTION,
      type: "bulk",
      onSuccess: () => fetchDataStatus(),
    },
    funding: {
      key: MESSAGE_KEYS.FUNDING_RATE_COLLECTION,
      type: "funding",
    },
    openinterest: {
      key: MESSAGE_KEYS.OPEN_INTEREST_COLLECTION,
      type: "openinterest",
    },
    feargreed: {
      key: MESSAGE_KEYS.FEAR_GREED_COLLECTION,
      type: "feargreed",
      onSuccess: (result: FearGreedCollectionResult) => {
        if (result.success) {
          fetchFearGreedData();
        }
        fetchDataStatus();
      },
    },
    alldata: {
      key: MESSAGE_KEYS.ALL_DATA_COLLECTION,
      type: "alldata",
      duration: MESSAGE_DURATION.MEDIUM,
      onSuccess: () => {
        fetchDataStatus();
        setTimeout(() => {
          fetchOHLCVData();
          fetchFundingRateData();
          fetchOpenInterestData();
        }, 3000);
      },
    },
  };

  return {
    /** データ収集開始時のメッセージハンドラー */
    handleCollectionStart,
    /** データ収集エラー時のメッセージハンドラー */
    handleCollectionError,
    /** 各種データ収集ハンドラーの定義 */
    collectionHandlers,
  };
};
