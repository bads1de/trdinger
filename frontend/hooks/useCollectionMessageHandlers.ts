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

export interface UseCollectionMessageHandlersDeps {
  setMessage: (key: string, message: string, duration?: number) => void;
  fetchFearGreedData: () => Promise<void> | void;
  fetchDataStatus: () => void;
  fetchOHLCVData: () => Promise<void> | void;
  fetchFundingRateData: () => Promise<void> | void;
  fetchOpenInterestData: () => Promise<void> | void;
  MESSAGE_KEYS: Record<string, string>;
  MESSAGE_DURATION: Record<"SHORT" | "MEDIUM" | "LONG", number>;
}

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
    handleCollectionStart,
    handleCollectionError,
    collectionHandlers,
  };
};
