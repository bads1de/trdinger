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
  const generateCollectionMessage = useCallback(
    (type: string, result: any): string => {
      switch (type) {
        case "bulk":
          return `🚀 ${result.message} (${result.total_tasks}タスク)`;
        case "funding":
          if ("total_symbols" in result) {
            return `🚀 ${result.message} (${result.successful_symbols}/${result.total_symbols}シンボル成功)`;
          }
          return `🚀 ${result.symbol}のFRデータ収集完了 (${result.saved_count}件保存)`;
        case "openinterest":
          if ("total_symbols" in result) {
            return `🚀 ${result.message} (${result.successful_symbols}/${result.total_symbols}シンボル成功)`;
          }
          return `🚀 ${result.symbol}のOIデータ収集完了 (${result.saved_count}件保存)`;
        case "feargreed":
          return result.success
            ? `🚀 Fear & Greed Index収集完了 (取得:${result.fetched_count}件, 挿入:${result.inserted_count}件)`
            : `❌ ${result.message}`;
        case "alldata":
          if (result.ohlcv_result?.status === "completed") {
            const ohlcvCount = result.ohlcv_result?.total_tasks || 0;
            const fundingCount =
              result.funding_rate_result?.total_saved_records || 0;
            const openInterestCount =
              result.open_interest_result?.total_saved_records || 0;
            return `🚀 全データ収集完了！ OHLCV:${ohlcvCount}タスク, FR:${fundingCount}件, OI:${openInterestCount}件, TI:自動計算済み`;
          }
          return `🔄 ${
            result.ohlcv_result?.message || "処理中..."
          } (実行中...)`;
        default:
          return `🚀 ${result.message || "処理完了"}`;
      }
    },
    []
  );

  const handleBulkCollectionStart = useCallback(
    (result: BulkOHLCVCollectionResult) => {
      setMessage(
        MESSAGE_KEYS.BULK_COLLECTION,
        generateCollectionMessage("bulk", result)
      );
      fetchDataStatus();
    },
    [
      setMessage,
      MESSAGE_KEYS.BULK_COLLECTION,
      generateCollectionMessage,
      fetchDataStatus,
    ]
  );

  const handleBulkCollectionError = useCallback(
    (errorMessage: string) => {
      setMessage(
        MESSAGE_KEYS.BULK_COLLECTION,
        `❌ ${errorMessage}`,
        MESSAGE_DURATION.SHORT
      );
    },
    [setMessage, MESSAGE_KEYS.BULK_COLLECTION, MESSAGE_DURATION.SHORT]
  );

  const handleFundingRateCollectionStart = useCallback(
    (result: BulkFundingRateCollectionResult | FundingRateCollectionResult) => {
      setMessage(
        MESSAGE_KEYS.FUNDING_RATE_COLLECTION,
        generateCollectionMessage("funding", result)
      );
    },
    [
      setMessage,
      MESSAGE_KEYS.FUNDING_RATE_COLLECTION,
      generateCollectionMessage,
    ]
  );

  const handleFundingRateCollectionError = useCallback(
    (errorMessage: string) => {
      setMessage(
        MESSAGE_KEYS.FUNDING_RATE_COLLECTION,
        `❌ ${errorMessage}`,
        MESSAGE_DURATION.SHORT
      );
    },
    [setMessage, MESSAGE_KEYS.FUNDING_RATE_COLLECTION, MESSAGE_DURATION.SHORT]
  );

  const handleOpenInterestCollectionStart = useCallback(
    (
      result: BulkOpenInterestCollectionResult | OpenInterestCollectionResult
    ) => {
      setMessage(
        MESSAGE_KEYS.OPEN_INTEREST_COLLECTION,
        generateCollectionMessage("openinterest", result)
      );
    },
    [
      setMessage,
      MESSAGE_KEYS.OPEN_INTEREST_COLLECTION,
      generateCollectionMessage,
    ]
  );

  const handleOpenInterestCollectionError = useCallback(
    (errorMessage: string) => {
      setMessage(
        MESSAGE_KEYS.OPEN_INTEREST_COLLECTION,
        `❌ ${errorMessage}`,
        MESSAGE_DURATION.SHORT
      );
    },
    [setMessage, MESSAGE_KEYS.OPEN_INTEREST_COLLECTION, MESSAGE_DURATION.SHORT]
  );

  const handleFearGreedCollectionStart = useCallback(
    (result: FearGreedCollectionResult) => {
      setMessage(
        MESSAGE_KEYS.FEAR_GREED_COLLECTION,
        generateCollectionMessage("feargreed", result)
      );
      if (result.success) {
        fetchFearGreedData();
      }
      fetchDataStatus();
    },
    [
      setMessage,
      MESSAGE_KEYS.FEAR_GREED_COLLECTION,
      generateCollectionMessage,
      fetchFearGreedData,
      fetchDataStatus,
    ]
  );

  const handleFearGreedCollectionError = useCallback(
    (errorMessage: string) => {
      setMessage(
        MESSAGE_KEYS.FEAR_GREED_COLLECTION,
        `❌ ${errorMessage}`,
        MESSAGE_DURATION.SHORT
      );
    },
    [setMessage, MESSAGE_KEYS.FEAR_GREED_COLLECTION, MESSAGE_DURATION.SHORT]
  );

  const handleAllDataCollectionStart = useCallback(
    (result: AllDataCollectionResult) => {
      setMessage(
        MESSAGE_KEYS.ALL_DATA_COLLECTION,
        generateCollectionMessage("alldata", result),
        MESSAGE_DURATION.MEDIUM
      );
      fetchDataStatus();

      setTimeout(() => {
        fetchOHLCVData();
        fetchFundingRateData();
        fetchOpenInterestData();
      }, 3000);
    },
    [
      setMessage,
      MESSAGE_KEYS.ALL_DATA_COLLECTION,
      MESSAGE_DURATION.MEDIUM,
      generateCollectionMessage,
      fetchDataStatus,
      fetchOHLCVData,
      fetchFundingRateData,
      fetchOpenInterestData,
    ]
  );

  const handleAllDataCollectionError = useCallback(
    (errorMessage: string) => {
      setMessage(
        MESSAGE_KEYS.ALL_DATA_COLLECTION,
        `❌ ${errorMessage}`,
        MESSAGE_DURATION.MEDIUM
      );
    },
    [setMessage, MESSAGE_KEYS.ALL_DATA_COLLECTION, MESSAGE_DURATION.MEDIUM]
  );

  return {
    generateCollectionMessage,
    handleBulkCollectionStart,
    handleBulkCollectionError,
    handleFundingRateCollectionStart,
    handleFundingRateCollectionError,
    handleOpenInterestCollectionStart,
    handleOpenInterestCollectionError,
    handleFearGreedCollectionStart,
    handleFearGreedCollectionError,
    handleAllDataCollectionStart,
    handleAllDataCollectionError,
  };
};
