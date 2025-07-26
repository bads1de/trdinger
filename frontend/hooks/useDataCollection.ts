/**
 * データ収集用カスタムフック
 *
 * 各種データ収集API呼び出しを統一的に管理するカスタムフックです。
 * OHLCV、ファンディングレート、オープンインタレストデータの収集を提供します。
 *
 */

import { useCallback } from "react";
import { useApiCall } from "./useApiCall";

const useCollection = <T>() => {
  const { execute, loading: isLoading, error } = useApiCall<T>();

  const collect = useCallback(
    async (
      endpoint: string,
      confirmMessage: string,
      onSuccess?: (data: T) => void,
      onError?: (error: string) => void
    ) => {
      await execute(endpoint, {
        method: "POST",
        confirmMessage,
        onSuccess,
        onError: (error) => {
          onError?.(error || "データ収集に失敗しました");
        },
      });
    },
    [execute]
  );

  return { isLoading, error, collect };
};

/**
 * データ収集用の専用フック
 */
export const useDataCollection = () => {
  const ohlcv = useCollection();
  const fundingRate = useCollection();
  const openInterest = useCollection();

  const collectOHLCVData = useCallback(
    (onSuccess?: (data: any) => void, onError?: (error: string) => void) => {
      ohlcv.collect(
        "/api/data-collection/bulk-historical",
        "全ペア・全時間軸でOHLCVデータを収集しますか？",
        onSuccess,
        onError
      );
    },
    [ohlcv]
  );

  const collectFundingRateData = useCallback(
    (onSuccess?: (data: any) => void, onError?: (error: string) => void) => {
      fundingRate.collect(
        "/api/funding-rates/bulk-collect",
        "FRデータを収集しますか？",
        onSuccess,
        onError
      );
    },
    [fundingRate]
  );

  const collectOpenInterestData = useCallback(
    (onSuccess?: (data: any) => void, onError?: (error: string) => void) => {
      openInterest.collect(
        "/api/open-interest/bulk-collect",
        "OIデータを収集しますか？",
        onSuccess,
        onError
      );
    },
    [openInterest]
  );

  return {
    ohlcv: {
      loading: ohlcv.isLoading,
      error: ohlcv.error,
      collect: collectOHLCVData,
    },
    fundingRate: {
      loading: fundingRate.isLoading,
      error: fundingRate.error,
      collect: collectFundingRateData,
    },
    openInterest: {
      loading: openInterest.isLoading,
      error: openInterest.error,
      collect: collectOpenInterestData,
    },
    isAnyLoading:
      ohlcv.isLoading || fundingRate.isLoading || openInterest.isLoading,
    hasAnyError: !!(ohlcv.error || fundingRate.error || openInterest.error),
  };
};
