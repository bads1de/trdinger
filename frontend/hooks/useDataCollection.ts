/**
 * データ収集用カスタムフック
 *
 * 各種データ収集API呼び出しを統一的に管理するカスタムフックです。
 * OHLCV、ファンディングレート、オープンインタレストデータの収集を提供します。
 *
 */

import { useCallback } from "react";
import { usePostRequest } from "./usePostRequest";

const createCollectionHook = <T,>() => {
  const { sendPostRequest, isLoading, error, data } = usePostRequest<T>();

  const collect = useCallback(
    async (
      endpoint: string,
      confirmMessage: string,
      onSuccess?: (data: T) => void,
      onError?: (error: string) => void
    ) => {
      if (window.confirm(confirmMessage)) {
        const { success, data, error } = await sendPostRequest(endpoint);
        if (success && data) {
          onSuccess?.(data);
        } else {
          onError?.(error || "データ収集に失敗しました");
        }
      }
    },
    [sendPostRequest]
  );

  return { isLoading, error, data, collect };
};

/**
 * データ収集用の専用フック
 */
export const useDataCollection = () => {
  const ohlcv = createCollectionHook();
  const fundingRate = createCollectionHook();
  const openInterest = createCollectionHook();

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
