/**
 * データ収集用カスタムフック
 *
 * 各種データ収集API呼び出しを統一的に管理するカスタムフックです。
 * OHLCV、ファンディングレート、オープンインタレストデータの収集を提供します。
 *
 */

import { useCallback } from "react";
import { useApiCall } from "./useApiCall";

/**
 * データ収集用の専用フック
 */
export const useDataCollection = () => {
  const ohlcvApi = useApiCall();
  const fundingRateApi = useApiCall();
  const openInterestApi = useApiCall();

  const collectOHLCVData = useCallback(
    async (
      onSuccess?: (data: any) => void,
      onError?: (error: string) => void
    ) => {
      return await ohlcvApi.execute("/api/data/ohlcv/bulk", {
        method: "POST",
        confirmMessage: "全ペア・全時間軸でOHLCVデータを収集しますか？",
        onSuccess,
        onError,
      });
    },
    [ohlcvApi]
  );

  const collectFundingRateData = useCallback(
    async (
      onSuccess?: (data: any) => void,
      onError?: (error: string) => void
    ) => {
      return await fundingRateApi.execute("/api/data/funding-rates/bulk", {
        method: "POST",
        confirmMessage: "FRデータを収集しますか？",
        onSuccess,
        onError,
      });
    },
    [fundingRateApi]
  );

  const collectOpenInterestData = useCallback(
    async (
      onSuccess?: (data: any) => void,
      onError?: (error: string) => void
    ) => {
      return await openInterestApi.execute("/api/data/open-interest/bulk", {
        method: "POST",
        confirmMessage: "OIデータを収集しますか？",
        onSuccess,
        onError,
      });
    },
    [openInterestApi]
  );

  return {
    // 個別のAPI状態
    ohlcv: {
      loading: ohlcvApi.loading,
      error: ohlcvApi.error,
      collect: collectOHLCVData,
      reset: ohlcvApi.reset,
    },
    fundingRate: {
      loading: fundingRateApi.loading,
      error: fundingRateApi.error,
      collect: collectFundingRateData,
      reset: fundingRateApi.reset,
    },
    openInterest: {
      loading: openInterestApi.loading,
      error: openInterestApi.error,
      collect: collectOpenInterestData,
      reset: openInterestApi.reset,
    },
    // 全体の状態
    isAnyLoading:
      ohlcvApi.loading || fundingRateApi.loading || openInterestApi.loading,
    hasAnyError: !!(
      ohlcvApi.error ||
      fundingRateApi.error ||
      openInterestApi.error
    ),
    resetAll: () => {
      ohlcvApi.reset();
      fundingRateApi.reset();
      openInterestApi.reset();
    },
  };
};
