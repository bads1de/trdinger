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
      successMessage: string | undefined,
      onSuccess?: (data: T) => void,
      onError?: (error: string) => void
    ) => {
      await execute(endpoint, {
        method: "POST",
        confirmMessage,
        successMessage,
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
 *
 * OHLCV、ファンディングレート、オープンインタレストデータの収集機能を提供します。
 * 各データタイプの収集状態を個別に管理し、統一的なインターフェースを提供します。
 *
 * @example
 * ```tsx
 * const {
 *   ohlcv,
 *   fundingRate,
 *   openInterest,
 *   isAnyLoading,
 *   hasAnyError
 * } = useDataCollection();
 *
 * // OHLCVデータを収集
 * ohlcv.collect(
 *   (data) => console.log('収集完了:', data),
 *   (error) => console.error('収集失敗:', error)
 * );
 *
 * // 全体のローディング状態を確認
 * if (isAnyLoading) {
 *   return <LoadingSpinner />;
 * }
 * ```
 *
 * @returns {{
 *   ohlcv: { loading: boolean, error: string | null, collect: (onSuccess?: (data: any) => void, onError?: (error: string) => void, successMessage?: string) => void },
 *   fundingRate: { loading: boolean, error: string | null, collect: (onSuccess?: (data: any) => void, onError?: (error: string) => void, successMessage?: string) => void },
 *   openInterest: { loading: boolean, error: string | null, collect: (onSuccess?: (data: any) => void, onError?: (error: string) => void, successMessage?: string) => void },
 *   isAnyLoading: boolean,
 *   hasAnyError: boolean
 * }} データ収集関連の状態と操作関数
 */
export const useDataCollection = () => {
  const ohlcv = useCollection();
  const fundingRate = useCollection();
  const openInterest = useCollection();

  const collectOHLCVData = useCallback(
    (
      onSuccess?: (data: any) => void,
      onError?: (error: string) => void,
      successMessage?: string
    ) => {
      ohlcv.collect(
        "/api/data-collection/bulk-historical",
        "全ペア・全時間軸でOHLCVデータを収集しますか？",
        successMessage,
        onSuccess,
        onError
      );
    },
    [ohlcv]
  );

  const collectFundingRateData = useCallback(
    (
      onSuccess?: (data: any) => void,
      onError?: (error: string) => void,
      successMessage?: string
    ) => {
      fundingRate.collect(
        "/api/funding-rates/bulk-collect",
        "FRデータを収集しますか？",
        successMessage,
        onSuccess,
        onError
      );
    },
    [fundingRate]
  );

  const collectOpenInterestData = useCallback(
    (
      onSuccess?: (data: any) => void,
      onError?: (error: string) => void,
      successMessage?: string
    ) => {
      openInterest.collect(
        "/api/open-interest/bulk-collect",
        "OIデータを収集しますか？",
        successMessage,
        onSuccess,
        onError
      );
    },
    [openInterest]
  );

  return {
    /** OHLCVデータ収集関連の状態と操作関数 */
    ohlcv: {
      /** OHLCVデータ収集中のローディング状態 */
      loading: ohlcv.isLoading,
      /** OHLCVデータ収集のエラーメッセージ */
      error: ohlcv.error,
      /** OHLCVデータを収集する関数 */
      collect: collectOHLCVData,
    },
    /** ファンディングレートデータ収集関連の状態と操作関数 */
    fundingRate: {
      /** ファンディングレートデータ収集中のローディング状態 */
      loading: fundingRate.isLoading,
      /** ファンディングレートデータ収集のエラーメッセージ */
      error: fundingRate.error,
      /** ファンディングレートデータを収集する関数 */
      collect: collectFundingRateData,
    },
    /** オープンインタレストデータ収集関連の状態と操作関数 */
    openInterest: {
      /** オープンインタレストデータ収集中のローディング状態 */
      loading: openInterest.isLoading,
      /** オープンインタレストデータ収集のエラーメッセージ */
      error: openInterest.error,
      /** オープンインタレストデータを収集する関数 */
      collect: collectOpenInterestData,
    },
    /** いずれかのデータ収集中かどうか */
    isAnyLoading:
      ohlcv.isLoading || fundingRate.isLoading || openInterest.isLoading,
    /** いずれかのデータ収集でエラーが発生しているかどうか */
    hasAnyError: !!(ohlcv.error || fundingRate.error || openInterest.error),
  };
};
