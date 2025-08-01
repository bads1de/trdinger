import { useState, useCallback, useEffect } from "react";
import { useApiCall } from "./useApiCall";
import { BACKEND_API_URL } from "@/constants";

export interface DataStatusResponse {
  success?: boolean;
  [key: string]: any;
}

/**
 * データステータス管理フック
 *
 * データベース内の各種データのステータス情報を取得・管理します。
 * OHLCV、ファンディングレート、オープンインタレスト、Fear & Greed Indexなどの
 * データ件数や最新タイムスタンプなどの情報を提供します。
 *
 * @example
 * ```tsx
 * const {
 *   dataStatus,
 *   dataStatusLoading,
 *   dataStatusError,
 *   fetchDataStatus
 * } = useDataStatus();
 *
 * // データステータスを再取得
 * fetchDataStatus();
 *
 * // ローディング状態を表示
 * if (dataStatusLoading) {
 *   return <LoadingSpinner />;
 * }
 *
 * // データステータスを表示
 * if (dataStatus) {
 *   console.log('OHLCV件数:', dataStatus.data_counts.ohlcv);
 * }
 * ```
 *
 * @returns {{
 *   dataStatus: DataStatusResponse | null,
 *   dataStatusLoading: boolean,
 *   dataStatusError: string | null,
 *   fetchDataStatus: () => Promise<void>
 * }} データステータス管理関連の状態と操作関数
 */
export const useDataStatus = () => {
  const [dataStatus, setDataStatus] = useState<DataStatusResponse | null>(null);
  const {
    execute: fetchDataStatusApi,
    loading: dataStatusLoading,
    error: dataStatusError,
  } = useApiCall<DataStatusResponse>();

  const fetchDataStatus = useCallback(() => {
    const url = `${BACKEND_API_URL}/api/data-reset/status`;
    fetchDataStatusApi(url, {
      onSuccess: (result) => {
        if (result) {
          setDataStatus(result);
        }
      },
      onError: (err) => {
        console.error("データ状況取得エラー:", err);
      },
    });
  }, [fetchDataStatusApi]);

  useEffect(() => {
    fetchDataStatus();
  }, [fetchDataStatus]);

  return {
    /** データステータス情報 */
    dataStatus,
    /** データステータス取得中のローディング状態 */
    dataStatusLoading,
    /** データステータス取得のエラーメッセージ */
    dataStatusError: dataStatusError || null,
    /** データステータスを再取得する関数 */
    fetchDataStatus,
  };
};
