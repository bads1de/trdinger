/**
 * 一括差分更新用カスタムフック
 *
 * OHLCV、ファンディングレート、オープンインタレストの
 * 差分データを一括で取得する機能を提供します。
 */

import { useApiCall } from "./useApiCall";
import { useCallback } from "react";
import { TimeFrame } from "@/types/market-data";
import { BulkIncrementalUpdateResponse } from "@/types/data-collection";

/**
 * 一括差分更新オプションの型
 */
interface BulkIncrementalUpdateOptions {
  /** 成功時のコールバック関数 */
  onSuccess?: (data: BulkIncrementalUpdateResponse) => void;
  /** エラー時のコールバック関数 */
  onError?: (error: string) => void;
}

/**
 * 一括差分更新フック
 *
 * OHLCV、ファンディングレート、オープンインタレストの
 * 差分データを一括で取得する機能を提供します。
 *
 * @example
 * ```tsx
 * const { bulkUpdate, loading, error } = useBulkIncrementalUpdate();
 *
 * // 差分更新を実行
 * bulkUpdate('BTC/USDT:USDT', '1h', {
 *   onSuccess: (data) => console.log('更新完了:', data),
 *   onError: (error) => console.error('更新失敗:', error)
 * });
 * ```
 *
 * @returns {{
 *   bulkUpdate: (symbol: string, timeframe: TimeFrame, options?: BulkIncrementalUpdateOptions) => Promise<void>,
 *   loading: boolean,
 *   error: string | null
 * }} 一括差分更新関連の状態と操作関数
 */
export const useBulkIncrementalUpdate = () => {
  const { execute, loading, error } =
    useApiCall<BulkIncrementalUpdateResponse>();

  const bulkUpdate = useCallback(
    async (
      symbol: string,
      timeframe: TimeFrame,
      options: BulkIncrementalUpdateOptions = {}
    ) => {
      const { onSuccess, onError } = options;

      const url = `/api/data-collection/bulk-incremental-update?symbol=${encodeURIComponent(
        symbol
      )}`;

      await execute(url, {
        method: "POST",
        onSuccess,
        onError: (error) => {
          onError?.(error || "一括差分更新に失敗しました");
        },
      });
    },
    [execute]
  );

  return {
    /** 一括差分更新を実行する関数 */
    bulkUpdate,
    /** 更新中のローディング状態 */
    loading,
    /** エラーメッセージ */
    error
  };
};
