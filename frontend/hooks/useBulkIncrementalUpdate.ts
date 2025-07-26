import { useApiCall } from "./useApiCall";
import { useCallback } from "react";
import { TimeFrame } from "@/types/market-data";
import { BulkIncrementalUpdateResponse } from "@/types/data-collection";

interface BulkIncrementalUpdateOptions {
  onSuccess?: (data: BulkIncrementalUpdateResponse) => void;
  onError?: (error: string) => void;
}

/**
 * 一括差分更新フック
 *
 * OHLCV、ファンディングレート、オープンインタレストの
 * 差分データを一括で取得する機能を提供します。
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

  return { bulkUpdate, loading, error };
};
