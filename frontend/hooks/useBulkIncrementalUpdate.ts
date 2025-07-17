import { useApiCall } from "./useApiCall";
import { useCallback } from "react";
import { BACKEND_API_URL } from "@/constants";
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
  const {
    execute: executeUpdate,
    loading,
    error,
    reset,
  } = useApiCall<BulkIncrementalUpdateResponse>();

  const bulkUpdate = useCallback(
    async (
      symbol: string,
      timeframe: TimeFrame,
      options: BulkIncrementalUpdateOptions = {}
    ) => {
      const { onSuccess, onError } = options;

      const url = `/api/data/bulk-incremental-update?symbol=${encodeURIComponent(
        symbol
      )}`;

      await executeUpdate(url, {
        method: "POST",
        onSuccess: (result) => {
          if (onSuccess) {
            onSuccess(result);
          }
        },
        onError: (errorMessage) => {
          if (onError) {
            onError(errorMessage);
          }
        },
      });
    },
    [executeUpdate]
  );

  return { bulkUpdate, loading, error, reset };
};
