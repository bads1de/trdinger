import { usePostRequest } from "./usePostRequest";
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
  const {
    sendPostRequest,
    isLoading,
    error,
  } = usePostRequest<BulkIncrementalUpdateResponse>();

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

      const { success, data, error: requestError } = await sendPostRequest(url);

      if (success && data) {
        onSuccess?.(data);
      } else {
        onError?.(requestError || "一括差分更新に失敗しました");
      }
    },
    [sendPostRequest]
  );

  return { bulkUpdate, loading: isLoading, error };
};
