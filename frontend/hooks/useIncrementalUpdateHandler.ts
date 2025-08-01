import { useCallback } from "react";
import { TimeFrame } from "@/types/market-data";
import { useBulkIncrementalUpdate } from "@/hooks/useBulkIncrementalUpdate";

export interface IncrementalUpdateDeps {
  setMessage: (key: string, message: string, duration?: number) => void;
  fetchOHLCVData: () => Promise<void> | void;
  fetchDataStatus: () => void;
  MESSAGE_KEYS: Record<string, string>;
  MESSAGE_DURATION: Record<"SHORT" | "MEDIUM" | "LONG", number>;
}

/**
 * 差分更新ハンドラーフック
 *
 * 一括差分更新機能のハンドリングを提供します。
 * 更新の実行、成功・失敗時のメッセージ表示、
 * データの再取得などの機能を統合的に管理します。
 *
 * @example
 * ```tsx
 * const {
 *   handleBulkIncrementalUpdate,
 *   bulkIncrementalUpdateLoading,
 *   bulkIncrementalUpdateError
 * } = useIncrementalUpdateHandler({
 *   setMessage,
 *   fetchOHLCVData,
 *   fetchDataStatus,
 *   MESSAGE_KEYS,
 *   MESSAGE_DURATION
 * });
 *
 * // 差分更新を実行
 * handleBulkIncrementalUpdate('BTC/USDT:USDT', '1h');
 * ```
 *
 * @param {IncrementalUpdateDeps} deps - 依存関係オブジェクト
 * @returns {{
 *   handleBulkIncrementalUpdate: (selectedSymbol: string, selectedTimeFrame: TimeFrame) => Promise<void>,
 *   bulkIncrementalUpdateLoading: boolean,
 *   bulkIncrementalUpdateError: string | null
 * }} 差分更新ハンドリング関連の状態と操作関数
 */
export const useIncrementalUpdateHandler = ({
  setMessage,
  fetchOHLCVData,
  fetchDataStatus,
  MESSAGE_KEYS,
  MESSAGE_DURATION,
}: IncrementalUpdateDeps) => {
  const {
    bulkUpdate: updateBulkIncrementalData,
    loading: bulkIncrementalUpdateLoading,
    error: bulkIncrementalUpdateError,
  } = useBulkIncrementalUpdate();

  const handleBulkIncrementalUpdate = useCallback(
    async (selectedSymbol: string, selectedTimeFrame: TimeFrame) => {
      setMessage(MESSAGE_KEYS.INCREMENTAL_UPDATE, "");
      await updateBulkIncrementalData(selectedSymbol, selectedTimeFrame, {
        onSuccess: async (result) => {
          const totalSavedCount = result.data.total_saved_count || 0;
          const ohlcvCount = result.data.data.ohlcv.saved_count || 0;
          const frCount = result.data.data.funding_rate.saved_count || 0;
          const oiCount = result.data.data.open_interest.saved_count || 0;

          let timeframeDetails = "";

          if (result.data.data.ohlcv.timeframe_results) {
            const tfResults = Object.entries(
              result.data.data.ohlcv.timeframe_results
            )
              .map(
                ([tf, res]: [string, any]) =>
                  `${tf}:${(res as any).saved_count}`
              )
              .join(", ");
            timeframeDetails = ` [${tfResults}]`;
          }

          setMessage(
            MESSAGE_KEYS.INCREMENTAL_UPDATE,
            `✅ 一括差分更新完了！ ${selectedSymbol} - 総計${totalSavedCount}件 (OHLCV:${ohlcvCount}${timeframeDetails}, FR:${frCount}, OI:${oiCount})`,
            MESSAGE_DURATION.MEDIUM
          );

          await fetchOHLCVData();

          fetchDataStatus();
        },
        onError: (errorMessage) => {
          setMessage(
            MESSAGE_KEYS.INCREMENTAL_UPDATE,
            `❌ ${errorMessage}`,
            MESSAGE_DURATION.SHORT
          );
          console.error("一括差分更新エラー:", errorMessage);
        },
      });
    },
    [
      setMessage,
      updateBulkIncrementalData,
      fetchOHLCVData,
      fetchDataStatus,
      MESSAGE_KEYS,
      MESSAGE_DURATION,
    ]
  );

  return {
    /** 一括差分更新を実行する関数 */
    handleBulkIncrementalUpdate,
    /** 一括差分更新中のローディング状態 */
    bulkIncrementalUpdateLoading,
    /** 一括差分更新のエラーメッセージ */
    bulkIncrementalUpdateError,
  };
};
