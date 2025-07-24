import { useState, useCallback } from "react";
import { useDataFetching } from "./useDataFetching";
import { DataResetResult } from "@/components/button/DataResetButton";

interface DataStatus {
  data_counts: {
    ohlcv: number;
    funding_rates: number;
    open_interest: number;
    fear_greed_index: number;
  };
  total_records: number;
  details?: {
    ohlcv?: any;
    funding_rates?: {
      count: number;
      latest_timestamp?: string;
      oldest_timestamp?: string;
    };
    open_interest?: {
      count: number;
      latest_timestamp?: string;
      oldest_timestamp?: string;
    };
    fear_greed_index?: {
      count: number;
      latest_timestamp?: string;
      oldest_timestamp?: string;
    };
  };
  timestamp: string;
}

export const useDataReset = (isVisible: boolean) => {
  const [resetMessage, setResetMessage] = useState<string>("");

  // 基本的なデータ取得は共通フックを使用
  const {
    data: dataStatusArray,
    loading: isLoading,
    error,
    refetch: fetchDataStatus,
  } = useDataFetching<DataStatus>({
    endpoint: "/api/data-reset/status",
    disableAutoFetch: !isVisible, // isVisibleがfalseの場合は自動取得しない
    dependencies: [isVisible],
    transform: (response) => [response], // 単一オブジェクトを配列に変換
    errorMessage: "データステータスの取得中にエラーが発生しました",
  });

  // 配列の最初の要素を取得（単一オブジェクトなので）
  const dataStatus = dataStatusArray.length > 0 ? dataStatusArray[0] : null;

  const handleResetComplete = useCallback(
    (result: DataResetResult) => {
      if (result.success) {
        let message = `✅ ${result.message}`;
        if (result.total_deleted !== undefined) {
          message += ` (${result.total_deleted.toLocaleString()}件削除)`;
        } else if (result.deleted_count !== undefined) {
          message += ` (${result.deleted_count.toLocaleString()}件削除)`;
        }
        setResetMessage(message);
      } else {
        setResetMessage(`❌ ${result.message}`);
      }

      setTimeout(() => {
        fetchDataStatus();
      }, 1000);

      setTimeout(() => {
        setResetMessage("");
      }, 10000);
    },
    [fetchDataStatus]
  );

  const handleResetError = useCallback((error: string) => {
    setResetMessage(`❌ ${error}`);
    setTimeout(() => {
      setResetMessage("");
    }, 10000);
  }, []);

  return {
    dataStatus,
    resetMessage,
    isLoading,
    error,
    fetchDataStatus,
    handleResetComplete,
    handleResetError,
  };
};
