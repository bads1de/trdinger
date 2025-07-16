import { useState, useEffect, useCallback } from "react";
import { useApiCall } from "./useApiCall";
import { DataResetResult } from "@/components/button/DataResetButton";

interface DataStatus {
  data_counts: {
    ohlcv: number;
    funding_rates: number;
    open_interest: number;
  };
  total_records: number;
  timestamp: string;
}

export const useDataReset = (isVisible: boolean) => {
  const [dataStatus, setDataStatus] = useState<DataStatus | null>(null);
  const [resetMessage, setResetMessage] = useState<string>("");
  const {
    execute: fetchStatusApi,
    loading: isLoading,
    reset,
  } = useApiCall<DataStatus>();

  const fetchDataStatus = useCallback(async () => {
    reset();
    const result = await fetchStatusApi("/api/data-reset/status", {
      method: "GET",
    });

    if (result) {
      setDataStatus(result);
    }
  }, [fetchStatusApi, reset]);

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

  useEffect(() => {
    if (isVisible) {
      fetchDataStatus();
    }
  }, [isVisible, fetchDataStatus]);

  return {
    dataStatus,
    resetMessage,
    isLoading,
    fetchDataStatus,
    handleResetComplete,
    handleResetError,
  };
};
