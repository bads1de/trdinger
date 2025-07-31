import { useState, useCallback, useEffect } from "react";
import { useApiCall } from "./useApiCall";
import { BACKEND_API_URL } from "@/constants";

export interface DataStatusResponse {
  success?: boolean;
  [key: string]: any;
}

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
    dataStatus,
    dataStatusLoading,
    dataStatusError: dataStatusError || null,
    fetchDataStatus,
  };
};
