import { useState, useCallback } from "react";
import { useApiCall } from "./useApiCall";

type PutRequestState<T> = {
  data: T | null;
  error: string | null;
  isLoading: boolean;
};

export const usePutRequest = <T, TBody = Record<string, any>>() => {
  const [data, setData] = useState<T | null>(null);
  const { execute, loading, error } = useApiCall<T>();

  const sendPutRequest = useCallback(
    async (endpoint: string, body?: TBody) => {
      setData(null);

      const result = await execute(endpoint, {
        method: "PUT",
        body,
        onSuccess: (responseData) => {
          setData(responseData);
        },
      });

      if (result) {
        return { success: true, data: result };
      } else {
        return { success: false, error: error || "リクエストに失敗しました" };
      }
    },
    [execute, error]
  );

  return {
    data,
    error,
    isLoading: loading,
    sendPutRequest,
  };
};
