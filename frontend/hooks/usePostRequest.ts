import { useState, useCallback } from "react";
import { useApiCall } from "./useApiCall";

type PostRequestState<T> = {
  data: T | null;
  error: string | null;
  isLoading: boolean;
};

export const usePostRequest = <T, TBody = Record<string, any>>() => {
  const [data, setData] = useState<T | null>(null);
  const { execute, loading, error } = useApiCall<T>();

  const sendPostRequest = useCallback(
    async (endpoint: string, body?: TBody) => {
      setData(null);

      const result = await execute(endpoint, {
        method: "POST",
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
    sendPostRequest,
  };
};
