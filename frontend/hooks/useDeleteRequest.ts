import { useState, useCallback } from "react";
import { useApiCall } from "./useApiCall";

type DeleteRequestState<T> = {
  data: T | null;
  error: string | null;
  isLoading: boolean;
};

export const useDeleteRequest = <T>() => {
  const [data, setData] = useState<T | null>(null);
  const { execute, loading, error } = useApiCall<T>();

  const sendDeleteRequest = useCallback(
    async (endpoint: string) => {
      setData(null);

      const result = await execute(endpoint, {
        method: "DELETE",
        onSuccess: (responseData) => {
          setData(responseData);
        },
      });

      if (result) {
        return { success: true, data: result };
      } else {
        return {
          success: false,
          error: error || "削除リクエストに失敗しました",
        };
      }
    },
    [execute, error]
  );

  return {
    data,
    error,
    isLoading: loading,
    sendDeleteRequest,
  };
};
