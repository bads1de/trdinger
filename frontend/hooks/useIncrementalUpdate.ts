import { useApiCall } from "./useApiCall";
import { useCallback } from "react";
import { BACKEND_API_URL } from "@/constants";
import { TimeFrame } from "@/types/strategy";

interface IncrementalUpdateOptions {
  onSuccess?: (data: any) => void;
  onError?: (error: string) => void;
}

export const useIncrementalUpdate = () => {
  const { execute: executeUpdate, loading, error, reset } = useApiCall();

  const update = useCallback(
    async (
      symbol: string,
      timeframe: TimeFrame,
      options: IncrementalUpdateOptions = {}
    ) => {
      const { onSuccess, onError } = options;

      const url = `${BACKEND_API_URL}/api/data-collection/update?symbol=${symbol}&timeframe=${timeframe}`;

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

  return { update, loading, error, reset };
};
