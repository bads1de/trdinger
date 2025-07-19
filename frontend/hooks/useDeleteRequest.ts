import { useState, useCallback } from "react";
import { BACKEND_API_URL } from "@/constants";

type DeleteRequestState<T> = {
  data: T | null;
  error: string | null;
  isLoading: boolean;
};

export const useDeleteRequest = <T,>() => {
  const [state, setState] = useState<DeleteRequestState<T>>({
    data: null,
    error: null,
    isLoading: false,
  });

  const sendDeleteRequest = useCallback(async (endpoint: string) => {
    setState({ data: null, error: null, isLoading: true });

    try {
      const response = await fetch(`${BACKEND_API_URL}${endpoint}`, {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const responseData = await response.json();

      if (!response.ok) {
        throw new Error(responseData.message || "削除リクエストに失敗しました");
      }

      setState({ data: responseData, error: null, isLoading: false });
      return { success: true, data: responseData };
    } catch (error: any) {
      const errorMessage = error.message || "予期せぬエラーが発生しました";
      setState({ data: null, error: errorMessage, isLoading: false });
      return { success: false, error: errorMessage };
    }
  }, []);

  return { ...state, sendDeleteRequest };
};