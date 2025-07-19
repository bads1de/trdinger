import { useState, useCallback } from "react";
import { BACKEND_API_URL } from "@/constants";

type PostRequestState<T> = {
  data: T | null;
  error: string | null;
  isLoading: boolean;
};

export const usePostRequest = <T, TBody = Record<string, any>>() => {
  const [state, setState] = useState<PostRequestState<T>>({
    data: null,
    error: null,
    isLoading: false,
  });

  const sendPostRequest = useCallback(async (endpoint: string, body?: TBody) => {
    setState({ data: null, error: null, isLoading: true });

    try {
      const response = await fetch(`${BACKEND_API_URL}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: body ? JSON.stringify(body) : null,
      });

      const responseData = await response.json();

      if (!response.ok) {
        throw new Error(responseData.message || "リクエストに失敗しました");
      }

      setState({ data: responseData, error: null, isLoading: false });
      return { success: true, data: responseData };
    } catch (error: any) {
      const errorMessage = error.message || "予期せぬエラーが発生しました";
      setState({ data: null, error: errorMessage, isLoading: false });
      return { success: false, error: errorMessage };
    }
  }, []);

  return { ...state, sendPostRequest };
};