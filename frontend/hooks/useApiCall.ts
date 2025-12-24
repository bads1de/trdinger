/**
 * API呼び出し用カスタムフック
 *
 * 汎用的なAPI呼び出し機能を提供し、ローディング状態、エラーハンドリング、
 * 成功時のコールバックを統一的に管理します。
 *
 */

import { useState, useCallback } from "react";
import { BACKEND_API_URL } from "@/constants";
import { toast } from "sonner";

/**
 * API呼び出しの設定オプション
 */
export interface ApiCallOptions {
  method?: "GET" | "POST" | "PUT" | "DELETE" | "PATCH";
  headers?: Record<string, string>;
  body?: any;
  confirmMessage?: string;
  successMessage?: string;
  onSuccess?: (data: any) => void;
  onError?: (error: string) => void;
  onFinally?: () => void;
}

/**
 * API呼び出しの結果
 */
export interface ApiCallResult<T = any> {
  loading: boolean;
  error: string | null;
  execute: (url: string, options?: ApiCallOptions) => Promise<T | null>;
  reset: () => void;
}

/**
 * API呼び出し用カスタムフック
 */
export const useApiCall = <T = any>(): ApiCallResult<T> => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const execute = useCallback(
    async (url: string, options: ApiCallOptions = {}): Promise<T | null> => {
      const {
        method = "GET",
        headers = { "Content-Type": "application/json" },
        body,
        confirmMessage,
        successMessage,
        onSuccess,
        onError,
        onFinally,
      } = options;

      // 確認ダイアログがある場合は表示
      if (confirmMessage && !confirm(confirmMessage)) {
        return null;
      }

      try {
        setLoading(true);
        setError(null);

        const requestOptions: RequestInit = {
          method,
          headers,
        };

        // ボディがある場合は追加
        if (body && method !== "GET") {
          requestOptions.body =
            typeof body === "string" ? body : JSON.stringify(body);
        }

        const fullUrl = url.startsWith("/") ? `${BACKEND_API_URL}${url}` : url;

        const response = await fetch(fullUrl, requestOptions);
        const responseText = await response.text();

        let result;

        try {
          result = JSON.parse(responseText);
        } catch (parseError) {
          console.error("JSON Parse Error:", parseError);
          console.error("Response was not valid JSON:", responseText);
          const jsonErrorMsg = `レスポンスが無効なJSON形式です: ${responseText.substring(
            0,
            100
          )}...`;
          setError(jsonErrorMsg);
          toast.error(jsonErrorMsg);
          onError?.(jsonErrorMsg);
          return null;
        }

        if (response.ok && (method === "GET" || result.success)) {
          if (successMessage) {
            toast.success(successMessage);
          }
          onSuccess?.(result);
          return result;
        } else {
          const errorMessage =
            result.message ||
            result.error ||
            `API呼び出しに失敗しました (${response.status})`;
          console.error("API Call Error:", {
            url,
            method,
            status: response.status,
            statusText: response.statusText,
            result,
            errorMessage,
            responseText,
            requestBody: options.body,
          });

          // 422エラーの場合は詳細なバリデーションエラーを表示
          if (response.status === 422) {
            console.error("Validation Error Details:", result);
            if (result.detail && Array.isArray(result.detail)) {
              const validationErrors = result.detail
                .map((err: any) => `${err.loc?.join(".")} - ${err.msg}`)
                .join(", ");
              console.error("Validation Errors:", validationErrors);
            }
          }

          setError(errorMessage);
          toast.error(errorMessage);
          onError?.(errorMessage);
          return null;
        }
      } catch (err) {
        const errorMessage =
          err instanceof Error
            ? err.message
            : "API呼び出し中にエラーが発生しました";

        setError(errorMessage);
        toast.error(errorMessage);
        onError?.(errorMessage);

        return null;
      } finally {
        setLoading(false);
        onFinally?.();
      }
    },
    []
  );

  const reset = useCallback(() => {
    setLoading(false);
    setError(null);
  }, []);

  return {
    loading,
    error,
    execute,
    reset,
  };
};
