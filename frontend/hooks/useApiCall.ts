/**
 * API呼び出し用カスタムフック
 *
 * 汎用的なAPI呼び出し機能を提供し、ローディング状態、エラーハンドリング、
 * 成功時のコールバックを統一的に管理します。
 *
 */

import { useState, useCallback } from "react";

/**
 * API呼び出しの設定オプション
 */
export interface ApiCallOptions {
  method?: "GET" | "POST" | "PUT" | "DELETE" | "PATCH";
  headers?: Record<string, string>;
  body?: any;
  confirmMessage?: string;
  onSuccess?: (data: any) => void;
  onError?: (error: string) => void;
}

/**
 * API呼び出しの結果
 */
export interface ApiCallResult<T = any> {
  /** ローディング状態 */
  loading: boolean;
  /** エラーメッセージ */
  error: string | null;
  /** API呼び出し関数 */
  execute: (url: string, options?: ApiCallOptions) => Promise<T | null>;
  /** 状態をリセットする関数 */
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
        onSuccess,
        onError,
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

        const response = await fetch(url, requestOptions);
        const responseText = await response.text();

        let result;

        try {
          result = JSON.parse(responseText);
        } catch (parseError) {
          console.error("JSON Parse Error:", parseError);
          console.error("Response was not valid JSON:", responseText);
          setError(
            `レスポンスが無効なJSON形式です: ${responseText.substring(
              0,
              100
            )}...`
          );
          onError?.(
            `レスポンスが無効なJSON形式です: ${responseText.substring(
              0,
              100
            )}...`
          );
          return null;
        }

        if (response.ok && (method === "GET" || result.success)) {
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
          });
          setError(errorMessage);
          onError?.(errorMessage);
          return null;
        }
      } catch (err) {
        const errorMessage =
          err instanceof Error
            ? err.message
            : "API呼び出し中にエラーが発生しました";

        setError(errorMessage);
        onError?.(errorMessage);

        return null;
      } finally {
        setLoading(false);
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
