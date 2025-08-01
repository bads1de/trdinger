/**
 * 汎用データ取得フック
 *
 * データ取得ロジックを抽象化し、API エンドポイントからのデータ取得、
 * 状態管理、エラーハンドリングを統一的に提供します。
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { useApiCall } from "./useApiCall";

/**
 * データ取得設定オプション
 */
export interface DataFetchingOptions<TData, TParams = Record<string, any>> {
  endpoint: string;
  method?: "GET" | "POST" | "PUT" | "DELETE" | "PATCH";
  initialParams?: TParams;
  transform?: (response: any) => TData[];
  dataPath?: string;
  disableAutoFetch?: boolean;
  dependencies?: any[];
  errorMessage?: string;
  onSuccess?: (data: TData[]) => void;
  onError?: (error: string) => void;
}

/**
 * データ取得結果
 */
export interface DataFetchingResult<TData, TParams = Record<string, any>> {
  data: TData[];
  loading: boolean;
  error: string | null;
  params: TParams;
  setParams: (params: Partial<TParams>) => void;
  refetch: () => Promise<void>;
  reset: () => void;
  setData: (data: TData[]) => void;
}

/**
 * オブジェクトから指定されたパスの値を取得
 */
const getNestedValue = (obj: any, path: string): any => {
  return path.split(".").reduce((current, key) => current?.[key], obj);
};

/**
 * URLSearchParamsを構築
 */
const buildSearchParams = (params: Record<string, any>): string => {
  const searchParams = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      if (Array.isArray(value)) {
        value.forEach((item) => searchParams.append(key, String(item)));
      } else {
        searchParams.set(key, String(value));
      }
    }
  });

  return searchParams.toString();
};

/**
 * 汎用データ取得フック
 *
 * データ取得ロジックを抽象化し、API エンドポイントからのデータ取得、
 * 状態管理、エラーハンドリングを統一的に提供します。
 * パラメータ変更に応じた自動再取得や、データ変換機能をサポートします。
 *
 * @example
 * ```tsx
 * const {
 *   data,
 *   loading,
 *   error,
 *   params,
 *   setParams,
 *   refetch
 * } = useDataFetching<User, { limit: number; offset: number }>({
 *   endpoint: "/api/users",
 *   initialParams: { limit: 20, offset: 0 },
 *   transform: (response) => response.users,
 *   errorMessage: "ユーザーデータの取得に失敗しました"
 * });
 *
 * // パラメータを更新
 * setParams({ limit: 50 });
 *
 * // 手動で再取得
 * refetch();
 * ```
 *
 * @typeParam TData - 取得するデータの型
 * @typeParam TParams - クエリパラメータの型
 * @param {DataFetchingOptions<TData, TParams>} options - データ取得設定オプション
 * @returns {DataFetchingResult<TData, TParams>} データ取得関連の状態と操作関数
 */
export const useDataFetching = <TData = any, TParams = Record<string, any>>(
  options: DataFetchingOptions<TData, TParams>
): DataFetchingResult<TData, TParams> => {
  const {
    endpoint,
    method = "GET",
    initialParams = {} as TParams,
    transform,
    dataPath,
    disableAutoFetch = false,
    dependencies = [],
    errorMessage = "データの取得中にエラーが発生しました",
    onSuccess,
    onError,
  } = options;

  const [data, setData] = useState<TData[]>([]);
  const [params, setParamsState] = useState<TParams>(initialParams);
  const { execute, loading, error, reset: resetApiCall } = useApiCall();

  // 無限ループを防ぐためのref
  const lastParamsRef = useRef<string>("");
  const lastDependenciesRef = useRef<string>("");

  /**
   * データを取得する関数
   */
  const fetchData = useCallback(async () => {
    try {
      // URLを構築
      const searchParams = buildSearchParams(params as Record<string, any>);
      const url = searchParams ? `${endpoint}?${searchParams}` : endpoint;

      const response = await execute(url, {
        method,
        onSuccess: (result) => {
          let extractedData: TData[] = [];

          if (transform) {
            // カスタム変換関数を使用
            extractedData = transform(result);
          } else if (dataPath) {
            // 指定されたパスからデータを抽出
            const nestedData = getNestedValue(result, dataPath);
            extractedData = Array.isArray(nestedData) ? nestedData : [];
          } else if (result.success && result.data) {
            // デフォルト: result.data から抽出
            if (Array.isArray(result.data)) {
              extractedData = result.data;
            } else if (typeof result.data === "object") {
              // data オブジェクト内の最初の配列を探す
              const firstArrayValue = Object.values(result.data).find((value) =>
                Array.isArray(value)
              );
              extractedData = firstArrayValue || [];
            }
          }

          setData(extractedData);
          onSuccess?.(extractedData);
        },
        onError: (errorMsg) => {
          console.error(`${errorMessage}:`, errorMsg);
          onError?.(errorMsg);
        },
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : errorMessage;
      console.error(`${errorMessage}:`, errorMsg);
      onError?.(errorMsg);
    }
  }, [
    endpoint,
    method,
    params,
    transform,
    dataPath,
    execute,
    errorMessage,
    onSuccess,
    onError,
  ]);

  /**
   * パラメータを更新する関数
   */
  const setParams = useCallback((newParams: Partial<TParams>) => {
    setParamsState((prev) => ({ ...prev, ...newParams }));
  }, []);

  /**
   * 状態をリセットする関数
   */
  const reset = useCallback(() => {
    setData([]);
    setParamsState(initialParams);
    resetApiCall();
  }, [initialParams, resetApiCall]);

  /**
   * 依存関係とパラメータの変更を監視して自動取得
   */
  useEffect(() => {
    if (disableAutoFetch) return;

    const currentParams = JSON.stringify(params);
    const currentDependencies = JSON.stringify(dependencies);

    // パラメータまたは依存関係が変更された場合のみ実行
    if (
      currentParams !== lastParamsRef.current ||
      currentDependencies !== lastDependenciesRef.current
    ) {
      lastParamsRef.current = currentParams;
      lastDependenciesRef.current = currentDependencies;

      // 必須パラメータがある場合はチェック
      const hasRequiredParams =
        dependencies.length === 0 ||
        dependencies.every(
          (dep) => dep !== undefined && dep !== null && dep !== ""
        );

      if (hasRequiredParams) {
        fetchData();
      }
    }
  }, [params, dependencies, disableAutoFetch, fetchData]);

  return {
    data,
    loading,
    error,
    params,
    setParams,
    refetch: fetchData,
    reset,
    setData,
  };
};

/**
 * 簡単なデータ取得フック（基本的なGETリクエスト用）
 *
 * エンドポイントとオプションを指定するだけで、簡単にデータ取得ができるフックです。
 * 基本的なGETリクエストに特化しており、最小限の設定で利用できます。
 *
 * @example
 * ```tsx
 * const { data, loading, error } = useSimpleDataFetching<User>("/api/users", {
 *   errorMessage: "ユーザーデータの取得に失敗しました"
 * });
 * ```
 *
 * @typeParam TData - 取得するデータの型
 * @param {string} endpoint - APIエンドポイント
 * @param {Partial<DataFetchingOptions<TData>>} options - データ取得オプション
 * @returns {DataFetchingResult<TData>} データ取得関連の状態と操作関数
 */
export const useSimpleDataFetching = <TData = any>(
  endpoint: string,
  options?: Partial<DataFetchingOptions<TData>>
) => {
  return useDataFetching<TData>({
    endpoint,
    ...options,
  });
};

/**
 * パラメータ付きデータ取得フック
 *
 * 初期パラメータを指定してデータ取得を行うフックです。
 * ページネーションやフィルタリングなど、パラメータを動的に変更する場合に便利です。
 *
 * @example
 * ```tsx
 * const { data, loading, error, params, setParams } = useParameterizedDataFetching<User, { limit: number; offset: number }>(
 *   "/api/users",
 *   { limit: 20, offset: 0 },
 *   {
 *     transform: (response) => response.users,
 *     errorMessage: "ユーザーデータの取得に失敗しました"
 *   }
 * );
 *
 * // ページネーション
 * const nextPage = () => setParams({ limit: 20, offset: params.offset + 20 });
 * ```
 *
 * @typeParam TData - 取得するデータの型
 * @typeParam TParams - クエリパラメータの型
 * @param {string} endpoint - APIエンドポイント
 * @param {TParams} initialParams - 初期パラメータ
 * @param {Partial<DataFetchingOptions<TData, TParams>>} options - データ取得オプション
 * @returns {DataFetchingResult<TData, TParams>} データ取得関連の状態と操作関数
 */
export const useParameterizedDataFetching = <
  TData = any,
  TParams = Record<string, any>
>(
  endpoint: string,
  initialParams: TParams,
  options?: Partial<DataFetchingOptions<TData, TParams>>
) => {
  return useDataFetching<TData, TParams>({
    endpoint,
    initialParams,
    ...options,
  });
};
