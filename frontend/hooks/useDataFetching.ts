/**
 * 汎用データ取得フック
 *
 * データ取得ロジックを抽象化し、API エンドポイントからのデータ取得、
 * 状態管理、エラーハンドリングを統一的に提供します。
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { useApiCall } from "./useApiCall";
import { BACKEND_API_URL } from "@/constants";

/**
 * データ取得設定オプション
 */
export interface DataFetchingOptions<TData, TParams = Record<string, any>> {
  /** APIエンドポイントURL */
  endpoint: string;
  /** HTTPメソッド（デフォルト: GET） */
  method?: "GET" | "POST" | "PUT" | "DELETE" | "PATCH";
  /** 初期パラメータ */
  initialParams?: TParams;
  /** データ変換関数 */
  transform?: (response: any) => TData[];
  /** データ抽出パス（例: "data.items" または "results"） */
  dataPath?: string;
  /** 自動初期取得を無効にする */
  disableAutoFetch?: boolean;
  /** 依存関係（これらが変更されたときに再取得） */
  dependencies?: any[];
  /** エラー時のデフォルトメッセージ */
  errorMessage?: string;
  /** 成功時のコールバック */
  onSuccess?: (data: TData[]) => void;
  /** エラー時のコールバック */
  onError?: (error: string) => void;
}

/**
 * データ取得結果
 */
export interface DataFetchingResult<TData, TParams = Record<string, any>> {
  /** 取得されたデータ */
  data: TData[];
  /** ローディング状態 */
  loading: boolean;
  /** エラーメッセージ */
  error: string | null;
  /** 現在のパラメータ */
  params: TParams;
  /** パラメータを更新する関数 */
  setParams: (params: Partial<TParams>) => void;
  /** データを再取得する関数 */
  refetch: () => Promise<void>;
  /** 状態をリセットする関数 */
  reset: () => void;
  /** 手動でデータを設定する関数 */
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
      // エンドポイントを絶対パスに変換（相対パスの場合）
      const absoluteEndpoint = endpoint.startsWith("/api/")
        ? `${BACKEND_API_URL}${endpoint}`
        : endpoint;

      // URLを構築
      const searchParams = buildSearchParams(params as Record<string, any>);
      const url = searchParams
        ? `${absoluteEndpoint}?${searchParams}`
        : absoluteEndpoint;

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
