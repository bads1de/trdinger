import { useCallback } from "react";

/**
 * 汎用的なlimit設定フック
 *
 * useParameterizedDataFetchingのsetParamsをラップして、
 * limitパラメータを更新する関数を生成します。
 *
 * @example
 * ```tsx
 * const setLimit = useSetLimit(setParams);
 * setLimit(200); // setParams({ limit: 200 })
 * ```
 *
 * @param setParams - useParameterizedDataFetchingから取得したsetParams関数
 * @returns limitを設定する関数
 */
export const useSetLimit = (
  setParams: (params: Record<string, unknown>) => void
): ((newLimit: number) => void) => {
  return useCallback(
    (newLimit: number) => {
      setParams({ limit: newLimit });
    },
    [setParams]
  );
};

/**
 * 単一オブジェクトを配列にラップするtransform関数
 *
 * useDataFetchingで単一オブジェクトを返すAPIレスポンスを
 * 配列形式に変換するためのユーティリティです。
 *
 * @example
 * ```tsx
 * useDataFetching({
 *   endpoint: "/api/status",
 *   transform: wrapInArray, // (response) => [response] と同じ
 * });
 * ```
 *
 * @param response - APIレスポンス
 * @returns レスポンスを1要素として含む配列
 */
export const wrapInArray = <T>(response: T): T[] => [response];
