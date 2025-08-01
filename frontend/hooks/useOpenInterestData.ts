import { useCallback } from "react";
import { OpenInterestData } from "@/types/open-interest";
import { useParameterizedDataFetching } from "./useDataFetching";

/**
 * オープンインタレストパラメータインターフェース
 *
 * オープンインタレストデータ取得時のパラメータを定義します。
 */
interface OpenInterestParams {
  /** 取引シンボル */
  symbol: string;
  /** 取得するデータ数の上限 */
  limit: number;
  /** 開始日（オプション） */
  start_date?: string;
  /** 終了日（オプション） */
  end_date?: string;
}

/**
 * オープンインタレストデータ取得フック
 *
 * 指定されたシンボルのオープンインタレストデータを取得します。
 * データの再取得や取得数の変更も可能です。
 *
 * @example
 * ```tsx
 * const {
 *   data,
 *   loading,
 *   error,
 *   refetch,
 *   setLimit,
 *   limit
 * } = useOpenInterestData('BTC/USDT:USDT', 100);
 *
 * // データを再取得
 * refetch();
 *
 * // 取得数を変更
 * setLimit(200);
 * ```
 *
 * @param {string} symbol - 取引シンボル（例: 'BTC/USDT:USDT'）
 * @param {number} initialLimit - 初期取得数（デフォルト: 100）
 * @returns {{
 *   data: OpenInterestData[],
 *   loading: boolean,
 *   error: string | null,
 *   refetch: () => Promise<void>,
 *   setLimit: (limit: number) => void,
 *   limit: number
 * }} オープンインタレストデータ取得関連の状態と操作関数
 */
export const useOpenInterestData = (symbol: string, initialLimit = 100) => {
  const {
    data,
    loading,
    error,
    params,
    setParams,
    refetch,
  } = useParameterizedDataFetching<OpenInterestData, OpenInterestParams>(
    "/api/open-interest/",
    { symbol, limit: initialLimit },
    {
      dataPath: "data.open_interest",
      dependencies: [symbol],
      errorMessage: "オープンインタレストデータの取得に失敗しました",
    }
  );

  /**
   * 取得数を設定
   *
   * @param {number} newLimit - 新しい取得数
   */
  const setLimit = useCallback(
    (newLimit: number) => {
      setParams({ limit: newLimit });
    },
    [setParams]
  );

  return {
    /** オープンインタレストデータ */
    data,
    /** ローディング状態 */
    loading,
    /** エラーメッセージ */
    error,
    /** データを再取得する関数 */
    refetch,
    /** 取得数を設定する関数 */
    setLimit,
    /** 現在の取得数 */
    limit: params.limit,
  };
};
