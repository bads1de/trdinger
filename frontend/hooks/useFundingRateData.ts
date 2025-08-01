import { useCallback } from "react";
import { FundingRateData } from "@/types/funding-rate";
import { useParameterizedDataFetching } from "./useDataFetching";

interface FundingRateParams {
  symbol: string;
  limit: number;
  start_date?: string;
  end_date?: string;
}

/**
 * ファンディングレートデータ管理フック
 *
 * 指定されたシンボルのファンディングレートデータを取得・管理します。
 * データの取得、表示件数の変更、パラメータ設定などの機能を提供します。
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
 * } = useFundingRateData('BTC/USDT:USDT', 100);
 *
 * // 表示件数を変更
 * setLimit(200);
 *
 * // 手動で再取得
 * refetch();
 * ```
 *
 * @param {string} symbol - 取得するシンボル
 * @param {number} [initialLimit=100] - 初期表示件数
 * @returns {{
 *   data: FundingRateData[],
 *   loading: boolean,
 *   error: string | null,
 *   refetch: () => Promise<void>,
 *   setLimit: (newLimit: number) => void,
 *   limit: number
 * }} ファンディングレートデータ管理関連の状態と操作関数
 */
export const useFundingRateData = (symbol: string, initialLimit = 100) => {
  const {
    data,
    loading,
    error,
    params,
    setParams,
    refetch,
  } = useParameterizedDataFetching<FundingRateData, FundingRateParams>(
    "/api/funding-rates/",
    { symbol, limit: initialLimit },
    {
      transform: (response: any) => {
        const fundingRateData = response.data?.funding_rates || [];
        return fundingRateData.map((rate: any) => ({
          symbol: rate.symbol,
          funding_rate: Number(rate.funding_rate),
          funding_timestamp: rate.funding_timestamp,
          timestamp: rate.timestamp,
          next_funding_timestamp: rate.next_funding_timestamp || null,
          mark_price: rate.mark_price ? Number(rate.mark_price) : null,
          index_price: rate.index_price ? Number(rate.index_price) : null,
        }));
      },
      dependencies: [symbol],
      errorMessage: "資金調達率データの取得に失敗しました",
    }
  );

  const setLimit = useCallback(
    (newLimit: number) => {
      setParams({ limit: newLimit });
    },
    [setParams]
  );

  return {
    /** ファンディングレートデータの配列 */
    data,
    /** データ取得中のローディング状態 */
    loading,
    /** エラーメッセージ */
    error,
    /** データを再取得する関数 */
    refetch,
    /** 表示件数を設定する関数 */
    setLimit,
    /** 現在の表示件数 */
    limit: params.limit,
  };
};
