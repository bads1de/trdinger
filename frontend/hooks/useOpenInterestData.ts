import { OpenInterestData } from "@/types/open-interest";
import { useMarketDataFetching } from "./useMarketDataFetching";

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
  return useMarketDataFetching<OpenInterestData>(
    "/api/open-interest/",
    { symbol, limit: initialLimit },
    {
      dataPath: "data.open_interest",
      dependencies: [symbol],
      errorMessage: "オープンインタレストデータの取得に失敗しました",
    }
  );
};