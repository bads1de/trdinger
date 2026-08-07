import { PriceData, TimeFrame } from "@/types/market-data";
import { useMarketDataFetching } from "./useMarketDataFetching";

/**
 * OHLCVデータ取得フック
 *
 * 指定されたシンボルと時間枠のOHLCV（Open, High, Low, Close, Volume）データを取得します。
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
 * } = useOhlcvData('BTC/USDT:USDT', '1h', 100);
 *
 * // データを再取得
 * refetch();
 *
 * // 取得数を変更
 * setLimit(200);
 * ```
 *
 * @param {string} symbol - 取引シンボル（例: 'BTC/USDT:USDT'）
 * @param {TimeFrame} timeframe - 時間枠（例: '1h', '4h', '1d'）
 * @param {number} initialLimit - 初期取得数（デフォルト: 100）
 * @returns {{
 *   data: PriceData[],
 *   loading: boolean,
 *   error: string | null,
 *   refetch: () => Promise<void>,
 *   setLimit: (limit: number) => void,
 *   limit: number
 * }} OHLCVデータ取得関連の状態と操作関数
 */
export const useOhlcvData = (
  symbol: string,
  timeframe: TimeFrame,
  initialLimit = 100
) => {
  return useMarketDataFetching<PriceData>(
    "/api/market-data/ohlcv",
    { symbol, timeframe, limit: initialLimit },
    {
      transform: (response: any) => {
        const ohlcvData = response.data?.ohlcv_data || [];

        if (!Array.isArray(ohlcvData)) {
          console.error("OHLCV data is not an array:", ohlcvData);
          return [];
        }

        const safeNumber = (val: unknown, fallback: number = 0): number => {
          if (val == null || typeof val !== 'number') return fallback;
          return Number(val.toFixed(2));
        };

        return ohlcvData
          .map((candle: number[]) => {
            const [timestamp, open, high, low, close, volume] = candle;

            return {
              timestamp: timestamp != null ? new Date(timestamp).toISOString() : null,
              open: safeNumber(open),
              high: safeNumber(high),
              low: safeNumber(low),
              close: safeNumber(close),
              volume: safeNumber(volume),
            };
          })
          .filter((candle): candle is typeof candle & { timestamp: string } => candle.timestamp !== null);
      },
      dependencies: [symbol, timeframe],
      errorMessage: "OHLCVデータの取得に失敗しました",
    }
  );
};