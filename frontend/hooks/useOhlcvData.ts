import { useCallback } from "react";
import { PriceData, TimeFrame } from "@/types/market-data";
import { useParameterizedDataFetching } from "./useDataFetching";

/**
 * OHLCVパラメータインターフェース
 *
 * OHLCVデータ取得時のパラメータを定義します。
 */
interface OhlcvParams {
  /** 取引シンボル */
  symbol: string;
  /** 時間枠 */
  timeframe: TimeFrame;
  /** 取得するデータ数の上限 */
  limit: number;
  /** 開始日（オプション） */
  start_date?: string;
  /** 終了日（オプション） */
  end_date?: string;
}

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
  const { data, loading, error, params, setParams, refetch } =
    useParameterizedDataFetching<PriceData, OhlcvParams>(
      "/api/market-data/ohlcv",
      { symbol, timeframe, limit: initialLimit },
      {
        transform: (response: any) => {
          const ohlcvData = response.data?.ohlcv_data || [];

          if (!Array.isArray(ohlcvData)) {
            console.error("OHLCV data is not an array:", ohlcvData);
            return [];
          }

          return ohlcvData.map((candle: number[]) => {
            const [timestamp, open, high, low, close, volume] = candle;

            return {
              timestamp: new Date(timestamp).toISOString(),
              open: Number(open.toFixed(2)),
              high: Number(high.toFixed(2)),
              low: Number(low.toFixed(2)),
              close: Number(close.toFixed(2)),
              volume: Number(volume.toFixed(2)),
            };
          });
        },
        dependencies: [symbol, timeframe],
        errorMessage: "OHLCVデータの取得に失敗しました",
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
    /** OHLCVデータ */
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
