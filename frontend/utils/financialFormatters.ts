import { formatLargeNumber } from "./formatters";

/**
 * 通貨の値をフォーマットする
 * @param value - フォーマットする数値
 * @returns フォーマットされた通貨文字列 (例: '$1,234.5')
 * @example
 * // returns '$1,234.5'
 * formatCurrency(1234.56)
 */
export const formatCurrency = (value: number) => {
  return new Intl.NumberFormat("ja-JP", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 1,
  }).format(value);
};

/**
 * 価格をフォーマットする
 * @param value - フォーマットする価格 (nullの場合は'-'を返す)
 * @returns フォーマットされた価格文字列 (例: '1234.56') または '-'
 * @example
 * // returns '1234.56'
 * formatPrice(1234.56)
 */
export const formatPrice = (value: number | null) => {
  if (value === null) return "-";

  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
};

/**
 * シンボルをフォーマットする (例: 'BTCUSDT' -> 'BTC/USDT')
 * @param symbol - フォーマットするシンボル文字列
 * @returns フォーマットされたシンボル文字列
 * @example
 * // returns 'BTC/USDT'
 * formatSymbol('BTCUSDT')
 */
export const formatSymbol = (symbol: string) => {
  if (symbol.endsWith("USDT")) {
    return `${symbol.slice(0, -4)}/USDT`;
  }

  return symbol;
};

/**
 * ファンディングレートをパーセンテージでフォーマットする
 * @param value - ファンディングレート (0.01 = 1%)
 * @returns フォーマットされたファンディングレート文字列 (例: '0.0123%')
 * @example
 * // returns '1.2300%'
 * formatFundingRate(0.0123)
 */
export const formatFundingRate = (value: number) => {
  return `${(value * 100).toFixed(4)}%`;
};

/**
 * 取引高をフォーマットする
 * @param value - フォーマットする取引高
 * @returns フォーマットされた取引高文字列 (例: '1.23K', '4.56M')
 * @example
 * // returns '1.23K'
 * formatVolume(1234)
 */
export const formatVolume = (value: number) => {
  return formatLargeNumber(value, 2);
};

// formatLargeNumber は formatters.ts から再エクスポート
// ドキュメントは formatters.ts を参照してください
export { formatLargeNumber };
