/**
 * アプリケーション全体で使用されるフォーマット関数
 */

/**
 * 日時文字列を 'YYYY/MM/DD HH:mm:ss' 形式にフォーマットします。
 * @param dateString - ISO 8601 形式の日時文字列
 * @returns フォーマットされた日時文字列
 */
export const formatDateTime = (dateString: string | number): string => {
  const date = new Date(dateString);
  if (isNaN(date.getTime())) {
    return "Invalid Date";
  }
  return date.toLocaleString("ja-JP", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
};

/**
 * 数値を価格文字列にフォーマットします。
 * @param price - 価格
 * @returns フォーマットされた価格文字列
 */
export const formatPrice = (price: number | null): string => {
  if (price === null || typeof price === "undefined") {
    return "-";
  }
  return price.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
};

/**
 * 通貨ペアのシンボルをフォーマットします (例: BTCUSDT -> BTC/USDT)。
 * @param symbol - 通貨ペアのシンボル
 * @returns フォーマットされたシンボル
 */
export const formatSymbol = (symbol: string): string => {
  if (symbol.endsWith("USDT")) {
    return `${symbol.slice(0, -4)}/USDT`;
  }
  return symbol;
};

/**
 * ファンディングレートをパーセンテージ形式でフォーマットします。
 * @param rate - ファンディングレート
 * @returns フォーマットされたファンディングレート文字列
 */
export const formatFundingRate = (rate: number): string => {
  return `${(rate * 100).toFixed(4)}%`;
};

/**
 * ファンディングレートの値に応じてテキスト色を返します。
 * @param rate - ファンディングレート
 * @returns Tailwind CSS のクラス名
 */
export const getFundingRateColor = (rate: number): string => {
  if (rate > 0) {
    return "text-green-400";
  }
  if (rate < 0) {
    return "text-red-400";
  }
  return "text-gray-400";
};

/**
 * 数値を米ドル通貨形式でフォーマットします。
 * @param value - 金額
 * @returns フォーマットされた通貨文字列
 */
export const formatCurrency = (value: number | null): string => {
  if (value === null || typeof value === "undefined") {
    return "-";
  }
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
  });
};

/**
 * 出来高を省略形でフォーマットします (例: 1.23M)。
 * @param volume - 出来高
 * @returns フォーマットされた出来高文字列
 */
export const formatVolume = (volume: number): string => {
  if (volume >= 1_000_000) {
    return `${(volume / 1_000_000).toFixed(2)}M`;
  }
  if (volume >= 1_000) {
    return `${(volume / 1_000).toFixed(2)}K`;
  }
  return volume.toString();
};

/**
 * 価格の変動に応じてテキスト色を返します。
 * @param open - 始値
 * @param close - 終値
 * @returns Tailwind CSS のクラス名
 */
export const getPriceChangeColor = (open: number, close: number): string => {
  if (close > open) {
    return "text-green-400";
  }
  if (close < open) {
    return "text-red-400";
  }
  return "text-gray-400";
};

/**
 * 大きな数値を桁区切りでフォーマットします。
 * @param num - 数値
 * @returns フォーマットされた数値文字列
 */
export const formatLargeNumber = (num: number): string => {
  return num.toLocaleString("en-US");
};
