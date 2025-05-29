/**
 * フォーマッター関数集
 *
 * 各種データの表示用フォーマット関数を提供します。
 * 数値、日時、通貨、パーセンテージなどの統一的なフォーマットを行います。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

/**
 * 通貨ペアシンボルをフォーマットする関数
 * "BTC/USDT:USDT" -> "BTC/USDT"
 */
export const formatSymbol = (symbol: string): string => {
  return symbol.replace(/:.*$/, "");
};

/**
 * 数値を通貨形式でフォーマットする関数
 */
export const formatCurrency = (value: number | null): string => {
  if (value === null || value === undefined) return "-";
  
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 8,
  }).format(value);
};

/**
 * 価格をフォーマットする関数（formatCurrencyのエイリアス）
 */
export const formatPrice = formatCurrency;

/**
 * 数値をコンパクト形式でフォーマットする関数
 */
export const formatCompactNumber = (value: number | null): string => {
  if (value === null || value === undefined) return "-";
  
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(value);
};

/**
 * 出来高をフォーマットする関数
 */
export const formatVolume = (value: number | null): string => {
  if (value === null || value === undefined) return "-";
  
  if (value >= 1000000) {
    return new Intl.NumberFormat("en-US", {
      notation: "compact",
      maximumFractionDigits: 1,
    }).format(value);
  }
  
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  }).format(value);
};

/**
 * 日時をフォーマットする関数
 */
export const formatDateTime = (dateString: string | null): string => {
  if (!dateString) return "-";
  
  const date = new Date(dateString);
  return date.toLocaleString("ja-JP", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
};

/**
 * ファンディングレートをパーセンテージでフォーマットする関数
 */
export const formatFundingRate = (rate: number | null): string => {
  if (rate === null || rate === undefined) return "-";
  
  // ファンディングレートは通常小数で提供されるため、パーセンテージに変換
  const percentage = rate * 100;
  
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    minimumFractionDigits: 4,
    maximumFractionDigits: 6,
  }).format(rate);
};

/**
 * パーセンテージをフォーマットする関数
 */
export const formatPercentage = (value: number | null, decimals: number = 2): string => {
  if (value === null || value === undefined) return "-";
  
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value / 100);
};

/**
 * ファンディングレートの色を取得する関数
 */
export const getFundingRateColor = (rate: number | null): string => {
  if (rate === null || rate === undefined) return "text-gray-400";
  
  if (rate > 0) return "text-red-400"; // 正のファンディングレート（ロングが支払い）
  if (rate < 0) return "text-green-400"; // 負のファンディングレート（ショートが支払い）
  return "text-gray-400"; // ゼロ
};

/**
 * 価格変動の色を取得する関数
 */
export const getPriceChangeColor = (openPrice: number, closePrice: number): string => {
  if (closePrice > openPrice) return "text-green-400"; // 上昇
  if (closePrice < openPrice) return "text-red-400"; // 下降
  return "text-gray-400"; // 変動なし
};

/**
 * 変動率を計算してフォーマットする関数
 */
export const formatPriceChange = (openPrice: number, closePrice: number): string => {
  if (!openPrice || !closePrice) return "-";
  
  const change = ((closePrice - openPrice) / openPrice) * 100;
  const sign = change >= 0 ? "+" : "";
  
  return `${sign}${change.toFixed(2)}%`;
};

/**
 * 大きな数値を読みやすい形式でフォーマットする関数
 */
export const formatLargeNumber = (value: number | null): string => {
  if (value === null || value === undefined) return "-";
  
  const absValue = Math.abs(value);
  
  if (absValue >= 1e12) {
    return `${(value / 1e12).toFixed(1)}T`;
  } else if (absValue >= 1e9) {
    return `${(value / 1e9).toFixed(1)}B`;
  } else if (absValue >= 1e6) {
    return `${(value / 1e6).toFixed(1)}M`;
  } else if (absValue >= 1e3) {
    return `${(value / 1e3).toFixed(1)}K`;
  }
  
  return value.toFixed(0);
};

/**
 * 時間の経過を人間が読みやすい形式でフォーマットする関数
 */
export const formatTimeAgo = (dateString: string): string => {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  
  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);
  
  if (diffDays > 0) return `${diffDays}日前`;
  if (diffHours > 0) return `${diffHours}時間前`;
  if (diffMinutes > 0) return `${diffMinutes}分前`;
  return `${diffSeconds}秒前`;
};

/**
 * 数値の範囲を色で表現する関数
 */
export const getValueRangeColor = (
  value: number,
  min: number,
  max: number
): string => {
  const ratio = (value - min) / (max - min);
  
  if (ratio <= 0.33) return "text-green-400";
  if (ratio <= 0.66) return "text-yellow-400";
  return "text-red-400";
};
