/**
 * 日時をフォーマットする
 * @param dateInput - フォーマットする日時 (文字列、数値、またはnull)
 * @returns フォーマットされた日付、時刻、および両方を含むオブジェクト
 * @example
 * // returns { date: '2023-01-01', time: '12:34:56', dateTime: '2023-01-01 12:34:56' }
 * formatDateTime('2023-01-01T12:34:56')
 */
export const formatDateTime = (
  dateInput: string | number | null
): { date: string; time: string; dateTime: string } => {
  if (dateInput === null || dateInput === undefined) {
    return { date: "-", time: "-", dateTime: "-" };
  }

  try {
    const date = new Date(dateInput);

    if (isNaN(date.getTime())) {
      return { date: String(dateInput), time: "", dateTime: String(dateInput) };
    }

    const dateString = new Intl.DateTimeFormat("ja-JP", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    })
      .format(date)
      .replace(/\//g, "-");

    const timeString = new Intl.DateTimeFormat("ja-JP", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }).format(date);

    return {
      date: dateString,
      time: timeString,
      dateTime: `${dateString} ${timeString}`,
    };
  } catch {
    return { date: String(dateInput), time: "", dateTime: String(dateInput) };
  }
};

/**
 * パーセンテージをフォーマットする
 * @param value - フォーマットする値 (0-1の範囲)
 * @returns フォーマットされたパーセンテージ文字列 (例: '12.34%') または 'N/A'
 * @example
 * // returns '12.34%'
 * formatPercentage(0.1234)
 */
export const formatPercentage = (value?: number | null) => {
  if (value === undefined || value === null || isNaN(value)) {
    return "N/A";
  }

  const formatted = value.toFixed(2);

  return `${formatted}%`;
};

/**
 * 数値をフォーマットする
 * @param value - フォーマットする数値
 * @param minDecimals - 最小の小数点以下の桁数 (デフォルト: 0)
 * @param maxDecimals - 最大の小数点以下の桁数 (デフォルト: 4)
 * @returns フォーマットされた数値文字列または 'N/A'
 * @example
 * // returns '1,234.56'
 * formatNumber(1234.56, 2, 2)
 */
export const formatNumber = (
  value?: number | null,
  minDecimals: number = 0,
  maxDecimals: number = 4
) => {
  if (value === undefined || value === null || isNaN(value)) {
    return "N/A";
  }

  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: minDecimals,
    maximumFractionDigits: maxDecimals,
  }).format(value);
};

/**
 * ファイルサイズを適切な単位でフォーマットする
 * @param sizeInMB - メガバイト単位のファイルサイズ
 * @returns フォーマットされたファイルサイズ文字列 (例: '1.5 MB', '768.2 KB')
 * @example
 * // returns '1.5 MB'
 * formatFileSize(1.5)
 */
export const formatFileSize = (sizeInMB?: number) => {
  if (sizeInMB === undefined || sizeInMB === null) return "不明";

  if (sizeInMB < 1) {
    return `${(sizeInMB * 1024).toFixed(1)} KB`;
  }

  return `${sizeInMB.toFixed(1)} MB`;
};

/**
 * トレーニング時間をフォーマットする
 * @param seconds - 秒数
 * @returns フォーマットされた時間文字列 (例: '2時間30分15秒')
 * @example
 * // returns '1時間2分3秒'
 * formatTrainingTime(3723)
 */
export const formatTrainingTime = (seconds?: number) => {
  if (seconds === undefined || seconds === null) return "不明";

  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}時間${minutes}分${secs}秒`;
  }

  if (minutes > 0) {
    return `${minutes}分${secs}秒`;
  }

  return `${secs}秒`;
};

/**
 * 大きな数値を適切な単位付きでフォーマットする
 * @param num - フォーマットする数値
 * @param digits - 小数点以下の桁数 (デフォルト: 2)
 * @returns フォーマットされた数値文字列 (例: '1.23K', '4.56M')
 * @example
 * // returns '1.23K'
 * formatLargeNumber(1234)
 */
export const formatLargeNumber = (num: number, digits = 2) => {
  const lookup = [
    { value: 1, symbol: "" },
    { value: 1e3, symbol: "K" },
    { value: 1e6, symbol: "M" },
    { value: 1e9, symbol: "B" },
    { value: 1e12, symbol: "T" },
  ];
  const item = lookup
    .slice()
    .reverse()
    .find((item) => num >= item.value);

  if (!item) {
    return num.toFixed(digits);
  }

  return (
    (num / item.value).toFixed(digits).replace(/\.0+$|(\.0*[^0])0+$/, "$1") +
    item.symbol
  );
};

/**
 * 確率をパーセンテージでフォーマットする
 * @param prob - 確率 (0-1の範囲)
 * @returns フォーマットされた確率文字列 (例: '12.3%') または 'N/A'
 * @example
 * // returns '12.3%'
 * formatProbability(0.123)
 */
export const formatProbability = (prob?: number) => {
  if (prob === undefined || prob === null) return "N/A";

  return `${(prob * 100).toFixed(1)}%`;
};

/**
 * 経過時間をフォーマットする
 * @param seconds - 秒数
 * @returns フォーマットされた時間文字列 (例: '2分 30秒')
 * @example
 * // returns '1分 2秒'
 * formatDuration(62)
 */
export const formatDuration = (seconds?: number) => {
  if (seconds === undefined || seconds === null) return "N/A";

  if (seconds < 60) {
    return `${seconds.toFixed(1)}秒`;
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;

  return `${minutes}分 ${remainingSeconds.toFixed(0)}秒`;
};

/**
 * スコアをフォーマットする
 * @param score - フォーマットするスコア
 * @returns フォーマットされたスコア文字列 (小数点以下4桁) または 'N/A'
 * @example
 * // returns '0.1234'
 * formatScore(0.12345)
 */
export const formatScore = (score?: number) => {
  if (score === undefined || score === null || isNaN(score)) return "N/A";

  return score.toFixed(4);
};
