/**
 * 値に基づいて色を決定する汎用関数
 * @param value 評価する数値
 * @param options オプション
 * @param options.invert 評価を反転するか（小さいほど良い場合はtrue）
 * @param options.threshold しきい値（デフォルトは0）
 * @returns Tailwind CSSのカラークラス
 */
export const getValueColorClass = (
  value: number | null | undefined,
  options: { invert?: boolean; threshold?: number } = {}
) => {
  const { invert = false, threshold = 0 } = options;

  if (value === null || value === undefined || isNaN(value)) {
    return "text-secondary-400";
  }

  const isPositive = value > threshold;
  const isNegative = value < threshold;

  if (invert) {
    if (isPositive) {
      return "text-red-400";
    }

    if (isNegative) {
      return "text-green-400";
    }
  } else {
    if (isPositive) {
      return "text-green-400";
    }

    if (isNegative) {
      return "text-red-400";
    }
  }

  return "text-secondary-400";
};

export const getPnlColor = (value: number): "green" | "red" | "gray" => {
  return getSemanticColor(value);
};

export const getPnlTextColor = (pnl: number) => {
  return getValueColorClass(pnl);
};

export const getPriceChangeColor = (open: number, close: number) => {
  return getValueColorClass(close - open);
};

export const getReturnColor = (
  value: number | null
): "green" | "red" | "gray" => {
  return getSemanticColor(value);
};

export const getSharpeColor = (
  value: number | null
): "green" | "red" | "gray" => {
  return getSemanticColor(value, { threshold: 1 });
};

export const getFundingRateColor = (value: number) => {
  if (value > 0.0002) {
    return "text-green-400";
  }

  if (value > 0) {
    return "text-green-600";
  }

  if (value < -0.0002) {
    return "text-red-400";
  }

  if (value < 0) {
    return "text-red-600";
  }

  return "text-gray-400";
};

/**
 * スコアに基づいて色を決定する関数
 * @param score 評価するスコア
 * @returns Tailwind CSSのカラークラス
 */
export const getScoreColorClass = (score?: number) => {
  if (score === undefined || score === null || isNaN(score)) {
    return "text-gray-400";
  }

  if (score >= 0.8) {
    return "text-green-400";
  }

  if (score >= 0.7) {
    return "text-yellow-400";
  }

  if (score >= 0.6) {
    return "text-orange-400";
  }

  return "text-red-400";
};

/**
 * 特徴量重要度のバーの色を生成する関数
 * @param index バーのインデックス
 * @param total 総数
 * @returns HSL色文字列
 */
export const getBarColor = (index: number, total: number) => {
  const intensity = 1 - index / total;
  const hue = 180 + intensity * 60;

  return `hsl(${hue}, 70%, ${50 + intensity * 20}%)`;
};

/**
 * MLトレーニングステータスに基づいて色を決定する関数
 * @param status トレーニングステータス
 * @returns Tailwind CSSのカラークラス
 */
export const getStatusColor = (status: string) => {
  switch (status) {
    case "completed":
      return "text-green-600";

    case "error":
      return "text-red-600";

    case "training":
    case "loading_data":
    case "initializing":
      return "text-blue-600";

    default:
      return "text-gray-600";
  }
};

export const getSemanticColor = <T extends string = "green" | "red" | "gray">(
  value: number | null | undefined,
  options: {
    threshold?: number;
    positiveColor?: T;
    negativeColor?: T;
    neutralColor?: T;
  } = {}
): T => {
  const {
    threshold = 0,
    positiveColor = "green" as T,
    negativeColor = "red" as T,
    neutralColor = "gray" as T,
  } = options;

  if (value === null || value === undefined || isNaN(value)) {
    return neutralColor;
  }

  if (value > threshold) {
    return positiveColor;
  }

  if (value < threshold) {
    return negativeColor;
  }

  return neutralColor;
};
