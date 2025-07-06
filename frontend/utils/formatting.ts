export const formatPercentage = (value: number | undefined | null) => {
  if (value === undefined || value === null || isNaN(value)) {
    // 勝率がnullまたはundefinedの場合、"0.00%"として表示する
    // もしくは、より明確な表示「データなし」などを検討することも可能
    return "0.00%";
  }
  // バックエンドから既にパーセンテージ値として渡されるため、100を掛けずにそのまま使用する
  return `${value.toFixed(2)}%`;
};

export const formatNumber = (
  value: number | undefined | null,
  decimals: number = 2
) => {
  if (value === undefined || value === null || isNaN(value)) {
    return "N/A";
  }
  return value.toFixed(decimals);
};

export const formatCurrency = (value: number | undefined | null) => {
  if (value === undefined || value === null || isNaN(value)) {
    return "N/A";
  }
  return `$${value.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
};

export const getReturnColor = (
  value: number | undefined | null
): "green" | "red" | "gray" => {
  if (value === undefined || value === null || isNaN(value)) return "gray";
  if (value > 0) return "green";
  if (value < 0) return "red";
  return "gray";
};

export const getSharpeColor = (
  value: number | undefined | null
): "green" | "yellow" | "red" | "gray" => {
  if (value === undefined || value === null || isNaN(value)) return "gray";
  if (value > 1.5) return "green";
  if (value > 1.0) return "yellow";
  if (value > 0) return "gray";
  return "red";
};
