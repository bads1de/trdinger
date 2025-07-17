import { formatLargeNumber } from "./formatters";

export const formatCurrency = (value: number) => {
  return new Intl.NumberFormat("ja-JP", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 1,
  }).format(value);
};

export const formatPrice = (value: number | null) => {
  if (value === null) return "-";

  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
};

export const formatSymbol = (symbol: string) => {
  if (symbol.endsWith("USDT")) {
    return `${symbol.slice(0, -4)}/USDT`;
  }

  return symbol;
};

export const formatFundingRate = (value: number) => {
  return `${(value * 100).toFixed(4)}%`;
};

export const formatVolume = (value: number) => {
  return formatLargeNumber(value, 2);
};
