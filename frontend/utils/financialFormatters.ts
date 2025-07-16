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

  if (!item) return num.toFixed(digits);

  return (
    (num / item.value)
      .toFixed(digits)
      .replace(/\.0+$|(\.[0-9]*[1-9])0+$/, "$1") + item.symbol
  );
};

export const formatVolume = (value: number) => {
  return formatLargeNumber(value, 2);
};
