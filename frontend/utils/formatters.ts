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

export const formatCurrency = (value: number) => {
  return new Intl.NumberFormat("ja-JP", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 1,
  }).format(value);
};

export const formatPercentage = (value: number) => {
  const formatted = value.toFixed(2);
  return `${formatted}%`;
};

export const formatNumber = (
  value: number,
  minDecimals: number = 0,
  maxDecimals: number = 4
) => {
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: minDecimals,
    maximumFractionDigits: maxDecimals,
  }).format(value);
};

export const getPnlColor = (pnl: number) => {
  if (pnl > 0) return "green";
  if (pnl < 0) return "red";
  return "gray"; // または適切なデフォルト色
};

export const getPnlTextColor = (pnl: number) => {
  if (pnl > 0) return "text-green-400";
  if (pnl < 0) return "text-red-400";
  return "text-secondary-400";
};
export const getPriceChangeColor = (open: number, close: number) => {
  if (close > open) return "text-green-400";
  if (close < open) return "text-red-400";
  return "text-gray-400";
};

export const getReturnColor = (value: number | null) => {
  if (value === null) return "gray";
  if (value > 0) return "green";
  if (value < 0) return "red";
  return "gray";
};

export const getSharpeColor = (value: number | null) => {
  if (value === null) return "gray";
  if (value > 1) return "green"; // 一般的にシャープレシオが1より大きいと良好
  if (value < 0) return "red";
  return "gray";
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

export const getFundingRateColor = (value: number) => {
  if (value > 0.0002) return "text-green-400";
  if (value > 0) return "text-green-600";
  if (value < -0.0002) return "text-red-400";
  if (value < 0) return "text-red-600";
  return "text-gray-400";
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
