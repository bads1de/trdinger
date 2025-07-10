export const formatDateTime = (dateInput: string | number) => {
  try {
    const date = new Date(dateInput);
    return {
      date: date.toLocaleDateString("ja-JP", {
        year: "2-digit",
        month: "2-digit",
        day: "2-digit",
      }),
      time: date.toLocaleTimeString("ja-JP", {
        hour: "2-digit",
        minute: "2-digit",
      }),
      fullDateTime: date.toLocaleString("ja-JP", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      }),
    };
  } catch {
    const safeDateInput =
      typeof dateInput === "number"
        ? new Date(dateInput).toISOString()
        : dateInput;
    return { date: safeDateInput, time: "", fullDateTime: safeDateInput };
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
