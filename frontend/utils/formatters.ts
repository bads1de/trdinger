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

export const formatPercentage = (value?: number | null) => {
  if (value === undefined || value === null || isNaN(value)) {
    return "N/A";
  }

  const formatted = value.toFixed(2);

  return `${formatted}%`;
};

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

export const formatFileSize = (sizeInMB?: number) => {
  if (sizeInMB === undefined || sizeInMB === null) return "不明";

  if (sizeInMB < 1) {
    return `${(sizeInMB * 1024).toFixed(1)} KB`;
  }

  return `${sizeInMB.toFixed(1)} MB`;
};

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

export const formatProbability = (prob?: number) => {
  if (prob === undefined || prob === null) return "N/A";

  return `${(prob * 100).toFixed(1)}%`;
};

export const formatDuration = (seconds?: number) => {
  if (seconds === undefined || seconds === null) return "N/A";

  if (seconds < 60) {
    return `${seconds.toFixed(1)}秒`;
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;

  return `${minutes}分 ${remainingSeconds.toFixed(0)}秒`;
};

export const formatScore = (score?: number) => {
  if (score === undefined || score === null || isNaN(score)) return "N/A";

  return score.toFixed(4);
};
