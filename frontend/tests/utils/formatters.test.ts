import {
  formatDateTime,
  formatPercentage,
  formatNumber,
  formatFileSize,
  formatTrainingTime,
  formatLargeNumber,
  formatProbability,
  formatDuration,
  formatScore,
} from "@/utils/formatters";

describe("formatDateTime", () => {
  it("有効な日付文字列を正しくフォーマットする", () => {
    const result = formatDateTime("2023-01-01T12:34:56");
    expect(result.date).toBe("2023-01-01");
    expect(result.time).toBe("12:34:56");
    expect(result.dateTime).toBe("2023-01-01 12:34:56");
  });

  it("nullの場合はハイフンを返す", () => {
    const result = formatDateTime(null);
    expect(result).toEqual({ date: "-", time: "-", dateTime: "-" });
  });

  it("無効な日付の場合は入力値をそのまま返す", () => {
    const result = formatDateTime("invalid");
    expect(result.date).toBe("invalid");
    expect(result.dateTime).toBe("invalid");
  });

  it("数値のタイムスタンプを処理する", () => {
    const timestamp = new Date("2023-06-15T10:30:00").getTime();
    const result = formatDateTime(timestamp);
    expect(result.date).toBe("2023-06-15");
  });
});

describe("formatPercentage", () => {
  it("パーセンテージを正しくフォーマットする", () => {
    expect(formatPercentage(12.345)).toBe("12.35%");
  });

  it("0を正しくフォーマットする", () => {
    expect(formatPercentage(0)).toBe("0.00%");
  });

  it("undefinedの場合はN/Aを返す", () => {
    expect(formatPercentage(undefined)).toBe("N/A");
  });

  it("nullの場合はN/Aを返す", () => {
    expect(formatPercentage(null)).toBe("N/A");
  });

  it("NaNの場合はN/Aを返す", () => {
    expect(formatPercentage(NaN)).toBe("N/A");
  });
});

describe("formatNumber", () => {
  it("数値をフォーマットする", () => {
    expect(formatNumber(1234.56, 2, 2)).toBe("1,234.56");
  });

  it("小数点以下の桁数を制御する", () => {
    expect(formatNumber(1234.5, 0, 0)).toBe("1,235");
  });

  it("undefinedの場合はN/Aを返す", () => {
    expect(formatNumber(undefined)).toBe("N/A");
  });

  it("nullの場合はN/Aを返す", () => {
    expect(formatNumber(null)).toBe("N/A");
  });
});

describe("formatFileSize", () => {
  it("MB単位でフォーマットする", () => {
    expect(formatFileSize(1.5)).toBe("1.5 MB");
  });

  it("KB単位でフォーマットする", () => {
    expect(formatFileSize(0.5)).toBe("512.0 KB");
  });

  it("undefinedの場合は不明を返す", () => {
    expect(formatFileSize(undefined)).toBe("不明");
  });
});

describe("formatTrainingTime", () => {
  it("秒数を時間形式でフォーマットする", () => {
    expect(formatTrainingTime(3723)).toBe("1時間2分3秒");
  });

  it("秒のみの場合", () => {
    expect(formatTrainingTime(45)).toBe("45秒");
  });

  it("undefinedの場合は不明を返す", () => {
    expect(formatTrainingTime(undefined)).toBe("不明");
  });
});

describe("formatLargeNumber", () => {
  it("K単位でフォーマットする", () => {
    expect(formatLargeNumber(1234)).toBe("1.23K");
  });

  it("M単位でフォーマットする", () => {
    expect(formatLargeNumber(1234567)).toBe("1.23M");
  });

  it("B単位でフォーマットする", () => {
    expect(formatLargeNumber(1234567890)).toBe("1.23B");
  });

  it("1未満の数値をフォーマットする", () => {
    expect(formatLargeNumber(500)).toBe("500");
  });

  it("小数点以下の桁数を指定する", () => {
    expect(formatLargeNumber(1234, 1)).toBe("1.2K");
  });
});

describe("formatProbability", () => {
  it("確率をパーセンテージでフォーマットする", () => {
    expect(formatProbability(0.123)).toBe("12.3%");
  });

  it("undefinedの場合はN/Aを返す", () => {
    expect(formatProbability(undefined)).toBe("N/A");
  });
});

describe("formatDuration", () => {
  it("60秒未満をフォーマットする", () => {
    expect(formatDuration(30.5)).toBe("30.5秒");
  });

  it("60秒以上をフォーマットする", () => {
    expect(formatDuration(90)).toBe("1分 30秒");
  });

  it("undefinedの場合はN/Aを返す", () => {
    expect(formatDuration(undefined)).toBe("N/A");
  });
});

describe("formatScore", () => {
  it("スコアを小数点4桁でフォーマットする", () => {
    expect(formatScore(0.12345)).toBe("0.1235");
  });

  it("undefinedの場合はN/Aを返す", () => {
    expect(formatScore(undefined)).toBe("N/A");
  });

  it("NaNの場合はN/Aを返す", () => {
    expect(formatScore(NaN)).toBe("N/A");
  });
});
