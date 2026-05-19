import {
  calculateDrawdown,
  transformEquityCurve,
  transformTradeHistory,
  generateMonthlyReturns,
  calculateReturnDistribution,
  sampleData,
  calculateBuyAndHoldReturn,
  calculateMaxDrawdown,
  filterDataByDateRange,
} from "@/utils/chartDataTransformers";
import { EquityPoint, Trade } from "@/types/backtest";

describe("calculateDrawdown", () => {
  it("空配列の場合は空配列を返す", () => {
    expect(calculateDrawdown([])).toEqual([]);
  });

  it("ドローダウンを正しく計算する", () => {
    const equityCurve: EquityPoint[] = [
      { timestamp: "2023-01-01", equity: 1000 },
      { timestamp: "2023-01-02", equity: 1100 },
      { timestamp: "2023-01-03", equity: 1050 },
      { timestamp: "2023-01-04", equity: 1200 },
    ];
    const result = calculateDrawdown(equityCurve);

    expect(result[0].drawdown_pct).toBe(0);
    expect(result[1].drawdown_pct).toBe(0);
    expect(result[2].drawdown_pct).toBeCloseTo(50 / 1100);
    expect(result[3].drawdown_pct).toBe(0);
  });

  it("資産が0の場合にドローダウンを0にする", () => {
    const equityCurve: EquityPoint[] = [
      { timestamp: "2023-01-01", equity: 0 },
      { timestamp: "2023-01-02", equity: 100 },
    ];
    const result = calculateDrawdown(equityCurve);
    expect(result[0].drawdown_pct).toBe(0);
  });
});

describe("transformEquityCurve", () => {
  it("空配列の場合は空配列を返す", () => {
    expect(transformEquityCurve([])).toEqual([]);
  });

  it("資産曲線をチャート用データに変換する", () => {
    const equityCurve: EquityPoint[] = [
      { timestamp: "2023-01-01T00:00:00", equity: 1000 },
      { timestamp: "2023-01-02T00:00:00", equity: 1100 },
    ];
    const result = transformEquityCurve(equityCurve);

    expect(result).toHaveLength(2);
    expect(result[0].equity).toBe(1000);
    expect(result[0].drawdown).toBe(0);
    expect(result[0].formattedDate).toContain("2023-01-01");
    expect(typeof result[0].date).toBe("number");
  });
});

describe("transformTradeHistory", () => {
  it("空配列の場合は空配列を返す", () => {
    expect(transformTradeHistory([])).toEqual([]);
  });

  it("取引履歴をチャート用データに変換する", () => {
    const trades: Trade[] = [
      {
        size: 1,
        entry_price: 100,
        exit_price: 110,
        pnl: 10,
        return_pct: 0.1,
        entry_time: "2023-01-01T00:00:00",
        exit_time: "2023-01-02T00:00:00",
      },
    ];
    const result = transformTradeHistory(trades);

    expect(result).toHaveLength(1);
    expect(result[0].pnl).toBe(10);
    expect(result[0].returnPct).toBe(10);
    expect(result[0].type).toBe("long");
    expect(result[0].isWin).toBe(true);
  });

  it("ショートポジションを正しく処理する", () => {
    const trades: Trade[] = [
      {
        size: -1,
        entry_price: 100,
        exit_price: 90,
        pnl: 10,
        return_pct: 0.1,
        entry_time: "2023-01-01T00:00:00",
        exit_time: "2023-01-02T00:00:00",
      },
    ];
    const result = transformTradeHistory(trades);
    expect(result[0].type).toBe("short");
  });
});

describe("generateMonthlyReturns", () => {
  it("空配列の場合は空配列を返す", () => {
    expect(generateMonthlyReturns([])).toEqual([]);
  });

  it("月次リターンを計算する", () => {
    const equityCurve: EquityPoint[] = [
      { timestamp: "2023-01-01", equity: 1000 },
      { timestamp: "2023-01-31", equity: 1100 },
      { timestamp: "2023-02-01", equity: 1100 },
      { timestamp: "2023-02-28", equity: 1210 },
    ];
    const result = generateMonthlyReturns(equityCurve);

    expect(result).toHaveLength(2);
    expect(result[0].month).toBe("2023-01");
    expect(result[0].return).toBeCloseTo(0.1);
    expect(result[1].month).toBe("2023-02");
    expect(result[1].return).toBeCloseTo(0.1);
  });
});

describe("calculateReturnDistribution", () => {
  it("空配列の場合は空配列を返す", () => {
    expect(calculateReturnDistribution([])).toEqual([]);
  });

  it("リターン分布を計算する", () => {
    const trades: Trade[] = [
      {
        size: 1,
        entry_price: 100,
        exit_price: 110,
        pnl: 10,
        return_pct: 0.1,
        entry_time: "2023-01-01",
        exit_time: "2023-01-02",
      },
      {
        size: 1,
        entry_price: 100,
        exit_price: 90,
        pnl: -10,
        return_pct: -0.1,
        entry_time: "2023-01-03",
        exit_time: "2023-01-04",
      },
    ];
    const result = calculateReturnDistribution(trades, 2);

    expect(result).toHaveLength(2);
    expect(result[0].count + result[1].count).toBe(2);
  });
});

describe("sampleData", () => {
  it("データが閾値以下の場合はそのまま返す", () => {
    const data = [1, 2, 3, 4, 5];
    expect(sampleData(data, 10)).toEqual(data);
  });

  it("データをサンプリングする", () => {
    const data = Array.from({ length: 100 }, (_, i) => i);
    const result = sampleData(data, 10);
    expect(result.length).toBeLessThanOrEqual(10);
    expect(result[0]).toBe(0);
  });

  it("空配列を処理する", () => {
    expect(sampleData([])).toEqual([]);
  });
});

describe("calculateBuyAndHoldReturn", () => {
  it("空配列の場合は0を返す", () => {
    expect(calculateBuyAndHoldReturn([])).toBe(0);
  });

  it("1要素の場合は0を返す", () => {
    expect(calculateBuyAndHoldReturn([{ timestamp: "2023-01-01", equity: 100 }])).toBe(0);
  });

  it("リターンを正しく計算する", () => {
    const equityCurve: EquityPoint[] = [
      { timestamp: "2023-01-01", equity: 1000 },
      { timestamp: "2023-12-31", equity: 1500 },
    ];
    expect(calculateBuyAndHoldReturn(equityCurve)).toBeCloseTo(0.5);
  });

  it("初期資産が0の場合は0を返す", () => {
    const equityCurve: EquityPoint[] = [
      { timestamp: "2023-01-01", equity: 0 },
      { timestamp: "2023-12-31", equity: 1500 },
    ];
    expect(calculateBuyAndHoldReturn(equityCurve)).toBe(0);
  });
});

describe("calculateMaxDrawdown", () => {
  it("空配列の場合は0を返す", () => {
    expect(calculateMaxDrawdown([])).toBe(0);
  });

  it("最大ドローダウンを計算する", () => {
    const equityCurve: EquityPoint[] = [
      { timestamp: "2023-01-01", equity: 1000 },
      { timestamp: "2023-01-02", equity: 1200 },
      { timestamp: "2023-01-03", equity: 900 },
      { timestamp: "2023-01-04", equity: 1100 },
    ];
    const result = calculateMaxDrawdown(equityCurve);
    expect(result).toBeCloseTo(300 / 1200);
  });
});

describe("filterDataByDateRange", () => {
  it("空配列の場合は空配列を返す", () => {
    expect(
      filterDataByDateRange([], new Date("2023-01-01"), new Date("2023-12-31"))
    ).toEqual([]);
  });

  it("日付範囲でデータをフィルタリングする", () => {
    const data = [
      { timestamp: "2023-01-15", value: 1 },
      { timestamp: "2023-06-15", value: 2 },
      { timestamp: "2023-12-15", value: 3 },
    ];
    const result = filterDataByDateRange(
      data,
      new Date("2023-03-01"),
      new Date("2023-09-01")
    );
    expect(result).toHaveLength(1);
    expect(result[0].value).toBe(2);
  });

  it("カスタム日付フィールド名をサポートする", () => {
    const data = [
      { date: "2023-01-15", value: 1 },
      { date: "2023-06-15", value: 2 },
    ];
    const result = filterDataByDateRange(
      data,
      new Date("2023-01-01"),
      new Date("2023-12-31"),
      "date"
    );
    expect(result).toHaveLength(2);
  });
});
