import {
  formatCurrency,
  formatPrice,
  formatSymbol,
  formatFundingRate,
  formatVolume,
} from "@/utils/financialFormatters";

describe("formatCurrency", () => {
  it("通貨をフォーマットする", () => {
    const result = formatCurrency(1234.56);
    expect(result).toContain("1,234.6");
    expect(result).toContain("$");
  });

  it("0をフォーマットする", () => {
    const result = formatCurrency(0);
    expect(result).toContain("0");
  });

  it("負の値をフォーマットする", () => {
    const result = formatCurrency(-500.25);
    expect(result).toContain("500.3");
  });
});

describe("formatPrice", () => {
  it("価格をフォーマットする", () => {
    expect(formatPrice(1234.56)).toBe("1,234.56");
  });

  it("nullの場合はハイフンを返す", () => {
    expect(formatPrice(null)).toBe("-");
  });

  it("整数値に小数点を追加する", () => {
    expect(formatPrice(1000)).toBe("1,000.00");
  });
});

describe("formatSymbol", () => {
  it("USDTシンボルをフォーマットする", () => {
    expect(formatSymbol("BTCUSDT")).toBe("BTC/USDT");
  });

  it("USDT以外のシンボルはそのまま返す", () => {
    expect(formatSymbol("ETHBTC")).toBe("ETHBTC");
  });

  it("空文字列を処理する", () => {
    expect(formatSymbol("")).toBe("");
  });
});

describe("formatFundingRate", () => {
  it("ファンディングレートをパーセンテージでフォーマットする", () => {
    expect(formatFundingRate(0.0123)).toBe("1.2300%");
  });

  it("0をフォーマットする", () => {
    expect(formatFundingRate(0)).toBe("0.0000%");
  });

  it("負の値をフォーマットする", () => {
    expect(formatFundingRate(-0.001)).toBe("-0.1000%");
  });
});

describe("formatVolume", () => {
  it("取引高をK単位でフォーマットする", () => {
    expect(formatVolume(1234)).toBe("1.23K");
  });

  it("取引高をM単位でフォーマットする", () => {
    expect(formatVolume(1234567)).toBe("1.23M");
  });

  it("小さな値をフォーマットする", () => {
    expect(formatVolume(500)).toBe("500");
  });
});
