import {
  getValueColorClass,
  getPnlColor,
  getPnlTextColor,
  getPriceChangeColor,
  getReturnColor,
  getSharpeColor,
  getFundingRateColor,
  getScoreColorClass,
  getBarColor,
  getStatusColor,
  getSemanticColor,
} from "@/utils/colorUtils";

describe("getValueColorClass", () => {
  it("正の値は緑色を返す", () => {
    expect(getValueColorClass(10)).toBe("text-green-400");
  });

  it("負の値は赤色を返す", () => {
    expect(getValueColorClass(-10)).toBe("text-red-400");
  });

  it("0はsecondary色を返す", () => {
    expect(getValueColorClass(0)).toBe("text-secondary-400");
  });

  it("nullの場合はsecondary色を返す", () => {
    expect(getValueColorClass(null)).toBe("text-secondary-400");
  });

  it("undefinedの場合はsecondary色を返す", () => {
    expect(getValueColorClass(undefined)).toBe("text-secondary-400");
  });

  it("invertオプションで色を反転する", () => {
    expect(getValueColorClass(10, { invert: true })).toBe("text-red-400");
    expect(getValueColorClass(-10, { invert: true })).toBe("text-green-400");
  });

  it("thresholdオプションを適用する", () => {
    expect(getValueColorClass(5, { threshold: 10 })).toBe("text-red-400");
    expect(getValueColorClass(15, { threshold: 10 })).toBe("text-green-400");
  });
});

describe("getPnlColor", () => {
  it("正の値はgreenを返す", () => {
    expect(getPnlColor(100)).toBe("green");
  });

  it("負の値はredを返す", () => {
    expect(getPnlColor(-100)).toBe("red");
  });

  it("0はgrayを返す", () => {
    expect(getPnlColor(0)).toBe("gray");
  });
});

describe("getPnlTextColor", () => {
  it("正のPnLは緑色を返す", () => {
    expect(getPnlTextColor(50)).toBe("text-green-400");
  });

  it("負のPnLは赤色を返す", () => {
    expect(getPnlTextColor(-50)).toBe("text-red-400");
  });
});

describe("getPriceChangeColor", () => {
  it("上昇は緑色を返す", () => {
    expect(getPriceChangeColor(100, 110)).toBe("text-green-400");
  });

  it("下落は赤色を返す", () => {
    expect(getPriceChangeColor(100, 90)).toBe("text-red-400");
  });
});

describe("getReturnColor", () => {
  it("正のリターンはgreenを返す", () => {
    expect(getReturnColor(0.05)).toBe("green");
  });

  it("負のリターンはredを返す", () => {
    expect(getReturnColor(-0.05)).toBe("red");
  });

  it("nullはgrayを返す", () => {
    expect(getReturnColor(null)).toBe("gray");
  });
});

describe("getSharpeColor", () => {
  it("1以上のシャープレシオはgreenを返す", () => {
    expect(getSharpeColor(1.5)).toBe("green");
  });

  it("1未満のシャープレシオはredを返す", () => {
    expect(getSharpeColor(0.5)).toBe("red");
  });

  it("nullはgrayを返す", () => {
    expect(getSharpeColor(null)).toBe("gray");
  });
});

describe("getFundingRateColor", () => {
  it("0.0002超は明るい緑色を返す", () => {
    expect(getFundingRateColor(0.0003)).toBe("text-green-400");
  });

  it("0超0.0002以下は暗い緑色を返す", () => {
    expect(getFundingRateColor(0.0001)).toBe("text-green-600");
  });

  it("-0.0002未満は明るい赤色を返す", () => {
    expect(getFundingRateColor(-0.0003)).toBe("text-red-400");
  });

  it("-0.0002以上0未満は暗い赤色を返す", () => {
    expect(getFundingRateColor(-0.0001)).toBe("text-red-600");
  });

  it("0は灰色を返す", () => {
    expect(getFundingRateColor(0)).toBe("text-gray-400");
  });
});

describe("getScoreColorClass", () => {
  it("0.8以上は緑色を返す", () => {
    expect(getScoreColorClass(0.9)).toBe("text-green-400");
  });

  it("0.7以上0.8未満は黄色を返す", () => {
    expect(getScoreColorClass(0.75)).toBe("text-yellow-400");
  });

  it("0.6以上0.7未満はオレンジ色を返す", () => {
    expect(getScoreColorClass(0.65)).toBe("text-orange-400");
  });

  it("0.6未満は赤色を返す", () => {
    expect(getScoreColorClass(0.5)).toBe("text-red-400");
  });

  it("undefinedは灰色を返す", () => {
    expect(getScoreColorClass(undefined)).toBe("text-gray-400");
  });
});

describe("getBarColor", () => {
  it("HSL色文字列を返す", () => {
    const color = getBarColor(0, 10);
    expect(color).toMatch(/^hsl\(/);
  });

  it("インデックスに応じて色が変化する", () => {
    const color0 = getBarColor(0, 10);
    const color9 = getBarColor(9, 10);
    expect(color0).not.toBe(color9);
  });
});

describe("getStatusColor", () => {
  it("completedは緑色を返す", () => {
    expect(getStatusColor("completed")).toBe("text-green-600");
  });

  it("errorは赤色を返す", () => {
    expect(getStatusColor("error")).toBe("text-red-600");
  });

  it("trainingは青色を返す", () => {
    expect(getStatusColor("training")).toBe("text-blue-600");
  });

  it("loading_dataは青色を返す", () => {
    expect(getStatusColor("loading_data")).toBe("text-blue-600");
  });

  it("初期化中は青色を返す", () => {
    expect(getStatusColor("initializing")).toBe("text-blue-600");
  });

  it("デフォルトは灰色を返す", () => {
    expect(getStatusColor("unknown")).toBe("text-gray-600");
  });
});

describe("getSemanticColor", () => {
  it("正の値はgreenを返す", () => {
    expect(getSemanticColor(5)).toBe("green");
  });

  it("負の値はredを返す", () => {
    expect(getSemanticColor(-5)).toBe("red");
  });

  it("0はgrayを返す", () => {
    expect(getSemanticColor(0)).toBe("gray");
  });

  it("nullはgrayを返す", () => {
    expect(getSemanticColor(null)).toBe("gray");
  });

  it("カスタム色をサポートする", () => {
    expect(
      getSemanticColor(5, {
        positiveColor: "blue",
        negativeColor: "orange",
        neutralColor: "black",
      })
    ).toBe("blue");
  });

  it("カスタムthresholdをサポートする", () => {
    expect(getSemanticColor(5, { threshold: 10 })).toBe("red");
  });
});
