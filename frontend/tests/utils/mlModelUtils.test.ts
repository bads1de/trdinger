import {
  getScoreBadgeVariant,
  getModelTypeBadgeVariant,
  getStatusBadgeVariant,
} from "@/utils/mlModelUtils";

describe("getScoreBadgeVariant", () => {
  it("0.8以上のスコアはsuccessを返す", () => {
    expect(getScoreBadgeVariant(0.85)).toBe("success");
    expect(getScoreBadgeVariant(0.8)).toBe("success");
    expect(getScoreBadgeVariant(1.0)).toBe("success");
  });

  it("0.7以上0.8未満のスコアはwarningを返す", () => {
    expect(getScoreBadgeVariant(0.75)).toBe("warning");
    expect(getScoreBadgeVariant(0.7)).toBe("warning");
    expect(getScoreBadgeVariant(0.79)).toBe("warning");
  });

  it("0.7未満のスコアはdestructiveを返す", () => {
    expect(getScoreBadgeVariant(0.5)).toBe("destructive");
    expect(getScoreBadgeVariant(0.69)).toBe("destructive");
  });

  it("スコアが0の場合はoutlineを返す（falsy判定）", () => {
    expect(getScoreBadgeVariant(0)).toBe("outline");
  });

  it("スコアが未指定（undefined）の場合はoutlineを返す", () => {
    expect(getScoreBadgeVariant(undefined)).toBe("outline");
  });

  it("スコアがnullの場合はoutlineを返す", () => {
    expect(getScoreBadgeVariant(null as unknown as number)).toBe("outline");
  });

  it("境界値を正しく処理すること", () => {
    expect(getScoreBadgeVariant(0.799)).toBe("warning");
    expect(getScoreBadgeVariant(0.800)).toBe("success");
  });
});

describe("getModelTypeBadgeVariant", () => {
  it("lightgbmはdefaultを返す", () => {
    expect(getModelTypeBadgeVariant("lightgbm")).toBe("default");
  });

  it("大文字小文字を区別しないこと", () => {
    expect(getModelTypeBadgeVariant("LightGBM")).toBe("default");
    expect(getModelTypeBadgeVariant("LIGHTGBM")).toBe("default");
  });

  it("xgboostはoutlineを返す", () => {
    expect(getModelTypeBadgeVariant("xgboost")).toBe("outline");
    expect(getModelTypeBadgeVariant("XGBoost")).toBe("outline");
  });

  it("その他のモデルタイプはoutlineを返す", () => {
    expect(getModelTypeBadgeVariant("catboost")).toBe("outline");
    expect(getModelTypeBadgeVariant("random_forest")).toBe("outline");
  });

  it("undefinedの場合はoutlineを返す", () => {
    expect(getModelTypeBadgeVariant(undefined)).toBe("outline");
  });

  it("空文字列の場合はoutlineを返す", () => {
    expect(getModelTypeBadgeVariant("")).toBe("outline");
  });
});

describe("getStatusBadgeVariant", () => {
  it("トレーニング中の場合はdefaultを返す", () => {
    expect(getStatusBadgeVariant(true, false)).toBe("default");
    expect(getStatusBadgeVariant(true, true)).toBe("default");
  });

  it("モデルロード済みの場合はsuccessを返す", () => {
    expect(getStatusBadgeVariant(false, true)).toBe("success");
  });

  it("それ以外の場合はoutlineを返す", () => {
    expect(getStatusBadgeVariant(false, false)).toBe("outline");
    expect(getStatusBadgeVariant(undefined, undefined)).toBe("outline");
    expect(getStatusBadgeVariant(false, undefined)).toBe("outline");
    expect(getStatusBadgeVariant(undefined, false)).toBe("outline");
  });
});
