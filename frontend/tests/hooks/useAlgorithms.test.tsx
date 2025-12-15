import { renderHook } from "@testing-library/react";
import { useAlgorithms } from "@/hooks/useAlgorithms";

describe("useAlgorithms", () => {
  it("アルゴリズム一覧が取得できること", () => {
    const { result } = renderHook(() => useAlgorithms());
    expect(result.current.algorithms.length).toBeGreaterThan(0);
    // 代表的なアルゴリズムが含まれているか確認（ID的なnameは小文字の可能性が高い）
    const hasLightGBM = result.current.algorithms.some(
      (a) => a.name.toLowerCase() === "lightgbm"
    );
    expect(hasLightGBM).toBe(true);
  });

  it("検索機能が正常に動作すること", () => {
    const { result } = renderHook(() => useAlgorithms());

    // "LightGBM" で検索
    const searchResult = result.current.searchAlgorithms("LightGBM");
    // ヒットするものが1つ以上あること
    expect(searchResult.length).toBeGreaterThan(0);
    // ヒットしたものはクエリに関連するはず
    expect(searchResult[0].name.toLowerCase()).toContain("lightgbm");

    // 大文字小文字を区別しない検索
    const searchResultLower = result.current.searchAlgorithms("lightgbm");
    expect(searchResultLower.length).toBeGreaterThan(0);

    // 存在しないクエリ
    const searchResultNone = result.current.searchAlgorithms(
      "NonExistentAlgorithmXYZ"
    );
    expect(searchResultNone.length).toBe(0);

    // 空クエリは全件返す
    const searchResultAll = result.current.searchAlgorithms("");
    expect(searchResultAll.length).toBe(result.current.algorithms.length);
  });

  it("推奨アルゴリズム機能が正常に動作すること", () => {
    const { result } = renderHook(() => useAlgorithms());

    // 確率予測が必要な場合
    const probAlgorithms = result.current.getRecommendedAlgorithms({
      needsProbability: true,
    });
    expect(probAlgorithms.every((a) => a.has_probability_prediction)).toBe(
      true
    );

    // 特徴量重要度が必要な場合
    const fiAlgorithms = result.current.getRecommendedAlgorithms({
      needsFeatureImportance: true,
    });
    expect(fiAlgorithms.every((a) => a.has_feature_importance)).toBe(true);
  });

  it("ユーティリティ関数が正しいラベルを返すこと", () => {
    const { result } = renderHook(() => useAlgorithms());

    // 存在するタイプ
    expect(result.current.getTypeLabel("boosting")).toBeTruthy();
    // 存在しないタイプはそのまま返る
    expect(result.current.getTypeLabel("unknown_type")).toBe("unknown_type");
  });
});
