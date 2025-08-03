import { useCallback, useMemo } from "react";
import {
  ALGORITHMS,
  ALGORITHM_LIST,
  ALGORITHM_NAMES,
  ALGORITHMS_BY_TYPE,
  ALGORITHMS_BY_CAPABILITY,
  PROBABILITY_ALGORITHMS,
  FEATURE_IMPORTANCE_ALGORITHMS,
  ALGORITHM_STATISTICS,
  ALGORITHM_TYPE_LABELS,
  CAPABILITY_LABELS,
  Algorithm,
  AlgorithmType,
  AlgorithmCapability,
} from "../constants/algorithms";

// 後方互換性のための型定義（既存のコンポーネントで使用されている可能性があるため）
export interface AlgorithmSummary {
  total_algorithms: number;
  by_type: Record<string, string[]>;
  by_capability: Record<string, string[]>;
  algorithms: Record<
    string,
    {
      type: string;
      description: string;
      capabilities: string[];
    }
  >;
}

export interface AlgorithmsResponse {
  success: boolean;
  algorithms: Record<string, Algorithm>;
  summary: AlgorithmSummary;
  total_count: number;
  message: string;
}

export interface AlgorithmInfoResponse {
  success: boolean;
  algorithm?: Algorithm;
  error?: string;
  available_algorithms?: string[];
  message: string;
}

// 型をエクスポート（既存のコンポーネントとの互換性のため）
export type { Algorithm, AlgorithmType, AlgorithmCapability };

export const useAlgorithms = () => {
  // 定数からアルゴリズム情報を取得（API呼び出し不要）
  const algorithms = useMemo(() => {
    return ALGORITHM_LIST.map((algo) => ({
      ...algo,
      type_label: ALGORITHM_TYPE_LABELS[algo.type],
      capability_labels: algo.capabilities.map((cap) => CAPABILITY_LABELS[cap]),
    }));
  }, []);

  // タイプ別アルゴリズム（定数から取得）
  const algorithmsByType = useMemo(() => ALGORITHMS_BY_TYPE, []);

  // 機能別アルゴリズム（定数から取得）
  const algorithmsByCapability = useMemo(() => ALGORITHMS_BY_CAPABILITY, []);

  // 確率予測対応アルゴリズム（定数から取得）
  const probabilityAlgorithms = useMemo(() => PROBABILITY_ALGORITHMS, []);

  // 特徴量重要度対応アルゴリズム（定数から取得）
  const featureImportanceAlgorithms = useMemo(
    () => FEATURE_IMPORTANCE_ALGORITHMS,
    []
  );

  // アルゴリズム検索
  const searchAlgorithms = useCallback(
    (query: string) => {
      if (!query.trim()) return algorithms;

      const lowerQuery = query.toLowerCase();
      return algorithms.filter(
        (algo) =>
          algo.name.toLowerCase().includes(lowerQuery) ||
          algo.display_name.toLowerCase().includes(lowerQuery) ||
          algo.description.toLowerCase().includes(lowerQuery) ||
          algo.pros.some((pro) => pro.toLowerCase().includes(lowerQuery)) ||
          algo.best_for.some((use) => use.toLowerCase().includes(lowerQuery))
      );
    },
    [algorithms]
  );

  // アルゴリズム推奨
  const getRecommendedAlgorithms = useCallback(
    (requirements: {
      dataSize?: "small" | "medium" | "large";
      needsProbability?: boolean;
      needsFeatureImportance?: boolean;
      needsSpeed?: boolean;
      needsAccuracy?: boolean;
      hasNoise?: boolean;
    }) => {
      let candidates = algorithms;

      // 確率予測が必要
      if (requirements.needsProbability) {
        candidates = candidates.filter(
          (algo) => algo.has_probability_prediction
        );
      }

      // 特徴量重要度が必要
      if (requirements.needsFeatureImportance) {
        candidates = candidates.filter((algo) => algo.has_feature_importance);
      }

      // データサイズに基づく推奨
      if (requirements.dataSize === "small") {
        candidates = candidates.filter((algo) =>
          algo.best_for.some(
            (use) => use.includes("小規模") || use.includes("少ない")
          )
        );
      } else if (requirements.dataSize === "large") {
        candidates = candidates.filter((algo) =>
          algo.best_for.some(
            (use) => use.includes("大規模") || use.includes("高速")
          )
        );
      }

      // 速度重視
      if (requirements.needsSpeed) {
        candidates = candidates.filter((algo) =>
          algo.pros.some((pro) => pro.includes("高速") || pro.includes("速い"))
        );
      }

      // 精度重視
      if (requirements.needsAccuracy) {
        candidates = candidates.filter((algo) =>
          algo.pros.some(
            (pro) => pro.includes("高い精度") || pro.includes("精度")
          )
        );
      }

      // ノイズ耐性
      if (requirements.hasNoise) {
        candidates = candidates.filter(
          (algo) =>
            algo.pros.some(
              (pro) => pro.includes("ノイズ") || pro.includes("耐性")
            ) || algo.best_for.some((use) => use.includes("ノイズ"))
        );
      }

      return candidates;
    },
    [algorithms]
  );

  // 統計情報（定数から取得）
  const statistics = useMemo(() => ALGORITHM_STATISTICS, []);

  return {
    // データ
    algorithms,
    algorithmsData: null, // 定数ベースなのでnull

    // グループ化されたデータ
    algorithmsByType,
    algorithmsByCapability,
    probabilityAlgorithms,
    featureImportanceAlgorithms,

    // 統計情報
    statistics,

    // 機能
    searchAlgorithms,
    getRecommendedAlgorithms,

    // 状態（定数ベースなので常に成功状態）
    isLoading: false,
    error: null,
    refetch: () => Promise.resolve(), // 何もしない関数

    // ユーティリティ
    getTypeLabel: (type: string) =>
      ALGORITHM_TYPE_LABELS[type as AlgorithmType] || type,
    getCapabilityLabel: (capability: string) =>
      CAPABILITY_LABELS[capability as AlgorithmCapability] || capability,
  };
};

// 個別アルゴリズム情報取得用フック（定数ベース）
export const useAlgorithmInfo = (algorithmName: string | null) => {
  const algorithmInfo = useMemo(() => {
    if (!algorithmName) return null;
    return ALGORITHMS[algorithmName] || null;
  }, [algorithmName]);

  return {
    algorithmInfo,
    isLoading: false,
    error:
      algorithmName && !algorithmInfo
        ? `アルゴリズム '${algorithmName}' が見つかりません`
        : null,
    refetch: () => Promise.resolve(),
  };
};
