"use client";

import { useEffect } from "react";
import { useDataFetching } from "@/hooks/useDataFetching";

export type FeatureAnalysisData = {
  top_features: Array<{
    feature_name: string;
    importance: number;
    feature_type: string;
    category: string;
    description: string;
  }>;
  type_statistics: Record<string, any>;
  category_statistics: Record<string, any>;
  automl_impact: Record<string, any>;
  total_features: number;
  analysis_summary: string;
};

type UseAutoMLFeatureAnalysisReturn = {
  data: FeatureAnalysisData | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
};

export function useAutoMLFeatureAnalysis(
  topN: number = 20,
  autoRefreshInterval?: number
): UseAutoMLFeatureAnalysisReturn {
  const {
    data,
    loading,
    error,
    refetch,
    setParams,
  } = useDataFetching<FeatureAnalysisData>({
    endpoint: "/api/ml/automl-feature-analysis",
    transform: (response: any) => [response as FeatureAnalysisData],
    initialParams: { top_n: topN },
    errorMessage: "AutoML特徴量分析の取得に失敗しました",
    dependencies: [topN],
  });

  // topN の変更に合わせてクエリパラメータを更新
  useEffect(() => {
    setParams({ top_n: topN });
  }, [topN, setParams]);

  // 既存の useModelInfo と同様にインターバルで自動再取得をサポート
  useEffect(() => {
    if (!autoRefreshInterval || autoRefreshInterval <= 0) return;
    const interval = setInterval(refetch, autoRefreshInterval * 1000);
    return () => clearInterval(interval);
  }, [autoRefreshInterval, refetch]);

  // useDataFetching は配列を返す設計のため、先頭要素を使う
  const analysis = Array.isArray(data) && data.length > 0 ? data[0] : null;

  return {
    data: analysis as FeatureAnalysisData | null,
    loading,
    error,
    refetch,
  };
}

export default useAutoMLFeatureAnalysis;