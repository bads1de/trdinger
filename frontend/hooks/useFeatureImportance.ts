import { useState, useEffect, useMemo } from "react";
import { useDataFetching } from "./useDataFetching";
import { getBarColor } from "@/utils/colorUtils";

interface FeatureImportanceData {
  feature_name: string;
  importance: number;
  rank: number;
}

interface FeatureImportanceParams {
  top_n: number;
}

export const useFeatureImportance = (
  topN: number,
  autoRefreshInterval?: number
) => {
  const [displayCount, setDisplayCount] = useState(topN);
  const [sortOrder, setSortOrder] = useState<"desc" | "asc">("desc");

  // 基本的なデータ取得は共通フックを使用
  const {
    data,
    loading,
    error,
    params,
    setParams,
    refetch: loadFeatureImportance,
  } = useDataFetching<FeatureImportanceData, FeatureImportanceParams>({
    endpoint: "/api/ml/feature-importance",
    initialParams: { top_n: displayCount },
    transform: (response) => {
      if (response?.feature_importance) {
        return Object.entries(response.feature_importance)
          .map(([feature_name, importance], index) => ({
            feature_name,
            importance: Number(importance),
            rank: index + 1,
          }))
          .sort((a, b) =>
            sortOrder === "desc"
              ? b.importance - a.importance
              : a.importance - b.importance
          );
      }
      return [];
    },
    dependencies: [displayCount, sortOrder],
    errorMessage: "特徴量重要度の取得中にエラーが発生しました",
  });

  // displayCountが変更されたらパラメータを更新
  useEffect(() => {
    setParams({ top_n: displayCount });
  }, [displayCount, setParams]);

  useEffect(() => {
    if (autoRefreshInterval && autoRefreshInterval > 0) {
      const interval = setInterval(
        loadFeatureImportance,
        autoRefreshInterval * 1000
      );
      return () => clearInterval(interval);
    }
  }, [autoRefreshInterval, loadFeatureImportance]);

  const chartData = useMemo(() => {
    // 特徴量重要度を正規化（最大値を100%とする）
    const maxImportance = Math.max(...data.map((d) => d.importance));

    return data.map((item, index) => ({
      ...item,
      shortName:
        item.feature_name.length > 15
          ? `${item.feature_name.substring(0, 12)}...`
          : item.feature_name,
      // 正規化された重要度をパーセンテージで表示
      importancePercent:
        maxImportance > 0
          ? ((item.importance / maxImportance) * 100).toFixed(2)
          : "0.00",
      // チャート用には正規化された値（0-1の範囲）を使用
      normalizedImportance:
        maxImportance > 0 ? item.importance / maxImportance : 0,
      colorIndex: index,
    }));
  }, [data]);

  return {
    data,
    chartData,
    loading,
    error,
    displayCount,
    sortOrder,
    setDisplayCount,
    setSortOrder,
    loadFeatureImportance,
    getBarColor,
  };
};
