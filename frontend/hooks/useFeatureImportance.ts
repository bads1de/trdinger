import { useState, useEffect, useCallback, useMemo } from "react";
import { useApiCall } from "@/hooks/useApiCall";
import { getBarColor } from "@/utils/colorUtils";

interface FeatureImportanceData {
  feature_name: string;
  importance: number;
  rank: number;
}

interface FeatureImportanceResponse {
  feature_importance: Record<string, number>;
}

export const useFeatureImportance = (
  topN: number,
  autoRefreshInterval?: number
) => {
  const [displayCount, setDisplayCount] = useState(topN);
  const [sortOrder, setSortOrder] = useState<"desc" | "asc">("desc");
  const [data, setData] = useState<FeatureImportanceData[]>([]);

  const {
    execute: fetchFeatureImportance,
    loading,
    error,
    reset,
  } = useApiCall<FeatureImportanceResponse>();

  const loadFeatureImportance = useCallback(async () => {
    reset();
    await fetchFeatureImportance(
      `/api/ml/feature-importance?top_n=${displayCount}`,
      {
        method: "GET",
        onSuccess: (response) => {
          if (response?.feature_importance) {
            const formattedData = Object.entries(response.feature_importance)
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
            setData(formattedData);
          }
        },
        onError: (errorMessage) => {
          console.error("特徴量重要度取得エラー:", errorMessage);
        },
      }
    );
  }, [displayCount, sortOrder, fetchFeatureImportance, reset]);

  useEffect(() => {
    loadFeatureImportance();
  }, [loadFeatureImportance]);

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
