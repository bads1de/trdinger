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

/**
 * 特徴量重要度管理フック
 *
 * 機械学習モデルの特徴量重要度を取得・管理します。
 * 表示件数の変更、ソート順の切り替え、自動更新などの機能をサポートします。
 * チャート表示用のデータ変換も提供します。
 *
 * @example
 * ```tsx
 * const {
 *   data,
 *   chartData,
 *   loading,
 *   error,
 *   displayCount,
 *   sortOrder,
 *   setDisplayCount,
 *   setSortOrder,
 *   loadFeatureImportance
 * } = useFeatureImportance(20, 30);
 *
 * // 表示件数を変更
 * setDisplayCount(50);
 *
 * // ソート順を切り替え
 * setSortOrder('asc');
 *
 * // 手動で再取得
 * loadFeatureImportance();
 * ```
 *
 * @param {number} topN - 取得する上位特徴量の数
 * @param {number} [autoRefreshInterval] - 自動更新間隔（秒）、指定しない場合は自動更新しない
 * @returns {{
 *   data: FeatureImportanceData[],
 *   chartData: any[],
 *   loading: boolean,
 *   error: string | null,
 *   displayCount: number,
 *   sortOrder: "desc" | "asc",
 *   setDisplayCount: (count: number) => void,
 *   setSortOrder: (order: "desc" | "asc") => void,
 *   loadFeatureImportance: () => Promise<void>,
 *   getBarColor: (index: number) => string
 * }} 特徴量重要度管理関連の状態と操作関数
 */
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
    const totalImportance = data.reduce((sum, d) => sum + d.importance, 0);

    return data.map((item, index) => ({
      ...item,
      shortName:
        item.feature_name.length > 15
          ? `${item.feature_name.substring(0, 12)}...`
          : item.feature_name,
      // 正規化された重要度をパーセンテージで表示（相対値）
      importancePercent:
        maxImportance > 0
          ? ((item.importance / maxImportance) * 100).toFixed(2)
          : "0.00",
      // 絶対値での重要度をパーセンテージで表示
      absoluteImportancePercent:
        totalImportance > 0
          ? ((item.importance / totalImportance) * 100).toFixed(2)
          : "0.00",
      // チャート用には正規化された値（0-1の範囲）を使用
      normalizedImportance:
        maxImportance > 0 ? item.importance / maxImportance : 0,
      // 実際の重要度値
      rawImportance: item.importance.toFixed(6),
      colorIndex: index,
    }));
  }, [data]);

  return {
    /** 特徴量重要度データの配列 */
    data,
    /** チャート表示用の変換済みデータ */
    chartData,
    /** データ取得中のローディング状態 */
    loading,
    /** エラーメッセージ */
    error,
    /** 表示件数 */
    displayCount,
    /** ソート順 */
    sortOrder,
    /** 表示件数を設定する関数 */
    setDisplayCount,
    /** ソート順を設定する関数 */
    setSortOrder,
    /** 特徴量重要度を再取得する関数 */
    loadFeatureImportance,
    /** バーの色を取得する関数 */
    getBarColor,
  };
};
