"use client";

/**
 * AutoML特徴量分析用カスタムフック
 *
 * AutoMLによる特徴量分析データを取得・管理するためのフックです。
 * 指定された上位N個の特徴量の重要度分析結果を取得し、
 * 自動更新機能もサポートします。
 */

import { useEffect } from "react";
import { useDataFetching } from "@/hooks/useDataFetching";

/**
 * AutoML特徴量分析データの型
 */
export type FeatureAnalysisData = {
  /** 上位特徴量のリスト */
  top_features: Array<{
    /** 特徴量名 */
    feature_name: string;
    /** 重要度スコア */
    importance: number;
    /** 特徴量のタイプ */
    feature_type: string;
    /** 特徴量のカテゴリ */
    category: string;
    /** 特徴量の説明 */
    description: string;
  }>;
  /** タイプ別統計情報 */
  type_statistics: Record<string, any>;
  /** カテゴリ別統計情報 */
  category_statistics: Record<string, any>;
  /** AutoMLの影響度分析 */
  automl_impact: Record<string, any>;
  /** 総特徴量数 */
  total_features: number;
  /** 分析サマリー */
  analysis_summary: string;
};

/**
 * AutoML特徴量分析フックの戻り値の型
 */
type UseAutoMLFeatureAnalysisReturn = {
  /** 特徴量分析データ */
  data: FeatureAnalysisData | null;
  /** データ取得中かどうか */
  loading: boolean;
  /** エラーメッセージ */
  error: string | null;
  /** データを再取得する関数 */
  refetch: () => Promise<void>;
};

/**
 * AutoML特徴量分析フック
 *
 * AutoMLによる特徴量分析データを取得・管理します。
 * 指定された上位N個の特徴量の重要度分析結果を取得し、
 * 自動更新機能もサポートします。
 *
 * @example
 * ```tsx
 * const { data, loading, error, refetch } = useAutoMLFeatureAnalysis(20, 30);
 * if (loading) return <LoadingSpinner />;
 * if (error) return <ErrorDisplay error={error} />;
 * return <FeatureAnalysisChart data={data} />;
 * ```
 *
 * @param {number} topN - 取得する上位特徴量の数（デフォルト: 20）
 * @param {number} [autoRefreshInterval] - 自動更新間隔（秒）、指定しない場合は自動更新しない
 * @returns {UseAutoMLFeatureAnalysisReturn} 特徴量分析データと操作関数
 */
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