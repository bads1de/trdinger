"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import LoadingSpinner from "@/components/common/LoadingSpinner";
import ErrorDisplay from "@/components/common/ErrorDisplay";
import { useApiCall } from "@/hooks/useApiCall";
import { TrendingUp, BarChart3, RefreshCw } from "lucide-react";

interface FeatureImportanceData {
  feature_name: string;
  importance: number;
  rank: number;
}

interface FeatureImportanceResponse {
  feature_importance: Record<string, number>;
}

interface FeatureImportanceChartProps {
  /** 表示する特徴量の数 */
  topN?: number;
  /** チャートの高さ */
  height?: number;
  /** 自動更新間隔（秒） */
  autoRefreshInterval?: number;
  /** カスタムクラス名 */
  className?: string;
}

/**
 * 特徴量重要度可視化コンポーネント
 *
 * MLモデルの特徴量重要度を棒グラフで表示し、
 * インタラクティブな機能を提供します。
 */
export default function FeatureImportanceChart({
  topN = 10,
  height = 400,
  autoRefreshInterval,
  className = "",
}: FeatureImportanceChartProps) {
  const [displayCount, setDisplayCount] = useState(topN);
  const [sortOrder, setSortOrder] = useState<"desc" | "asc">("desc");

  const {
    execute: fetchFeatureImportance,
    loading,
    error,
    reset,
  } = useApiCall<FeatureImportanceResponse>();

  const [data, setData] = useState<FeatureImportanceData[]>([]);

  // データ取得関数
  const loadFeatureImportance = async () => {
    reset();
    await fetchFeatureImportance(
      `/api/ml/feature-importance?top_n=${displayCount}`,
      {
        method: "GET",
        onSuccess: (response) => {
          console.log("FeatureImportanceChart - APIレスポンス:", response);
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
            console.log(
              "FeatureImportanceChart - 処理済みデータ:",
              formattedData
            );
          } else {
            console.warn(
              "FeatureImportanceChart - feature_importanceが見つかりません:",
              response
            );
          }
        },
        onError: (errorMessage) => {
          console.error("特徴量重要度取得エラー:", errorMessage);
        },
      }
    );
  };

  // 初期データ読み込み
  useEffect(() => {
    loadFeatureImportance();
  }, [displayCount, sortOrder]);

  // 自動更新設定
  useEffect(() => {
    if (autoRefreshInterval && autoRefreshInterval > 0) {
      const interval = setInterval(
        loadFeatureImportance,
        autoRefreshInterval * 1000
      );
      return () => clearInterval(interval);
    }
  }, [autoRefreshInterval, displayCount, sortOrder]);

  // チャートデータの処理
  const chartData = useMemo(() => {
    return data.map((item, index) => ({
      ...item,
      // 特徴量名を短縮表示用に加工
      shortName:
        item.feature_name.length > 15
          ? `${item.feature_name.substring(0, 12)}...`
          : item.feature_name,
      // 重要度を百分率に変換
      importancePercent: (item.importance * 100).toFixed(2),
      // カラーグラデーション用のインデックス
      colorIndex: index,
    }));
  }, [data]);

  // カスタムツールチップ
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-gray-900/95 border border-gray-700 rounded-lg p-3 shadow-lg">
          <p className="text-cyan-300 font-medium mb-1">{data.feature_name}</p>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">重要度:</span>
              <span className="text-white font-medium">
                {data.importancePercent}%
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">順位:</span>
              <Badge variant="outline" className="text-xs">
                #{data.rank}
              </Badge>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  // 色の生成（重要度に基づくグラデーション）
  const getBarColor = (index: number, total: number) => {
    const intensity = 1 - index / total;
    const hue = 180 + intensity * 60; // 青緑から青へのグラデーション
    return `hsl(${hue}, 70%, ${50 + intensity * 20}%)`;
  };

  if (loading) {
    return (
      <Card className={`bg-gray-900/50 border-gray-800 ${className}`}>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <BarChart3 className="h-5 w-5 text-cyan-400" />
            <span>特徴量重要度</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <LoadingSpinner text="特徴量重要度を読み込んでいます..." size="md" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={`bg-gray-900/50 border-gray-800 ${className}`}>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <BarChart3 className="h-5 w-5 text-cyan-400" />
            <span>特徴量重要度</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <ErrorDisplay message={error} onRetry={loadFeatureImportance} />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card
      className={`bg-gray-900/50 border-gray-800 transition-all duration-300 hover:border-cyan-500/60 ${className}`}
    >
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <BarChart3 className="h-5 w-5 text-cyan-400" />
            <span>特徴量重要度</span>
            <Badge variant="outline" className="ml-2">
              Top {displayCount}
            </Badge>
          </CardTitle>

          <div className="flex items-center space-x-2">
            {/* 表示件数選択 */}
            <select
              value={displayCount}
              onChange={(e) => setDisplayCount(Number(e.target.value))}
              className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white focus:border-cyan-500 focus:outline-none"
            >
              <option value={5}>Top 5</option>
              <option value={10}>Top 10</option>
              <option value={15}>Top 15</option>
              <option value={20}>Top 20</option>
            </select>

            {/* ソート順切り替え */}
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                setSortOrder(sortOrder === "desc" ? "asc" : "desc")
              }
              className="border-gray-700 hover:border-cyan-500"
            >
              <TrendingUp
                className={`h-4 w-4 ${sortOrder === "asc" ? "rotate-180" : ""}`}
              />
            </Button>

            {/* 手動更新 */}
            <Button
              variant="outline"
              size="sm"
              onClick={loadFeatureImportance}
              className="border-gray-700 hover:border-cyan-500"
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {data.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-gray-400">
            <BarChart3 className="h-12 w-12 mb-4 opacity-50" />
            <p className="text-lg font-medium mb-2">特徴量データがありません</p>
            <p className="text-sm text-center">
              モデルを学習すると特徴量重要度が表示されます
            </p>
          </div>
        ) : (
          <div style={{ height }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                barCategoryGap="20%"
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#374151"
                  opacity={0.3}
                />

                <XAxis
                  dataKey="shortName"
                  tick={{ fill: "#9CA3AF", fontSize: 12 }}
                  tickLine={{ stroke: "#6B7280" }}
                  axisLine={{ stroke: "#6B7280" }}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />

                <YAxis
                  tick={{ fill: "#9CA3AF", fontSize: 12 }}
                  tickLine={{ stroke: "#6B7280" }}
                  axisLine={{ stroke: "#6B7280" }}
                  tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
                />

                <Tooltip content={<CustomTooltip />} />

                <Bar
                  dataKey="importance"
                  radius={[4, 4, 0, 0]}
                  stroke="#0891b2"
                  strokeWidth={1}
                >
                  {chartData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={getBarColor(index, chartData.length)}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* 統計情報 */}
        {data.length > 0 && (
          <div className="mt-4 pt-4 border-t border-gray-700">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="text-center">
                <p className="text-gray-400">特徴量数</p>
                <p className="text-white font-medium">{data.length}</p>
              </div>
              <div className="text-center">
                <p className="text-gray-400">最高重要度</p>
                <p className="text-cyan-400 font-medium">
                  {(Math.max(...data.map((d) => d.importance)) * 100).toFixed(
                    2
                  )}
                  %
                </p>
              </div>
              <div className="text-center">
                <p className="text-gray-400">平均重要度</p>
                <p className="text-white font-medium">
                  {(
                    (data.reduce((sum, d) => sum + d.importance, 0) /
                      data.length) *
                    100
                  ).toFixed(2)}
                  %
                </p>
              </div>
              <div className="text-center">
                <p className="text-gray-400">ソート順</p>
                <Badge variant={sortOrder === "desc" ? "default" : "secondary"}>
                  {sortOrder === "desc" ? "降順" : "昇順"}
                </Badge>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
