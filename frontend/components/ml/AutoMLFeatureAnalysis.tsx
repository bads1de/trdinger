"use client";

import React from "react";
import useAutoMLFeatureAnalysis from "@/hooks/useAutoMLFeatureAnalysis";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import LoadingSpinner from "@/components/common/LoadingSpinner";
import ErrorDisplay from "@/components/common/ErrorDisplay";
import {
  Bot,
  TrendingUp,
  Brain,
  RefreshCw,
  Info,
  Activity,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart as RechartsPieChart,
  Pie,
  Cell,
} from "recharts";

interface AutoMLFeatureAnalysisProps {
  /** 分析する上位特徴量数 */
  topN?: number;
  /** 自動更新間隔（秒） */
  autoRefreshInterval?: number;
  /** カスタムクラス名 */
  className?: string;
}

interface FeatureAnalysisData {
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
}

const FEATURE_TYPE_COLORS = {
  manual: "#10b981", // green
  tsfresh: "#3b82f6", // blue
  featuretools: "#8b5cf6", // purple
  autofeat: "#f59e0b", // amber
  unknown: "#6b7280", // gray
};

const CATEGORY_COLORS = {
  statistical: "#ef4444", // red
  frequency: "#06b6d4", // cyan
  temporal: "#84cc16", // lime
  aggregation: "#f97316", // orange
  interaction: "#ec4899", // pink
  genetic: "#8b5cf6", // purple
  manual: "#10b981", // green
  unknown: "#6b7280", // gray
};

export default function AutoMLFeatureAnalysis({
  topN = 20,
  autoRefreshInterval,
  className = "",
}: AutoMLFeatureAnalysisProps) {
  const { data, loading, error, refetch } = useAutoMLFeatureAnalysis(
    topN,
    autoRefreshInterval
  );

  // タイプ別統計をチャート用データに変換
  const typeChartData = data?.type_statistics
    ? Object.entries(data.type_statistics).map(
        ([type, stats]: [string, any]) => ({
          name: type,
          count: stats.count,
          importance: stats.total_importance,
          percentage: stats.percentage,
          color:
            FEATURE_TYPE_COLORS[type as keyof typeof FEATURE_TYPE_COLORS] ||
            "#6b7280",
        })
      )
    : [];

  // カテゴリ別統計をチャート用データに変換
  const categoryChartData = data?.category_statistics
    ? Object.entries(data.category_statistics).map(
        ([category, stats]: [string, any]) => ({
          name: category,
          count: stats.count,
          importance: stats.total_importance,
          percentage: stats.percentage,
          color:
            CATEGORY_COLORS[category as keyof typeof CATEGORY_COLORS] ||
            "#6b7280",
        })
      )
    : [];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg">
          <p className="text-white font-medium">{label}</p>
          <p className="text-cyan-400">
            特徴量数: {data.count}個 ({data.percentage.toFixed(1)}%)
          </p>
          <p className="text-green-400">
            重要度合計: {data.importance.toFixed(3)}
          </p>
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <Card className={`bg-gray-900/50 border-gray-800 ${className}`}>
        <CardContent className="flex items-center justify-center py-8">
          <LoadingSpinner size="lg" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={`bg-gray-900/50 border-gray-800 ${className}`}>
        <CardContent className="py-8">
          <ErrorDisplay message={error} />
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card className={`bg-gray-900/50 border-gray-800 ${className}`}>
        <CardContent className="flex flex-col items-center justify-center py-8 text-gray-400">
          <Bot className="h-12 w-12 mb-4 opacity-50" />
          <p className="text-lg font-medium mb-2">
            AutoML分析データがありません
          </p>
          <p className="text-sm text-center">
            AutoML機能を有効にしてモデルを学習すると分析結果が表示されます
          </p>
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
            <Bot className="h-5 w-5 text-cyan-400" />
            <span>AutoML特徴量分析</span>
            <Badge variant="outline" className="ml-2">
              {data?.total_features ?? 0}個の特徴量
            </Badge>
          </CardTitle>
          <Button
            variant="outline"
            size="sm"
            onClick={refetch}
            disabled={loading}
          >
            <RefreshCw
              className={`h-4 w-4 mr-1 ${loading ? "animate-spin" : ""}`}
            />
            更新
          </Button>
        </div>
      </CardHeader>

      <CardContent>
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">概要</TabsTrigger>
            <TabsTrigger value="types">タイプ別</TabsTrigger>
            <TabsTrigger value="categories">カテゴリ別</TabsTrigger>
            <TabsTrigger value="top-features">上位特徴量</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            {/* AutoML効果サマリー */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="bg-gray-800/50 border-gray-700">
                <CardContent className="p-4">
                  <div className="flex items-center space-x-2">
                    <Activity className="h-5 w-5 text-green-400" />
                    <div>
                      <p className="text-sm text-gray-400">AutoML貢献度</p>
                      <p className="text-lg font-bold text-white">
                        {(
                          data?.automl_impact?.automl_importance_ratio ?? 0
                        ).toFixed(1)}
                        %
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gray-800/50 border-gray-700">
                <CardContent className="p-4">
                  <div className="flex items-center space-x-2">
                    <Brain className="h-5 w-5 text-blue-400" />
                    <div>
                      <p className="text-sm text-gray-400">AutoML特徴量</p>
                      <p className="text-lg font-bold text-white">
                        {data?.automl_impact?.automl_features ?? 0}個
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gray-800/50 border-gray-700">
                <CardContent className="p-4">
                  <div className="flex items-center space-x-2">
                    <TrendingUp className="h-5 w-5 text-purple-400" />
                    <div>
                      <p className="text-sm text-gray-400">手動特徴量</p>
                      <p className="text-lg font-bold text-white">
                        {data?.automl_impact?.manual_features ?? 0}個
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* 分析サマリー */}
            <Card className="bg-gray-800/50 border-gray-700">
              <CardContent className="p-4">
                <div className="flex items-start space-x-2">
                  <Info className="h-5 w-5 text-cyan-400 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-white mb-2">
                      分析サマリー
                    </h4>
                    <pre className="text-sm text-gray-300 whitespace-pre-wrap">
                      {data.analysis_summary}
                    </pre>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="types" className="space-y-4">
            <div style={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={typeChartData}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#374151"
                    opacity={0.3}
                  />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: "#9CA3AF", fontSize: 12 }}
                  />
                  <YAxis tick={{ fill: "#9CA3AF", fontSize: 12 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {typeChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>

          <TabsContent value="categories" className="space-y-4">
            <div style={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <RechartsPieChart>
                  <Pie
                    data={categoryChartData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="count"
                    label={({ name, percentage }) =>
                      `${name} (${percentage.toFixed(1)}%)`
                    }
                  >
                    {categoryChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </RechartsPieChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>

          <TabsContent value="top-features" className="space-y-4">
            <div className="space-y-2">
              {(data?.top_features ?? []).slice(0, 10).map((feature, index) => (
                <div
                  key={feature.feature_name}
                  className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg border border-gray-700"
                >
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-medium text-white">
                        #{index + 1} {feature.feature_name}
                      </span>
                      <Badge
                        variant="outline"
                        style={{
                          borderColor:
                            FEATURE_TYPE_COLORS[
                              feature.feature_type as keyof typeof FEATURE_TYPE_COLORS
                            ],
                          color:
                            FEATURE_TYPE_COLORS[
                              feature.feature_type as keyof typeof FEATURE_TYPE_COLORS
                            ],
                        }}
                      >
                        {feature.feature_type}
                      </Badge>
                    </div>
                    <p className="text-xs text-gray-400 mt-1">
                      {feature.description ?? ""}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-medium text-cyan-400">
                      {(feature.importance ?? 0).toFixed(4)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
