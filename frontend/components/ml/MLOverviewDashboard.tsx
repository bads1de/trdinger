"use client";

import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import LoadingSpinner from "@/components/common/LoadingSpinner";
import ErrorDisplay from "@/components/common/ErrorDisplay";
import FeatureImportanceChart from "./FeatureImportanceChart";
import ModelInfoCard from "./ModelInfoCard";
import ModelPerformanceCard from "./ModelPerformanceCard";
import MLModelList from "./MLModelList";
import { useMLModels } from "@/hooks/useMLModels";
import {
  Brain,
  Database,
  TrendingUp,
  BarChart3,
  RefreshCw,
  Settings,
  Activity,
  AlertTriangle,
} from "lucide-react";

interface MLOverviewDashboardProps {
  /** カスタムクラス名 */
  className?: string;
}

/**
 * ML概要ダッシュボードコンポーネント
 *
 * 機械学習モデルの詳細情報を統合的に可視化するダッシュボード
 */
export default function MLOverviewDashboard({
  className = "",
}: MLOverviewDashboardProps) {
  const [refreshKey, setRefreshKey] = useState(0);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  // モデル一覧データを取得
  const {
    models,
    isLoading: modelsLoading,
    error: modelsError,
    fetchModels,
  } = useMLModels(5);

  // 手動更新関数
  const handleManualRefresh = () => {
    setRefreshKey((prev) => prev + 1);
    setLastRefresh(new Date());
    fetchModels();
  };

  // 統計情報の計算
  const stats = {
    totalModels: models.length,
    activeModels: models.filter((model) => model.is_active).length,
    latestModel: models.length > 0 ? models[0] : null,
    totalSize: models.reduce((sum, model) => sum + (model.size_mb || 0), 0),
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* ダッシュボードヘッダー */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Brain className="h-6 w-6 text-cyan-400" />
          <div>
            <h2 className="text-xl font-bold text-foreground">
              ML概要ダッシュボード
            </h2>
            <p className="text-sm text-muted-foreground">
              機械学習モデルの詳細情報と性能指標
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <div className="text-right text-xs text-gray-500">
            <p>最終更新: {lastRefresh.toLocaleTimeString("ja-JP")}</p>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={handleManualRefresh}
            className="border-gray-700 hover:border-cyan-500"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            更新
          </Button>
        </div>
      </div>

      {/* 統計サマリーカード */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">総モデル数</p>
                <p className="text-2xl font-bold text-white">
                  {stats.totalModels}
                </p>
              </div>
              <Database className="h-8 w-8 text-cyan-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">アクティブモデル</p>
                <p className="text-2xl font-bold text-green-400">
                  {stats.activeModels}
                </p>
              </div>
              <Activity className="h-8 w-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">総ファイルサイズ</p>
                <p className="text-2xl font-bold text-purple-400">
                  {stats.totalSize.toFixed(1)} MB
                </p>
              </div>
              <BarChart3 className="h-8 w-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">最新モデル</p>
                <p className="text-sm font-medium text-white truncate max-w-24">
                  {stats.latestModel?.name || "なし"}
                </p>
                {stats.latestModel && (
                  <p className="text-xs text-gray-500">
                    {new Date(stats.latestModel.modified_at).toLocaleDateString(
                      "ja-JP"
                    )}
                  </p>
                )}
              </div>
              <TrendingUp className="h-8 w-8 text-yellow-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* メインコンテンツグリッド */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* 左列 */}
        <div className="space-y-6">
          {/* モデル基本情報 */}
          <ModelInfoCard
            key={`model-info-${refreshKey}`}
            defaultExpanded={false}
          />

          {/* 性能指標 */}
          <ModelPerformanceCard key={`model-performance-${refreshKey}`} />
        </div>

        {/* 右列 */}
        <div className="space-y-6">
          {/* 特徴量重要度チャート */}
          <FeatureImportanceChart
            key={`feature-importance-${refreshKey}`}
            topN={10}
            height={400}
          />
        </div>
      </div>

      {/* 最近のモデル一覧 */}
      <Card className="bg-gray-900/50 border-gray-800">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center space-x-2">
              <Database className="h-5 w-5 text-cyan-400" />
              <span>最近のモデル</span>
              <Badge variant="outline" className="ml-2">
                最新 {models.length} 件
              </Badge>
            </CardTitle>

            {modelsError && (
              <div className="flex items-center space-x-2 text-red-400">
                <AlertTriangle className="h-4 w-4" />
                <span className="text-sm">データ取得エラー</span>
              </div>
            )}
          </div>
        </CardHeader>

        <CardContent>
          {modelsLoading ? (
            <LoadingSpinner text="モデル一覧を読み込んでいます..." size="md" />
          ) : modelsError ? (
            <ErrorDisplay message={modelsError} onRetry={fetchModels} />
          ) : models.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-gray-400">
              <Database className="h-12 w-12 mb-4 opacity-50" />
              <p className="text-lg font-medium mb-2">モデルがありません</p>
              <p className="text-sm text-center">
                トレーニングタブでモデルを学習してください
              </p>
            </div>
          ) : (
            <MLModelList limit={5} showActions={false} />
          )}
        </CardContent>
      </Card>

      {/* フッター情報 */}
      <div className="text-center text-xs text-gray-500 pt-4 border-t border-gray-800">
        <p>「更新」ボタンで、モデルの状態を手動で更新できます。</p>
      </div>
    </div>
  );
}
