"use client";

import React, { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import FeatureImportanceChart from "./FeatureImportanceChart";
import ModelInfoCard from "./ModelInfoCard";
import ModelPerformanceCard from "./ModelPerformanceCard";
import ModelManagement from "./ModelManagement";
import { useMLModels } from "@/hooks/useMLModels";
import {
  Brain,
  Database,
  TrendingUp,
  BarChart3,
  RefreshCw,
  Activity,
} from "lucide-react";

interface MLOverviewDashboardProps {
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
          <div className="pt-1">
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
          <CardContent className="pt-2">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">総モデル数</p>
                <p className="p-1 text-2xl font-bold text-white">
                  {stats.totalModels}
                </p>
              </div>
              <Database className="h-10 w-10 text-cyan-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="pt-2">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">アクティブモデル</p>
                <p className="p-1 text-2xl font-bold text-green-400">
                  {stats.activeModels}
                </p>
              </div>
              <Activity className="h-10 w-10 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="pt-2">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">総ファイルサイズ</p>
                <p className="p-1 text-2xl font-bold text-purple-400">
                  {stats.totalSize.toFixed(1)} MB
                </p>
              </div>
              <BarChart3 className="h-10 w-10 text-purple-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="pt-2">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">最新モデル</p>
                <p className="p-1text-sm font-medium text-white truncate max-w-24">
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
              <TrendingUp className="h-10 w-10 text-yellow-400" />
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

      {/* モデル管理セクション */}
      <ModelManagement key={`model-management-${refreshKey}`} />

      {/* フッター情報 */}
      <div className="text-center text-xs text-gray-500 pt-4 border-t border-gray-800">
        <p>「更新」ボタンで、モデルの状態を手動で更新できます。</p>
      </div>
    </div>
  );
}
