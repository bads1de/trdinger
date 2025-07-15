"use client";

import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import LoadingSpinner from "@/components/common/LoadingSpinner";
import ErrorDisplay from "@/components/common/ErrorDisplay";
import { useModelInfo } from "@/hooks/useModelInfo";
import {
  Brain,
  Database,
  Settings,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Info,
} from "lucide-react";

interface ModelInfoCardProps {
  /** 自動更新間隔（秒） */
  autoRefreshInterval?: number;
  /** カスタムクラス名 */
  className?: string;
  /** 詳細情報の初期表示状態 */
  defaultExpanded?: boolean;
}

/**
 * モデル基本情報カードコンポーネント
 *
 * 現在のMLモデルの基本情報、設定、統計を表示します。
 */
export default function ModelInfoCard({
  autoRefreshInterval,
  className = "",
  defaultExpanded = false,
}: ModelInfoCardProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const {
    modelStatus,
    loading,
    error,
    loadModelStatus,
    formatDateTime,
    formatFileSize,
    getModelTypeBadgeVariant,
    getAccuracyBadgeVariant,
  } = useModelInfo(autoRefreshInterval);

  if (loading) {
    return (
      <Card className={`bg-gray-900/50 border-gray-800 ${className}`}>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5 text-cyan-400" />
            <span>モデル情報</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <LoadingSpinner text="モデル情報を読み込んでいます..." size="md" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={`bg-gray-900/50 border-gray-800 ${className}`}>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5 text-cyan-400" />
            <span>モデル情報</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ErrorDisplay message={error} onRetry={loadModelStatus} />
        </CardContent>
      </Card>
    );
  }

  const modelInfo = modelStatus?.model_info;

  return (
    <Card
      className={`bg-gray-900/50 border-gray-800 transition-all duration-300 hover:border-cyan-500/60 ${className}`}
    >
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5 text-cyan-400" />
            <span>モデル情報</span>
            {modelStatus?.is_model_loaded ? (
              <Badge variant="success" className="ml-2">
                読み込み済み
              </Badge>
            ) : (
              <Badge variant="outline" className="ml-2">
                未読み込み
              </Badge>
            )}
          </CardTitle>

          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={loadModelStatus}
              className="border-gray-700 hover:border-cyan-500"
            >
              <RefreshCw className="h-4 w-4" />
            </Button>

            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-gray-400 hover:text-cyan-400"
            >
              {isExpanded ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {!modelStatus?.is_model_loaded ? (
          <div className="flex flex-col items-center justify-center py-8 text-gray-400">
            <Brain className="h-12 w-12 mb-4 opacity-50" />
            <p className="text-lg font-medium mb-2">
              モデルが読み込まれていません
            </p>
            <p className="text-sm text-center">
              モデルを学習するか、既存のモデルを読み込んでください
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {/* 基本情報 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400 text-sm">モデル種別</span>
                  <Badge
                    variant={getModelTypeBadgeVariant(modelInfo?.model_type)}
                  >
                    {modelInfo?.model_type || "不明"}
                  </Badge>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-gray-400 text-sm">精度</span>
                  <Badge variant={getAccuracyBadgeVariant(modelInfo?.accuracy)}>
                    {modelInfo?.accuracy
                      ? `${(modelInfo.accuracy * 100).toFixed(2)}%`
                      : "不明"}
                  </Badge>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-gray-400 text-sm">学習サンプル数</span>
                  <span className="text-white font-medium">
                    {modelInfo?.training_samples?.toLocaleString() || "不明"}
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-gray-400 text-sm">特徴量数</span>
                  <span className="text-white font-medium">
                    {modelStatus?.feature_count || "不明"}
                  </span>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400 text-sm">ファイルサイズ</span>
                  <span className="text-white font-medium">
                    {formatFileSize(modelInfo?.file_size_mb)}
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-gray-400 text-sm">最終更新</span>
                  <span className="text-white font-medium text-xs">
                    {formatDateTime(modelInfo?.last_updated)}
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-gray-400 text-sm">学習状態</span>
                  <Badge
                    variant={modelStatus?.is_trained ? "success" : "outline"}
                  >
                    {modelStatus?.is_trained ? "学習済み" : "未学習"}
                  </Badge>
                </div>
              </div>
            </div>

            {/* 詳細情報（展開可能） */}
            {isExpanded && (
              <div className="mt-6 pt-4 border-t border-gray-700 space-y-4">
                {/* モデルパラメータ */}
                {modelInfo?.parameters &&
                  Object.keys(modelInfo.parameters).length > 0 && (
                    <div>
                      <h4 className="flex items-center text-sm font-medium text-gray-300 mb-3">
                        <Settings className="h-4 w-4 mr-2" />
                        モデルパラメータ
                      </h4>
                      <div className="bg-gray-800/50 rounded-lg p-3">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                          {Object.entries(modelInfo.parameters).map(
                            ([key, value]) => (
                              <div key={key} className="flex justify-between">
                                <span className="text-gray-400">{key}:</span>
                                <span className="text-white font-mono">
                                  {typeof value === "object"
                                    ? JSON.stringify(value)
                                    : String(value)}
                                </span>
                              </div>
                            )
                          )}
                        </div>
                      </div>
                    </div>
                  )}

                {/* 学習設定 */}
                {modelInfo?.training_config &&
                  Object.keys(modelInfo.training_config).length > 0 && (
                    <div>
                      <h4 className="flex items-center text-sm font-medium text-gray-300 mb-3">
                        <Database className="h-4 w-4 mr-2" />
                        学習設定
                      </h4>
                      <div className="bg-gray-800/50 rounded-lg p-3">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                          {Object.entries(modelInfo.training_config).map(
                            ([key, value]) => (
                              <div key={key} className="flex justify-between">
                                <span className="text-gray-400">{key}:</span>
                                <span className="text-white font-mono">
                                  {typeof value === "object"
                                    ? JSON.stringify(value)
                                    : String(value)}
                                </span>
                              </div>
                            )
                          )}
                        </div>
                      </div>
                    </div>
                  )}

                {/* 追加統計情報 */}
                <div>
                  <h4 className="flex items-center text-sm font-medium text-gray-300 mb-3">
                    <Info className="h-4 w-4 mr-2" />
                    追加情報
                  </h4>
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                      <div className="flex justify-between">
                        <span className="text-gray-400">モデルパス:</span>
                        <span className="text-white font-mono text-right max-w-48 truncate">
                          {modelStatus?.model_path || "不明"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">最終予測時刻:</span>
                        <span className="text-white font-mono">
                          {formatDateTime(modelStatus?.last_prediction_time)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
