"use client";

import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import ErrorDisplay from "@/components/common/ErrorDisplay";
import LoadingSpinner from "@/components/common/LoadingSpinner";
import { useMLModelStatus } from "@/hooks/useMLModelStatus";
import { formatProbability } from "@/utils/formatters";
import { Brain, TrendingUp, BarChart3, Activity } from "lucide-react";

/**
 * MLモデル状態表示コンポーネント
 *
 * 現在のモデルの状態、精度、特徴量重要度などを表示するダッシュボード
 */
export default function MLModelStatus() {
  const { modelStatus, featureImportance, isLoading, error } =
    useMLModelStatus();

  const getStatusBadge = () => {
    if (!modelStatus) return null;

    if (modelStatus.is_model_loaded && modelStatus.is_trained) {
      return <Badge className="bg-green-100 text-green-800">アクティブ</Badge>;
    } else if (modelStatus.is_model_loaded) {
      return (
        <Badge className="bg-yellow-100 text-yellow-800">読み込み済み</Badge>
      );
    } else {
      return (
        <Badge variant="secondary" className="mt-1">
          未読み込み
        </Badge>
      );
    }
  };

  if (isLoading) {
    return <LoadingSpinner text="モデル状態を読み込んでいます..." />;
  }

  if (error) {
    return <ErrorDisplay message={error} />;
  }

  if (!modelStatus) {
    return (
      <div className="text-center p-8 text-gray-500">
        <Brain className="h-12 w-12 mx-auto mb-4 text-gray-300" />
        <p>モデル情報を取得できませんでした</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* モデル基本情報 */}
      <div className="grid grid-cols-1 gap-4">
        {modelStatus.last_predictions && (
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-2 mb-3">
                <TrendingUp className="h-5 w-5 text-blue-600" />
                <h3 className="font-medium text-gray-900">最新予測</h3>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">上昇確率</span>
                  <span className="font-medium text-green-600">
                    {formatProbability(modelStatus.last_predictions.up)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">下落確率</span>
                  <span className="font-medium text-red-600">
                    {formatProbability(modelStatus.last_predictions.down)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">レンジ確率</span>
                  <span className="font-medium text-blue-600">
                    {formatProbability(modelStatus.last_predictions.range)}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* モデル詳細情報 */}
      {modelStatus.model_info && (
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <BarChart3 className="h-5 w-5 text-purple-600" />
                <h3 className="font-medium text-gray-900">モデル詳細</h3>
              </div>
              {getStatusBadge()}
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {(modelStatus.model_info?.accuracy ?? 0 * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600">精度</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {modelStatus.model_info.model_type}
                </div>
                <div className="text-sm text-gray-600">モデルタイプ</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {modelStatus.model_info?.training_samples?.toLocaleString()}
                </div>
                <div className="text-sm text-gray-600">学習サンプル</div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-600">
                  {new Date(
                    modelStatus.model_info?.last_updated ?? new Date()
                  ).toLocaleDateString("ja-JP")}
                </div>
                <div className="text-sm text-gray-600">最終更新</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 特徴量重要度 */}
      {Object.keys(featureImportance).length > 0 && (
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2 mb-4">
              <Activity className="h-5 w-5 text-orange-600" />
              <h3 className="font-medium text-gray-900">
                特徴量重要度 (上位10個)
              </h3>
            </div>
            <div className="space-y-2">
              {Object.entries(featureImportance)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 10)
                .map(([feature, importance]) => (
                  <div key={feature} className="flex items-center space-x-2">
                    <div className="flex-1 flex items-center justify-between">
                      <span className="text-sm text-gray-600 truncate">
                        {feature}
                      </span>
                      <span className="text-sm font-medium text-gray-900">
                        {importance.toFixed(3)}
                      </span>
                    </div>
                    <div className="w-24 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{
                          width: `${
                            (importance /
                              Math.max(...Object.values(featureImportance))) *
                            100
                          }%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
