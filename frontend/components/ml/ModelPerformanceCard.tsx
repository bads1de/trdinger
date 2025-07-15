"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import LoadingSpinner from "@/components/common/LoadingSpinner";
import ErrorDisplay from "@/components/common/ErrorDisplay";
import { useApiCall } from "@/hooks/useApiCall";
import {
  TrendingUp,
  Target,
  Activity,
  BarChart3,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Clock,
} from "lucide-react";

interface PerformanceMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  auc_score?: number;
  loss?: number;
  val_accuracy?: number;
  val_loss?: number;
  training_time?: number;
  last_evaluation?: string;
}

interface ModelStatusResponse {
  is_model_loaded: boolean;
  is_trained: boolean;
  last_predictions?: {
    up: number;
    down: number;
    range: number;
  };
  feature_count: number;
  model_info?: {
    accuracy?: number;
    model_type?: string;
    training_samples?: number;
    last_updated?: string;
  };
  performance_metrics?: PerformanceMetrics;
  is_training?: boolean;
  training_progress?: number;
  status?: string;
}

interface ModelPerformanceCardProps {
  /** カスタムクラス名 */
  className?: string;
}

/**
 * モデル性能指標カードコンポーネント
 *
 * MLモデルの性能指標、学習状態、評価結果を表示します。
 */
export default function ModelPerformanceCard({
  className = "",
}: ModelPerformanceCardProps) {
  const [modelStatus, setModelStatus] = useState<ModelStatusResponse | null>(
    null
  );

  const {
    execute: fetchModelStatus,
    loading,
    error,
    reset,
  } = useApiCall<ModelStatusResponse>();

  // データ取得関数
  const loadModelStatus = async () => {
    reset();
    await fetchModelStatus("/api/ml/status", {
      method: "GET",
      onSuccess: (response) => {
        console.log("ModelPerformanceCard - APIレスポンス:", response);
        console.log(
          "ModelPerformanceCard - is_model_loaded:",
          response?.is_model_loaded
        );
        console.log("ModelPerformanceCard - is_trained:", response?.is_trained);
        console.log("ModelPerformanceCard - model_info:", response?.model_info);
        console.log(
          "ModelPerformanceCard - feature_count:",
          response?.feature_count
        );
        setModelStatus(response);
      },
      onError: (errorMessage) => {
        console.error("モデル状態取得エラー:", errorMessage);
      },
    });
  };

  // 初期データ読み込み
  useEffect(() => {
    loadModelStatus();
  }, []);

  // 性能スコアの色を取得
  const getScoreColor = (score?: number) => {
    if (!score) return "text-gray-400";
    if (score >= 0.8) return "text-green-400";
    if (score >= 0.7) return "text-yellow-400";
    if (score >= 0.6) return "text-orange-400";
    return "text-red-400";
  };

  // 性能スコアのバッジバリアントを取得
  const getScoreBadgeVariant = (score?: number) => {
    if (!score) return "outline";
    if (score >= 0.8) return "success";
    if (score >= 0.7) return "warning";
    return "destructive";
  };

  // 学習状態のアイコンを取得
  const getStatusIcon = () => {
    if (modelStatus?.is_training) {
      return <Activity className="h-5 w-5 text-blue-400 animate-pulse" />;
    }

    if (modelStatus?.is_model_loaded) {
      return <CheckCircle className="h-5 w-5 text-green-400" />;
    }

    return <AlertCircle className="h-5 w-5 text-yellow-400" />;
  };

  // 学習状態のテキストを取得
  const getStatusText = () => {
    if (modelStatus?.is_training) {
      return `学習中 (${modelStatus.training_progress || 0}%)`;
    }

    if (modelStatus?.is_model_loaded) {
      return "準備完了";
    }

    return "未読み込み";
  };

  // 学習状態のバッジバリアントを取得
  const getStatusBadgeVariant = () => {
    if (modelStatus?.is_training) return "default";

    if (modelStatus?.is_model_loaded) return "success";

    return "outline";
  };

  // 時間フォーマット関数
  const formatTrainingTime = (seconds?: number) => {
    if (!seconds) return "不明";

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}時間${minutes}分${secs}秒`;
    }

    if (minutes > 0) {
      return `${minutes}分${secs}秒`;
    }

    return `${secs}秒`;
  };

  if (loading) {
    return (
      <Card className={`bg-gray-900/50 border-gray-800 ${className}`}>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <TrendingUp className="h-5 w-5 text-cyan-400" />
            <span>性能指標</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <LoadingSpinner text="性能指標を読み込んでいます..." size="md" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={`bg-gray-900/50 border-gray-800 ${className}`}>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <TrendingUp className="h-5 w-5 text-cyan-400" />
            <span>性能指標</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ErrorDisplay message={error} onRetry={loadModelStatus} />
        </CardContent>
      </Card>
    );
  }

  // データ変数を定義
  const metrics = modelStatus?.performance_metrics;
  const modelInfo = modelStatus?.model_info;

  return (
    <Card
      className={`bg-gray-900/50 border-gray-800 transition-all duration-300 hover:border-cyan-500/60 ${className}`}
    >
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <TrendingUp className="h-5 w-5 text-cyan-400" />
            <span>性能指標</span>
          </CardTitle>

          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-2">
              {getStatusIcon()}
              <Badge variant={getStatusBadgeVariant()}>{getStatusText()}</Badge>
            </div>

            <Button
              variant="outline"
              size="sm"
              onClick={loadModelStatus}
              className="border-gray-700 hover:border-cyan-500"
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {(() => {
          console.log("ModelPerformanceCard - 条件分岐チェック:");
          console.log("  - is_model_loaded:", modelStatus?.is_model_loaded);
          console.log("  - is_training:", modelStatus?.is_training);
          console.log(
            "  - 表示条件:",
            !modelStatus?.is_model_loaded && !modelStatus?.is_training
          );
          return null;
        })()}

        {!modelStatus?.is_model_loaded && !modelStatus?.is_training ? (
          <div className="flex flex-col items-center justify-center py-8 text-gray-400">
            <BarChart3 className="h-12 w-12 mb-4 opacity-50" />
            <p className="text-lg font-medium mb-2">性能データがありません</p>
            <p className="text-sm text-center">
              モデルを学習すると性能指標が表示されます
            </p>
          </div>
        ) : (
          <div className="space-y-6">
            {(() => {
              const metrics = modelStatus?.performance_metrics;
              const modelInfo = modelStatus?.model_info;
              console.log("ModelPerformanceCard - データ確認:");
              console.log("  - metrics:", metrics);
              console.log("  - modelInfo:", modelInfo);
              console.log("  - modelStatus:", modelStatus);
              return null;
            })()}

            {/* 学習進行状況（学習中の場合） */}
            {modelStatus?.is_training && (
              <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-blue-300 font-medium">
                    学習進行状況
                  </span>
                  <span className="text-blue-300 text-sm">
                    {modelStatus.training_progress || 0}%
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${modelStatus.training_progress || 0}%` }}
                  />
                </div>
                {modelStatus.status && (
                  <p className="text-blue-300 text-xs mt-2">
                    {modelStatus.status}
                  </p>
                )}
              </div>
            )}

            {/* 主要性能指標 */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="text-center p-3 bg-gray-800/50 rounded-lg">
                <div className="flex items-center justify-center mb-2">
                  <Target className="h-4 w-4 text-cyan-400 mr-1" />
                  <span className="text-gray-400 text-sm">精度</span>
                </div>
                <div
                  className={`text-lg font-bold ${getScoreColor(
                    modelInfo?.accuracy || metrics?.accuracy
                  )}`}
                >
                  {(() => {
                    const accuracy = modelInfo?.accuracy || metrics?.accuracy;
                    console.log("精度表示:", accuracy);
                    return accuracy ? `${(accuracy * 100).toFixed(2)}%` : "N/A";
                  })()}
                </div>
              </div>

              <div className="text-center p-3 bg-gray-800/50 rounded-lg">
                <div className="flex items-center justify-center mb-2">
                  <TrendingUp className="h-4 w-4 text-green-400 mr-1" />
                  <span className="text-gray-400 text-sm">F1スコア</span>
                </div>
                <div
                  className={`text-lg font-bold ${getScoreColor(
                    metrics?.f1_score
                  )}`}
                >
                  {(() => {
                    const f1Score = metrics?.f1_score;
                    console.log("F1スコア表示:", f1Score);
                    return f1Score ? `${(f1Score * 100).toFixed(2)}%` : "N/A";
                  })()}
                </div>
              </div>

              <div className="text-center p-3 bg-gray-800/50 rounded-lg">
                <div className="flex items-center justify-center mb-2">
                  <BarChart3 className="h-4 w-4 text-purple-400 mr-1" />
                  <span className="text-gray-400 text-sm">AUC</span>
                </div>
                <div
                  className={`text-lg font-bold ${getScoreColor(
                    metrics?.auc_score
                  )}`}
                >
                  {(() => {
                    const aucScore = metrics?.auc_score;
                    console.log("AUCスコア表示:", aucScore);
                    return aucScore ? `${(aucScore * 100).toFixed(2)}%` : "N/A";
                  })()}
                </div>
              </div>
            </div>

            {/* 詳細指標 */}
            {metrics && (
              <div className="space-y-3">
                <h4 className="text-sm font-medium text-gray-300 border-b border-gray-700 pb-2">
                  詳細指標
                </h4>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                  {metrics.precision && (
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">適合率:</span>
                      <Badge variant={getScoreBadgeVariant(metrics.precision)}>
                        {(metrics.precision * 100).toFixed(2)}%
                      </Badge>
                    </div>
                  )}

                  {metrics.recall && (
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">再現率:</span>
                      <Badge variant={getScoreBadgeVariant(metrics.recall)}>
                        {(metrics.recall * 100).toFixed(2)}%
                      </Badge>
                    </div>
                  )}

                  {metrics.loss && (
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">損失:</span>
                      <span className="text-white font-medium">
                        {metrics.loss.toFixed(4)}
                      </span>
                    </div>
                  )}

                  {metrics.val_accuracy && (
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">検証精度:</span>
                      <Badge
                        variant={getScoreBadgeVariant(metrics.val_accuracy)}
                      >
                        {(metrics.val_accuracy * 100).toFixed(2)}%
                      </Badge>
                    </div>
                  )}

                  {metrics.val_loss && (
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">検証損失:</span>
                      <span className="text-white font-medium">
                        {metrics.val_loss.toFixed(4)}
                      </span>
                    </div>
                  )}

                  {metrics.training_time && (
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">学習時間:</span>
                      <div className="flex items-center space-x-1">
                        <Clock className="h-3 w-3 text-gray-400" />
                        <span className="text-white font-medium">
                          {formatTrainingTime(metrics.training_time)}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* 学習データ統計 */}
            {modelInfo && (
              <div className="bg-gray-800/30 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-300 mb-3">
                  学習データ統計
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">サンプル数:</span>
                    <span className="text-white font-medium">
                      {modelInfo.training_samples?.toLocaleString() || "不明"}
                    </span>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">モデル種別:</span>
                    <Badge variant="outline">
                      {modelInfo.model_type || "不明"}
                    </Badge>
                  </div>
                </div>
              </div>
            )}

            {/* 最終評価時刻 */}
            {metrics?.last_evaluation && (
              <div className="text-center text-xs text-gray-500 pt-2 border-t border-gray-700">
                最終評価:{" "}
                {new Date(metrics.last_evaluation).toLocaleString("ja-JP")}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
