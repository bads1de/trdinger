"use client";

import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import LoadingSpinner from "@/components/common/LoadingSpinner";
import ErrorDisplay from "@/components/common/ErrorDisplay";
import InfoModal from "@/components/common/InfoModal";
import { useModelPerformance } from "@/hooks/useModelPerformance";
import { formatTrainingTime } from "@/utils/formatters";
import { getScoreColorClass } from "@/utils/colorUtils";
import { ML_METRICS_INFO } from "@/constants/ml-metrics-info";
import {
  TrendingUp,
  Target,
  Activity,
  BarChart3,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Clock,
  Info,
  ChevronDown,
  ChevronRight,
} from "lucide-react";

interface ModelPerformanceCardProps {
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
  const {
    modelStatus,
    loading,
    error,
    loadModelStatus,
    getScoreBadgeVariant,
    getStatusBadgeVariant,
  } = useModelPerformance();

  // InfoModal状態管理
  const [isInfoModalOpen, setIsInfoModalOpen] = useState(false);
  const [modalContent, setModalContent] = useState<{
    title: string;
    content: React.ReactNode;
  } | null>(null);

  // 詳細指標の折り畳み状態管理
  const [isDetailedMetricsExpanded, setIsDetailedMetricsExpanded] =
    useState(false);

  // InfoModalを開く関数
  const openInfoModal = (metricKey: string) => {
    const metricInfo = ML_METRICS_INFO[metricKey];
    if (metricInfo) {
      setModalContent({
        title: metricInfo.name,
        content: (
          <div className="space-y-4">
            <div>
              <h4 className="font-semibold text-secondary-100 mb-2">説明</h4>
              <p className="text-secondary-300">{metricInfo.description}</p>
            </div>
            <div>
              <h4 className="font-semibold text-secondary-100 mb-2">
                値の範囲
              </h4>
              <p className="text-secondary-300">{metricInfo.range}</p>
            </div>
            <div>
              <h4 className="font-semibold text-secondary-100 mb-2">
                解釈方法
              </h4>
              <p className="text-secondary-300">{metricInfo.interpretation}</p>
            </div>
            <div>
              <h4 className="font-semibold text-secondary-100 mb-2">具体例</h4>
              <p className="text-secondary-300">{metricInfo.example}</p>
            </div>
            <div>
              <h4 className="font-semibold text-secondary-100 mb-2">
                重要な場面
              </h4>
              <p className="text-secondary-300">{metricInfo.whenImportant}</p>
            </div>
          </div>
        ),
      });
      setIsInfoModalOpen(true);
    }
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

  // メトリクス表示用のヘルパー関数
  const renderMetricItem = (
    metricKey: string,
    value: number | undefined,
    label: string,
    icon: React.ReactNode
  ) => {
    // undefinedの場合のみ非表示、0.0も有効な値として表示
    if (value === undefined) return null;

    // モデルが読み込まれていない場合の表示
    const isModelNotLoaded = !modelStatus?.is_model_loaded;
    const displayValue =
      isModelNotLoaded && value === 0
        ? "未学習"
        : metricKey.includes("loss") || metricKey.includes("brier")
        ? value.toFixed(4)
        : `${(value * 100).toFixed(2)}%`;

    return (
      <div className="text-center p-3 bg-gray-800/50 rounded-lg">
        <div className="flex items-center justify-center mb-2">
          {icon}
          <span className="text-gray-400 text-sm ml-1">{label}</span>
          <button
            onClick={() => openInfoModal(metricKey)}
            className="ml-2 p-1 hover:bg-gray-700 rounded transition-colors"
            title={`${label}の詳細説明`}
          >
            <Info className="h-3 w-3 text-gray-500 hover:text-blue-400" />
          </button>
        </div>
        <div
          className={`text-lg font-bold ${
            isModelNotLoaded && value === 0
              ? "text-gray-500"
              : getScoreColorClass(value)
          }`}
        >
          {displayValue}
        </div>
      </div>
    );
  };

  // 詳細メトリクス表示用のヘルパー関数
  const renderDetailedMetricItem = (
    metricKey: string,
    value: number | undefined,
    label: string
  ) => {
    // undefinedの場合のみ非表示、0.0も有効な値として表示
    if (value === undefined) return null;

    return (
      <div className="flex justify-between items-center">
        <div className="flex items-center">
          <span className="text-gray-400">{label}:</span>
          <button
            onClick={() => openInfoModal(metricKey)}
            className="ml-1 p-1 hover:bg-gray-700 rounded transition-colors"
            title={`${label}の詳細説明`}
          >
            <Info className="h-3 w-3 text-gray-500 hover:text-blue-400" />
          </button>
        </div>
        <Badge variant={getScoreBadgeVariant(value)}>
          {metricKey.includes("loss") || metricKey.includes("brier")
            ? value.toFixed(4)
            : `${(value * 100).toFixed(2)}%`}
        </Badge>
      </div>
    );
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
  const originalMetrics = modelStatus?.performance_metrics;
  const modelInfo = modelStatus?.model_info;

  // メトリクスデータを処理（0.0も有効な値として扱う）
  const metrics = originalMetrics
    ? {
        ...originalMetrics,
        // undefinedの場合のみ代替値を使用、0.0は有効な値として扱う
        balanced_accuracy:
          originalMetrics.balanced_accuracy !== undefined
            ? originalMetrics.balanced_accuracy
            : undefined,
        matthews_corrcoef:
          originalMetrics.matthews_corrcoef !== undefined
            ? originalMetrics.matthews_corrcoef
            : undefined,
        cohen_kappa:
          originalMetrics.cohen_kappa !== undefined
            ? originalMetrics.cohen_kappa
            : undefined,
        specificity:
          originalMetrics.specificity !== undefined
            ? originalMetrics.specificity
            : undefined,
        sensitivity:
          originalMetrics.sensitivity !== undefined
            ? originalMetrics.sensitivity
            : originalMetrics.recall !== undefined
            ? originalMetrics.recall
            : undefined,
        npv:
          originalMetrics.npv !== undefined ? originalMetrics.npv : undefined,
        ppv:
          originalMetrics.ppv !== undefined
            ? originalMetrics.ppv
            : originalMetrics.precision !== undefined
            ? originalMetrics.precision
            : undefined,
        auc_pr:
          originalMetrics.auc_pr !== undefined
            ? originalMetrics.auc_pr
            : undefined,
        log_loss:
          originalMetrics.log_loss !== undefined
            ? originalMetrics.log_loss
            : undefined,
        brier_score:
          originalMetrics.brier_score !== undefined
            ? originalMetrics.brier_score
            : undefined,
      }
    : originalMetrics;

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
          return null;
        })()}

        {!modelStatus?.is_model_loaded && !modelStatus?.is_training ? (
          <div className="flex flex-col items-center justify-center py-8 text-gray-400">
            <BarChart3 className="h-12 w-12 mb-4 opacity-50" />
            <p className="text-lg font-medium mb-2">
              学習済みモデルが見つかりません
            </p>
            <p className="text-sm text-center mb-4">
              MLモデルの学習を実行して性能指標を表示してください
            </p>
            <div className="flex flex-col space-y-2 text-xs text-gray-500">
              <p>
                •
                F1スコア、バランス精度、MCC、PR-AUCなどの指標が利用可能になります
              </p>
              <p>• 学習には市場データが必要です</p>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {(() => {
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
              {renderMetricItem(
                "accuracy",
                modelInfo?.accuracy || metrics?.accuracy,
                "精度",
                <Target className="h-4 w-4 text-cyan-400 mr-1" />
              )}

              {renderMetricItem(
                "f1_score",
                metrics?.f1_score,
                "F1スコア",
                <TrendingUp className="h-4 w-4 text-green-400 mr-1" />
              )}

              {renderMetricItem(
                "auc_roc",
                metrics?.auc_roc || metrics?.auc_score,
                "AUC-ROC",
                <BarChart3 className="h-4 w-4 text-purple-400 mr-1" />
              )}

              {renderMetricItem(
                "balanced_accuracy",
                metrics?.balanced_accuracy,
                "バランス精度",
                <Target className="h-4 w-4 text-orange-400 mr-1" />
              )}

              {renderMetricItem(
                "matthews_corrcoef",
                metrics?.matthews_corrcoef,
                "MCC",
                <BarChart3 className="h-4 w-4 text-indigo-400 mr-1" />
              )}

              {renderMetricItem(
                "auc_pr",
                metrics?.auc_pr,
                "PR-AUC",
                <TrendingUp className="h-4 w-4 text-pink-400 mr-1" />
              )}
            </div>

            {/* 詳細指標（折り畳み可能） */}
            {metrics && (
              <div className="space-y-4">
                {/* 詳細指標の展開/折り畳みボタン */}
                <div className="border-t border-gray-700 pt-4">
                  <button
                    onClick={() =>
                      setIsDetailedMetricsExpanded(!isDetailedMetricsExpanded)
                    }
                    className="flex items-center justify-between w-full p-3 bg-gray-800/30 hover:bg-gray-800/50 rounded-lg transition-colors"
                  >
                    <div className="flex items-center space-x-2">
                      <BarChart3 className="h-4 w-4 text-cyan-400" />
                      <span className="text-sm font-medium text-gray-300">
                        詳細な評価指標
                      </span>
                      <span className="text-xs text-gray-500">
                        ({isDetailedMetricsExpanded ? "折り畳む" : "展開する"})
                      </span>
                    </div>
                    {isDetailedMetricsExpanded ? (
                      <ChevronDown className="h-4 w-4 text-gray-400" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-gray-400" />
                    )}
                  </button>
                </div>

                {/* 詳細指標の内容（条件付き表示） */}
                {isDetailedMetricsExpanded && (
                  <div className="space-y-4 pl-4">
                    {/* 基本指標 */}
                    <div className="space-y-3">
                      <h4 className="text-sm font-medium text-gray-300 border-b border-gray-700 pb-2">
                        基本指標
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                        {renderDetailedMetricItem(
                          "precision",
                          metrics.precision,
                          "適合率"
                        )}
                        {renderDetailedMetricItem(
                          "recall",
                          metrics.recall,
                          "再現率"
                        )}
                        {renderDetailedMetricItem(
                          "specificity",
                          metrics.specificity,
                          "特異度"
                        )}
                        {renderDetailedMetricItem(
                          "sensitivity",
                          metrics.sensitivity,
                          "感度"
                        )}
                      </div>
                    </div>

                    {/* 高度な指標 */}
                    <div className="space-y-3">
                      <h4 className="text-sm font-medium text-gray-300 border-b border-gray-700 pb-2">
                        高度な指標
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                        {renderDetailedMetricItem(
                          "cohen_kappa",
                          metrics.cohen_kappa,
                          "コーエンのカッパ"
                        )}
                        {renderDetailedMetricItem(
                          "npv",
                          metrics.npv,
                          "陰性的中率"
                        )}
                        {renderDetailedMetricItem(
                          "ppv",
                          metrics.ppv,
                          "陽性的中率"
                        )}
                      </div>
                    </div>

                    {/* 確率指標 */}
                    {(metrics.log_loss !== undefined ||
                      metrics.brier_score !== undefined) && (
                      <div className="space-y-3">
                        <h4 className="text-sm font-medium text-gray-300 border-b border-gray-700 pb-2">
                          確率指標
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                          {renderDetailedMetricItem(
                            "log_loss",
                            metrics.log_loss,
                            "対数損失"
                          )}
                          {renderDetailedMetricItem(
                            "brier_score",
                            metrics.brier_score,
                            "ブライアスコア"
                          )}
                        </div>
                      </div>
                    )}

                    {/* その他の指標 */}
                    {(metrics.loss !== undefined ||
                      metrics.val_accuracy !== undefined ||
                      metrics.val_loss !== undefined ||
                      metrics.training_time !== undefined) && (
                      <div className="space-y-3">
                        <h4 className="text-sm font-medium text-gray-300 border-b border-gray-700 pb-2">
                          その他
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                          {metrics.loss !== undefined && (
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
                                variant={getScoreBadgeVariant(
                                  metrics.val_accuracy
                                )}
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
                  </div>
                )}
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

      {/* InfoModal */}
      {modalContent && (
        <InfoModal
          isOpen={isInfoModalOpen}
          onClose={() => setIsInfoModalOpen(false)}
          title={modalContent.title}
        >
          {modalContent.content}
        </InfoModal>
      )}
    </Card>
  );
}
