"use client";

import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import ActionButton from "@/components/common/ActionButton";
import { InputField } from "@/components/common/InputField";
import { SelectField } from "@/components/common/SelectField";
import ErrorDisplay from "@/components/common/ErrorDisplay";
import { Progress } from "@/components/ui/progress";
import { useMLTraining } from "@/hooks/useMLTraining";
import { getStatusColor } from "@/utils/colorUtils";
import {
  Play,
  Square,
  Brain,
  Settings,
  CheckCircle,
  AlertCircle,
  Clock,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import OptimizationSettings, {
  OptimizationSettingsConfig,
} from "./OptimizationSettings";
import DataPreprocessingSettings, {
  defaultDataPreprocessingConfig,
} from "./DataPreprocessingSettings";
import EnsembleSettings, { EnsembleSettingsConfig } from "./EnsembleSettings";
import { SingleModelSettingsConfig } from "./SingleModelSettings";
import { StopTrainingDialog } from "@/components/common/ConfirmDialog";

/**
 * MLトレーニングコンポーネント
 *
 * 新しいMLモデルの学習を開始・管理するコンポーネント
 */
export default function MLTraining() {
  const {
    config,
    setConfig,
    trainingStatus,
    error,
    startTraining,
    stopTraining,
    getActiveProcesses,
    forceStopProcess,
    availableModels,
    fetchAvailableModels,
  } = useMLTraining();

  const [optimizationSettings, setOptimizationSettings] =
    useState<OptimizationSettingsConfig>({
      enabled: false,
      method: "optuna",
      n_calls: 50,
      parameter_space: {},
    });

  const [preprocessingSettings, setPreprocessingSettings] = useState(
    defaultDataPreprocessingConfig
  );

  const [ensembleSettings, setEnsembleSettings] =
    useState<EnsembleSettingsConfig>({
      enabled: true,
      method: "stacking",
      stacking_params: {
        base_models: ["lightgbm", "xgboost"],
        meta_model: "lightgbm",
        cv_folds: 3,
        use_probas: true,
        random_state: 42,
      },
    });

  const [singleModelSettings, setSingleModelSettings] =
    useState<SingleModelSettingsConfig>({
      model_type: "lightgbm",
    });

  const [showStopDialog, setShowStopDialog] = useState(false);

  const getStatusIcon = () => {
    switch (trainingStatus.status) {
      case "completed":
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      case "error":
        return <AlertCircle className="h-5 w-5 text-red-600" />;
      case "training":
      case "loading_data":
      case "initializing":
        return <Clock className="h-5 w-5 text-blue-600 animate-spin" />;
      default:
        return <Brain className="h-5 w-5 text-gray-600" />;
    }
  };

  // 停止処理のハンドラー
  const handleStopTraining = () => {
    setShowStopDialog(true);
  };

  const handleConfirmStop = () => {
    stopTraining(false);
  };

  const handleForceStop = () => {
    stopTraining(true);
  };

  return (
    <div className="space-y-6">
      {/* エラー表示 */}
      {error && <ErrorDisplay message={error} />}

      {/* 停止確認ダイアログ */}
      <StopTrainingDialog
        open={showStopDialog}
        onOpenChange={setShowStopDialog}
        onConfirm={handleConfirmStop}
        onForceConfirm={handleForceStop}
        isTraining={trainingStatus.is_training}
        processId={trainingStatus.process_id}
      />

      {/* トレーニング設定 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Settings className="h-5 w-5" />
            <span>トレーニング設定</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <SelectField
                label="シンボル"
                value={config.symbol}
                onChange={(value) =>
                  setConfig((prev) => ({ ...prev, symbol: value }))
                }
                options={[{ value: "BTC/USDT:USDT", label: "BTC/USDT:USDT" }]}
                disabled={trainingStatus.is_training}
              />
            </div>

            <div>
              <SelectField
                label="時間足"
                value={config.timeframe}
                onChange={(value) =>
                  setConfig((prev) => ({ ...prev, timeframe: value }))
                }
                options={[
                  { value: "15m", label: "15分" },
                  { value: "30m", label: "30分" },
                  { value: "1h", label: "1時間" },
                  { value: "4h", label: "4時間" },
                  { value: "1d", label: "1日" },
                ]}
                disabled={trainingStatus.is_training}
              />
            </div>

            <div>
              <InputField
                label="開始日"
                type="date"
                value={config.start_date}
                onChange={(value) =>
                  setConfig((prev) => ({ ...prev, start_date: value }))
                }
                disabled={trainingStatus.is_training}
              />
            </div>

            <div>
              <InputField
                label="終了日"
                type="date"
                value={config.end_date}
                onChange={(value) =>
                  setConfig((prev) => ({ ...prev, end_date: value }))
                }
                disabled={trainingStatus.is_training}
              />
            </div>

            <div>
              <InputField
                label="学習/テスト分割比率"
                type="number"
                min={0.1}
                max={0.9}
                step={0.1}
                value={config.train_test_split}
                onChange={(value) =>
                  setConfig((prev) => ({ ...prev, train_test_split: value }))
                }
                disabled={trainingStatus.is_training}
              />
            </div>

            <div>
              <InputField
                label="ランダムシード"
                type="number"
                value={config.random_state}
                onChange={(value) =>
                  setConfig((prev) => ({ ...prev, random_state: value }))
                }
                disabled={trainingStatus.is_training}
              />
            </div>

            <div>
              <SelectField
                label="ラベル生成手法"
                value={config.threshold_method || "TREND_SCANNING"}
                onChange={(value) =>
                  setConfig((prev) => ({ ...prev, threshold_method: value }))
                }
                options={[
                  {
                    value: "TREND_SCANNING",
                    label: "Trend Scanning (推奨 - 高精度)",
                  },
                  {
                    value: "TRIPLE_BARRIER",
                    label: "Triple Barrier (従来手法 - バランス)",
                  },
                ]}
                disabled={trainingStatus.is_training}
              />
            </div>
          </div>

          {/* 最適化設定 */}
          <div className="mt-6">
            <OptimizationSettings
              settings={optimizationSettings}
              onChange={setOptimizationSettings}
            />
          </div>

          {/* アンサンブル設定 */}
          <EnsembleSettings
            settings={ensembleSettings}
            onChange={setEnsembleSettings}
            singleModelSettings={singleModelSettings}
            onSingleModelChange={setSingleModelSettings}
            availableModels={availableModels}
          />

          <div className="flex items-center space-x-4">
            {!trainingStatus.is_training ? (
              <ActionButton
                onClick={() => {
                  // 最適化設定、アンサンブル設定、単一モデル設定を渡す
                  startTraining(
                    optimizationSettings,
                    ensembleSettings,
                    singleModelSettings
                  );
                }}
                variant="primary"
                icon={<Play className="h-4 w-4" />}
              >
                {(() => {
                  const features = [];
                  if (optimizationSettings.enabled) features.push("最適化");
                  if (ensembleSettings.enabled) {
                    features.push("アンサンブル(スタッキング)");
                  } else {
                    features.push(
                      `単一モデル(${singleModelSettings.model_type.toUpperCase()})`
                    );
                  }

                  return features.length > 0
                    ? `${features.join("+")} トレーニング開始`
                    : "トレーニング開始";
                })()}
              </ActionButton>
            ) : (
              <ActionButton
                onClick={handleStopTraining}
                variant="danger"
                icon={<Square className="h-4 w-4" />}
              >
                停止
              </ActionButton>
            )}
          </div>
        </CardContent>
      </Card>

      {/* トレーニング状態 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            {getStatusIcon()}
            <span>トレーニング状態</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <span
              className={`font-medium ${getStatusColor(trainingStatus.status)}`}
            >
              {trainingStatus.message}
            </span>
            <span className="text-sm text-gray-500">
              {trainingStatus.progress}%
            </span>
          </div>

          <Progress value={trainingStatus.progress} className="w-full" />

          {trainingStatus.start_time && (
            <div className="text-sm text-gray-600">
              開始時刻:{" "}
              {new Date(trainingStatus.start_time).toLocaleString("ja-JP")}
            </div>
          )}

          {trainingStatus.end_time && (
            <div className="text-sm text-gray-600">
              終了時刻:{" "}
              {new Date(trainingStatus.end_time).toLocaleString("ja-JP")}
            </div>
          )}

          {trainingStatus.process_id && (
            <div className="text-sm text-gray-600">
              プロセスID: {trainingStatus.process_id}
            </div>
          )}

          {trainingStatus.model_info && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {(trainingStatus.model_info.accuracy * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600">精度</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {trainingStatus.model_info.feature_count}
                </div>
                <div className="text-sm text-gray-600">特徴量数</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {trainingStatus.model_info.training_samples?.toLocaleString() ||
                    "0"}
                </div>
                <div className="text-sm text-gray-600">学習サンプル</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {trainingStatus.model_info.test_samples?.toLocaleString() ||
                    "0"}
                </div>
                <div className="text-sm text-gray-600">テストサンプル</div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
