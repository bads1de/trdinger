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
  Target,
} from "lucide-react";
import BayesianOptimizationModal from "@/components/bayesian-optimization/BayesianOptimizationModal";
import ProfileSelector from "@/components/bayesian-optimization/ProfileSelector";

interface MLTrainingProps {
  onTabChange?: (tab: string) => void;
}

/**
 * MLトレーニングコンポーネント
 *
 * 新しいMLモデルの学習を開始・管理するコンポーネント
 */
export default function MLTraining({ onTabChange }: MLTrainingProps) {
  const {
    config,
    setConfig,
    trainingStatus,
    error,
    selectedProfile,
    setSelectedProfile,
    startTraining,
    stopTraining,
  } = useMLTraining();

  const [showBayesianModal, setShowBayesianModal] = useState(false);

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

  return (
    <div className="space-y-6">
      {/* エラー表示 */}
      {error && <ErrorDisplay message={error} />}

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
          </div>

          {/* プロファイル選択 */}
          <div className="mt-6">
            <div className="flex items-center space-x-2 mb-4">
              <input
                type="checkbox"
                id="use-profile"
                checked={config.use_profile}
                onChange={(e) => {
                  const useProfile = e.target.checked;
                  setConfig((prev) => ({
                    ...prev,
                    use_profile: useProfile,
                    profile_id: useProfile ? selectedProfile?.id : undefined,
                    profile_name: useProfile
                      ? selectedProfile?.profile_name
                      : undefined,
                  }));
                }}
                disabled={trainingStatus.is_training}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <label htmlFor="use-profile" className="text-sm font-medium">
                最適化プロファイルを使用する
              </label>
            </div>

            {config.use_profile && (
              <ProfileSelector
                selectedProfileId={selectedProfile?.id}
                onProfileSelect={(profile) => {
                  setSelectedProfile(profile);
                  setConfig((prev) => ({
                    ...prev,
                    profile_id: profile?.id,
                    profile_name: profile?.profile_name,
                  }));
                }}
                modelType="LightGBM"
                className="mt-2"
                showManagement={true}
                onManagementClick={() => onTabChange?.("profiles")}
              />
            )}
          </div>

          <div className="flex items-center space-x-4">
            {!trainingStatus.is_training ? (
              <>
                <ActionButton
                  onClick={startTraining}
                  variant="primary"
                  icon={<Play className="h-4 w-4" />}
                >
                  トレーニング開始
                </ActionButton>
                <ActionButton
                  onClick={() => setShowBayesianModal(true)}
                  variant="secondary"
                  icon={<Target className="h-4 w-4" />}
                >
                  ハイパーパラメータ最適化
                </ActionButton>
              </>
            ) : (
              <ActionButton
                onClick={stopTraining}
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
                  {trainingStatus.model_info.training_samples.toLocaleString()}
                </div>
                <div className="text-sm text-gray-600">学習サンプル</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {trainingStatus.model_info.test_samples.toLocaleString()}
                </div>
                <div className="text-sm text-gray-600">テストサンプル</div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ベイジアン最適化モーダル */}
      <BayesianOptimizationModal
        isOpen={showBayesianModal}
        onClose={() => setShowBayesianModal(false)}
      />
    </div>
  );
}
