"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import ActionButton from "@/components/common/ActionButton";
import { InputField } from "@/components/common/InputField";
import { Alert, AlertDescription } from "@/components/ui/alert";
import ErrorDisplay from "@/components/common/ErrorDisplay";
import LoadingSpinner from "@/components/common/LoadingSpinner";
import InfoModal from "@/components/common/InfoModal";
import { useApiCall } from "@/hooks/useApiCall";
import {
  Settings,
  Save,
  RotateCcw,
  Database,
  Clock,
  Brain,
  Trash2,
  Info,
} from "lucide-react";
import { ML_INFO_MESSAGES } from "@/constants/info";

interface MLConfig {
  data_processing: {
    max_ohlcv_rows: number;
    max_feature_rows: number;
    feature_calculation_timeout: number;
    model_training_timeout: number;
  };
  model: {
    model_save_path: string;
    max_model_versions: number;
    model_retention_days: number;
  };
  lightgbm: {
    learning_rate: number;
    num_leaves: number;
    feature_fraction: number;
    bagging_fraction: number;
    num_boost_round: number;
    early_stopping_rounds: number;
  };
  training: {
    train_test_split: number;
    prediction_horizon: number;
    threshold_up: number;
    threshold_down: number;
    min_training_samples: number;
  };
  prediction: {
    default_up_prob: number;
    default_down_prob: number;
    default_range_prob: number;
  };
}

/**
 * ML設定コンポーネント
 *
 * ML関連の設定を変更・管理するコンポーネント
 */
export default function MLSettings() {
  const [config, setConfig] = useState<MLConfig | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [isInfoModalOpen, setIsInfoModalOpen] = useState(false);
  const [modalContent, setModalContent] = useState({ title: "", content: "" });

  const {
    loading: isLoading,
    error: fetchError,
    execute: fetchConfigApi,
  } = useApiCall<MLConfig>();
  const {
    loading: isSaving,
    error: saveError,
    execute: saveConfigApi,
  } = useApiCall();
  const {
    loading: isResetting,
    error: resetError,
    execute: resetConfigApi,
  } = useApiCall();
  const {
    loading: isCleaning,
    error: cleanupError,
    execute: cleanupApi,
  } = useApiCall();

  const error = fetchError || saveError || resetError || cleanupError;

  const openInfoModal = (title: string, content: string) => {
    setModalContent({ title, content });
    setIsInfoModalOpen(true);
  };

  const showSuccessMessage = (message: string) => {
    setSuccessMessage(message);
    setTimeout(() => setSuccessMessage(null), 3000);
  };

  const fetchConfig = useCallback(async () => {
    const result = await fetchConfigApi("/api/ml/config");
    if (result) {
      // APIルートが { success: true, ...config } を返すため、
      // successプロパティを除いた残りをstateにセットする
      const { success, ...configData } = result as any;
      setConfig(configData);
    }
  }, [fetchConfigApi]);

  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  const saveConfig = async () => {
    if (!config) return;
    await saveConfigApi("/api/ml/config", {
      method: "PUT",
      body: config,
      onSuccess: () => showSuccessMessage("設定が正常に保存されました"),
    });
  };

  const resetToDefaults = async () => {
    await resetConfigApi("/api/ml/config/reset", {
      method: "POST",
      confirmMessage: "設定をデフォルト値にリセットしますか？",
      onSuccess: () => {
        fetchConfig();
        showSuccessMessage("設定がデフォルト値にリセットされました");
      },
    });
  };

  const cleanupOldModels = async () => {
    await cleanupApi("/api/ml/models/cleanup", {
      method: "POST",
      confirmMessage:
        "古いモデルファイルを削除しますか？この操作は取り消せません。",
      onSuccess: () => showSuccessMessage("古いモデルファイルが削除されました"),
    });
  };

  const updateConfig = (section: keyof MLConfig, key: string, value: any) => {
    if (!config) return;

    setConfig((prev) => ({
      ...prev!,
      [section]: {
        ...prev![section],
        [key]: value,
      },
    }));
  };

  if (isLoading) {
    return <LoadingSpinner text="設定を読み込んでいます..." />;
  }

  if (error && !config) {
    return <ErrorDisplay message={error} />;
  }

  if (!config) {
    return (
      <div className="text-center p-8 text-gray-500">
        <Settings className="h-12 w-12 mx-auto mb-4 text-gray-300" />
        <p>設定を読み込めませんでした</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 成功・エラーメッセージ */}
      {successMessage && (
        <Alert className="border-green-200 bg-green-50">
          <AlertDescription className="text-green-800">
            {successMessage}
          </AlertDescription>
        </Alert>
      )}
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* データ処理設定 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Database className="h-5 w-5" />
            <span>データ処理設定</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <InputField
                label="最大OHLCV行数"
                type="number"
                value={config.data_processing.max_ohlcv_rows}
                onChange={(value) =>
                  updateConfig("data_processing", "max_ohlcv_rows", value)
                }
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "最大OHLCV行数",
                        ML_INFO_MESSAGES.maxOhlcvRows
                      )
                    }
                  />
                }
              />
            </div>
            <div>
              <InputField
                label="最大特徴量行数"
                type="number"
                value={config.data_processing.max_feature_rows}
                onChange={(value) =>
                  updateConfig("data_processing", "max_feature_rows", value)
                }
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "最大特徴量行数",
                        ML_INFO_MESSAGES.maxFeatureRows
                      )
                    }
                  />
                }
              />
            </div>
            <div>
              <InputField
                label="特徴量計算タイムアウト（秒）"
                type="number"
                value={config.data_processing.feature_calculation_timeout}
                onChange={(value) =>
                  updateConfig(
                    "data_processing",
                    "feature_calculation_timeout",
                    value
                  )
                }
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "特徴量計算タイムアウト（秒）",
                        ML_INFO_MESSAGES.featureCalculationTimeout
                      )
                    }
                  />
                }
              />
            </div>
            <div>
              <InputField
                label="モデル学習タイムアウト（秒）"
                type="number"
                value={config.data_processing.model_training_timeout}
                onChange={(value) =>
                  updateConfig(
                    "data_processing",
                    "model_training_timeout",
                    value
                  )
                }
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "モデル学習タイムアウト（秒）",
                        ML_INFO_MESSAGES.modelTrainingTimeout
                      )
                    }
                  />
                }
              />
            </div>
            <div>
              <InputField
                label="最小学習サンプル数"
                type="number"
                value={config.training.min_training_samples}
                onChange={(value) =>
                  updateConfig("training", "min_training_samples", value)
                }
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "最小学習サンプル数",
                        ML_INFO_MESSAGES.minTrainingSamples
                      )
                    }
                  />
                }
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 予測設定 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5" />
            <span>予測設定</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <InputField
                label="デフォルト上昇確率"
                type="number"
                step={0.01}
                value={config.prediction.default_up_prob}
                onChange={(value) =>
                  updateConfig("prediction", "default_up_prob", value)
                }
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "デフォルト上昇確率",
                        ML_INFO_MESSAGES.defaultUpProb
                      )
                    }
                  />
                }
              />
            </div>
            <div>
              <InputField
                label="デフォルト下落確率"
                type="number"
                step={0.01}
                value={config.prediction.default_down_prob}
                onChange={(value) =>
                  updateConfig("prediction", "default_down_prob", value)
                }
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "デフォルト下落確率",
                        ML_INFO_MESSAGES.defaultDownProb
                      )
                    }
                  />
                }
              />
            </div>
            <div>
              <InputField
                label="デフォルト範囲確率"
                type="number"
                step={0.01}
                value={config.prediction.default_range_prob}
                onChange={(value) =>
                  updateConfig("prediction", "default_range_prob", value)
                }
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "デフォルト範囲確率",
                        ML_INFO_MESSAGES.defaultRangeProb
                      )
                    }
                  />
                }
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* LightGBM設定 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5" />
            <span>LightGBM設定</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <InputField
                label="学習率"
                type="number"
                step={0.01}
                value={config.lightgbm.learning_rate}
                onChange={(value) =>
                  updateConfig("lightgbm", "learning_rate", value)
                }
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "学習率 (learning_rate)",
                        ML_INFO_MESSAGES.learningRate
                      )
                    }
                  />
                }
              />
            </div>
            <div>
              <InputField
                label="葉の数"
                type="number"
                value={config.lightgbm.num_leaves}
                onChange={(value) =>
                  updateConfig("lightgbm", "num_leaves", value)
                }
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "葉の数 (num_leaves)",
                        ML_INFO_MESSAGES.numLeaves
                      )
                    }
                  />
                }
              />
            </div>
            <div>
              <InputField
                label="特徴量サンプリング率"
                type="number"
                step={0.1}
                min={0.1}
                max={1.0}
                value={config.lightgbm.feature_fraction}
                onChange={(value) =>
                  updateConfig("lightgbm", "feature_fraction", value)
                }
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "特徴量サンプリング率 (feature_fraction)",
                        ML_INFO_MESSAGES.featureFraction
                      )
                    }
                  />
                }
              />
            </div>
            <div>
              <InputField
                label="バギング率"
                type="number"
                step={0.1}
                min={0.1}
                max={1.0}
                value={config.lightgbm.bagging_fraction}
                onChange={(value) =>
                  updateConfig("lightgbm", "bagging_fraction", value)
                }
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "バギング率 (bagging_fraction)",
                        ML_INFO_MESSAGES.baggingFraction
                      )
                    }
                  />
                }
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 学習設定 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Clock className="h-5 w-5" />
            <span>学習設定</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <InputField
                label="学習/テスト分割比率"
                type="number"
                step={0.1}
                min={0.1}
                max={0.9}
                value={config.training.train_test_split}
                onChange={(value) =>
                  updateConfig("training", "train_test_split", value)
                }
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "学習/テスト分割比率",
                        ML_INFO_MESSAGES.trainTestSplit
                      )
                    }
                  />
                }
              />
            </div>
            <div>
              <InputField
                label="予測期間（時間）"
                type="number"
                value={config.training.prediction_horizon}
                onChange={(value) =>
                  updateConfig("training", "prediction_horizon", value)
                }
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "予測期間（時間）",
                        ML_INFO_MESSAGES.predictionHorizon
                      )
                    }
                  />
                }
              />
            </div>
            <div>
              <InputField
                label="上昇判定閾値"
                type="number"
                step={0.01}
                value={config.training.threshold_up}
                onChange={(value) =>
                  updateConfig("training", "threshold_up", value)
                }
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "上昇判定閾値",
                        ML_INFO_MESSAGES.thresholdUp
                      )
                    }
                  />
                }
              />
            </div>
            <div>
              <InputField
                label="下落判定閾値"
                type="number"
                step={0.01}
                value={config.training.threshold_down}
                onChange={(value) =>
                  updateConfig("training", "threshold_down", value)
                }
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "下落判定閾値",
                        ML_INFO_MESSAGES.thresholdDown
                      )
                    }
                  />
                }
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* アクションボタン */}
      <div className="flex flex-wrap gap-4">
        <ActionButton
          onClick={saveConfig}
          loading={isSaving}
          icon={<Save className="h-4 w-4" />}
        >
          設定を保存
        </ActionButton>
        <ActionButton
          onClick={resetToDefaults}
          loading={isResetting}
          variant="secondary"
          icon={<RotateCcw className="h-4 w-4" />}
        >
          デフォルトに戻す
        </ActionButton>
        <ActionButton
          onClick={cleanupOldModels}
          loading={isCleaning}
          variant="danger"
          icon={<Trash2 className="h-4 w-4" />}
        >
          古いモデルを削除
        </ActionButton>
      </div>

      <InfoModal
        isOpen={isInfoModalOpen}
        onClose={() => setIsInfoModalOpen(false)}
        title={modalContent.title}
      >
        <p>{modalContent.content}</p>
      </InfoModal>
    </div>
  );
}
