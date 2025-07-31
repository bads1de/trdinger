"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import ActionButton from "@/components/common/ActionButton";
import { InputField } from "@/components/common/InputField";
import { Alert, AlertDescription } from "@/components/ui/alert";
import ErrorDisplay from "@/components/common/ErrorDisplay";
import LoadingSpinner from "@/components/common/LoadingSpinner";
import InfoModal from "@/components/common/InfoModal";
import { useMLSettings, AutoMLConfig } from "@/hooks/useMLSettings";
import AutoMLFeatureSettings from "./AutoMLFeatureSettings";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Settings,
  Save,
  RotateCcw,
  Clock,
  Brain,
  Trash2,
  Info,
  Bot,
} from "lucide-react";
import { ML_INFO_MESSAGES } from "@/constants/info";

/**
 * ML設定コンポーネント
 *
 * ML関連の設定を変更・管理するコンポーネント
 */
export default function MLSettings() {
  const {
    config,
    automlConfig,
    isLoading,
    isSaving,
    isResetting,
    isCleaning,
    isAutomlLoading,
    isAutomlSaving,
    error,
    successMessage,
    saveConfig,
    resetToDefaults,
    cleanupOldModels,
    updateConfig,
    fetchAutoMLConfig,
    validateAutoMLConfig: validateAutoMLConfigAPI,
    generateAutoMLFeatures,
    clearAutoMLCache,
    setAutomlConfig,
  } = useMLSettings();

  const [isInfoModalOpen, setIsInfoModalOpen] = useState(false);
  const [modalContent, setModalContent] = useState({ title: "", content: "" });

  const openInfoModal = (title: string, content: string) => {
    setModalContent({ title, content });
    setIsInfoModalOpen(true);
  };

  const handleAutoMLConfigChange = (newConfig: AutoMLConfig) => {
    setAutomlConfig(newConfig);
  };

  const validateAutoMLConfig = async (config: AutoMLConfig) => {
    try {
      return await validateAutoMLConfigAPI(config);
    } catch (error) {
      // フォールバック：簡単な検証ロジック
      const errors: string[] = [];
      const warnings: string[] = [];

      if (config.tsfresh.enabled && config.tsfresh.feature_count_limit > 300) {
        warnings.push(
          "TSFresh特徴量数が多すぎます。処理時間が長くなる可能性があります。"
        );
      }

      return {
        valid: errors.length === 0,
        errors,
        warnings,
      };
    }
  };

  // AutoML設定を初期化
  useEffect(() => {
    if (!automlConfig) {
      fetchAutoMLConfig();
    }
  }, [automlConfig, fetchAutoMLConfig]);

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

      {/* タブ形式の設定 */}
      <Tabs defaultValue="basic" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="basic" className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            基本設定
          </TabsTrigger>
          <TabsTrigger value="automl" className="flex items-center gap-2">
            <Bot className="h-4 w-4" />
            AutoML特徴量
          </TabsTrigger>
        </TabsList>

        {/* 基本設定タブ */}
        <TabsContent value="basic" className="space-y-6">
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
              onClick={() => config && saveConfig(config)}
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
        </TabsContent>

        {/* AutoML特徴量タブ */}
        <TabsContent value="automl">
          {isAutomlLoading ? (
            <LoadingSpinner text="AutoML設定を読み込んでいます..." />
          ) : automlConfig ? (
            <AutoMLFeatureSettings
              settings={automlConfig}
              onChange={handleAutoMLConfigChange}
              onValidate={validateAutoMLConfig}
              isLoading={isAutomlSaving}
            />
          ) : (
            <div className="text-center py-8">
              <p className="text-gray-500 mb-4">
                AutoML設定を読み込めませんでした
              </p>
              <ActionButton
                onClick={fetchAutoMLConfig}
                variant="secondary"
                size="sm"
              >
                再読み込み
              </ActionButton>
            </div>
          )}
        </TabsContent>
      </Tabs>

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
