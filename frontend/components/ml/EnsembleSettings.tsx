"use client";

import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { InputField } from "@/components/common/InputField";
import { Badge } from "@/components/ui/badge";
import { Layers, BarChart3, Info } from "lucide-react";
import InfoModal from "@/components/common/InfoModal";
import { ML_INFO_MESSAGES } from "@/constants/info";
import {
  AVAILABLE_MODEL_NAMES,
  AVAILABLE_MODELS,
  META_MODELS,
} from "@/constants/models";

export interface EnsembleSettingsConfig {
  enabled: boolean;
  method: "stacking";
  stacking_params: {
    base_models: string[];
    meta_model: string;
    cv_folds: number;
    use_probas: boolean;
    random_state?: number;
  };
}

export interface SingleModelSettingsConfig {
  model_type: string;
}

interface EnsembleSettingsProps {
  settings: EnsembleSettingsConfig;
  onChange: (settings: EnsembleSettingsConfig) => void;
  singleModelSettings?: SingleModelSettingsConfig;
  onSingleModelChange?: (settings: SingleModelSettingsConfig) => void;
  availableModels?: string[];
}

import SingleModelSettings, {
  SingleModelSettingsConfig as ExtractedSingleModelSettingsConfig,
} from "./SingleModelSettings";

export default function EnsembleSettings({
  settings,
  onChange,
  singleModelSettings = { model_type: "lightgbm" },
  onSingleModelChange,
  availableModels = AVAILABLE_MODEL_NAMES,
}: EnsembleSettingsProps) {
  // InfoModal state
  const [isInfoModalOpen, setIsInfoModalOpen] = useState(false);
  const [modalContent, setModalContent] = useState<{
    title: string;
    content: string;
  } | null>(null);

  const updateSettings = (updates: Partial<EnsembleSettingsConfig>) => {
    onChange({ ...settings, ...updates });
  };

  // InfoModalを開く関数
  const openInfoModal = (title: string, content: string) => {
    setModalContent({ title, content });
    setIsInfoModalOpen(true);
  };

  const updateStackingParams = (
    updates: Partial<EnsembleSettingsConfig["stacking_params"]>
  ) => {
    updateSettings({
      stacking_params: { ...settings.stacking_params, ...updates },
    });
  };

  const toggleStackingModel = (modelType: string) => {
    const current = settings.stacking_params.base_models;
    const updated = current.includes(modelType)
      ? current.filter((m) => m !== modelType)
      : [...current, modelType];

    updateStackingParams({ base_models: updated });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Layers className="h-5 w-5" />
          <span>アンサンブル学習設定</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* トレーニングモード選択 */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <Label htmlFor="ensemble-enabled" className="text-sm font-medium">
                アンサンブル学習を有効化
              </Label>
              <p className="text-xs text-muted-foreground">
                複数のモデルを組み合わせて予測精度を向上させます
              </p>
            </div>
            <Switch
              id="ensemble-enabled"
              checked={settings.enabled}
              onCheckedChange={(enabled) => updateSettings({ enabled })}
            />
          </div>

          {/* シングルモード設定 */}
          {!settings.enabled && (
            <SingleModelSettings
              singleModelSettings={singleModelSettings}
              onSingleModelChange={onSingleModelChange}
              availableModels={availableModels}
            />
          )}
        </div>

        {settings.enabled && (
          <>
            {/* スタッキング設定 */}
            <div className="space-y-4 p-4 bg-muted/50 border rounded-lg">
              <div className="space-y-4 p-4 bg-muted/50 border rounded-lg">
                <h4 className="font-medium flex items-center space-x-2">
                  <BarChart3 className="h-4 w-4" />
                  <span>スタッキング設定</span>
                </h4>

                <div className="space-y-4">
                  {/* ベースモデル選択 */}
                  <div className="space-y-2">
                    <Label className="text-sm font-medium">
                      ベースモデル（2つ以上選択）
                    </Label>
                    <div className="flex flex-wrap gap-2">
                      {AVAILABLE_MODELS.map((model, index) => (
                        <Badge
                          key={`stacking-model-${model.value}-${index}`}
                          variant={
                            settings.stacking_params.base_models.includes(
                              model.value
                            )
                              ? "default"
                              : "outline"
                          }
                          className="cursor-pointer"
                          onClick={() => toggleStackingModel(model.value)}
                        >
                          {model.label}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* メタモデル選択 */}
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <Label className="text-sm">メタモデル</Label>
                        <Info
                          className="h-4 w-4 text-gray-400 cursor-pointer hover:text-gray-300"
                          onClick={() =>
                            openInfoModal(
                              "メタモデル",
                              ML_INFO_MESSAGES.metaModel
                            )
                          }
                        />
                      </div>
                      <Select
                        value={settings.stacking_params.meta_model}
                        onValueChange={(value) =>
                          updateStackingParams({ meta_model: value })
                        }
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {META_MODELS.map((model, index) => (
                            <SelectItem key={`meta-model-${model.value}-${index}`} value={model.value}>
                              {model.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    {/* CV分割数 */}
                    <InputField
                      label="クロスバリデーション分割数"
                      type="number"
                      min={2}
                      max={10}
                      value={settings.stacking_params.cv_folds}
                      onChange={(value) =>
                        updateStackingParams({ cv_folds: Number(value) })
                      }
                      labelAddon={
                        <Info
                          className="h-4 w-4 text-gray-400 cursor-pointer hover:text-gray-300"
                          onClick={() =>
                            openInfoModal(
                              "クロスバリデーション分割数",
                              ML_INFO_MESSAGES.crossValidationFolds
                            )
                          }
                        />
                      }
                    />
                  </div>

                  {/* 確率使用設定 */}
                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label className="text-sm font-medium">
                        予測確率を使用
                      </Label>
                      <p className="text-xs text-muted-foreground">
                        クラス予測ではなく予測確率をメタ特徴量として使用
                      </p>
                    </div>
                    <Switch
                      checked={settings.stacking_params.use_probas}
                      onCheckedChange={(use_probas) =>
                        updateStackingParams({ use_probas })
                      }
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* 設定サマリー */}
            <div className="p-3 bg-primary/10 rounded-lg">
              <h5 className="font-medium text-primary/90 mb-2">設定サマリー</h5>
              <div className="text-sm text-primary/80 space-y-1">
                <p>手法: スタッキング</p>
                <p>
                  ベースモデル:{" "}
                  {settings.stacking_params.base_models.join(", ")}
                </p>
                <p>メタモデル: {settings.stacking_params.meta_model}</p>
              </div>
            </div>
          </>
        )}
      </CardContent>

      {/* InfoModal */}
      {modalContent && (
        <InfoModal
          isOpen={isInfoModalOpen}
          onClose={() => setIsInfoModalOpen(false)}
          title={modalContent.title}
        >
          <div className="text-secondary-300">
            {modalContent.content}
          </div>
        </InfoModal>
      )}
    </Card>
  );
}
