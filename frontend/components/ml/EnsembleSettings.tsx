"use client";

import React from "react";
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
import { Layers, Shuffle, BarChart3, Cpu } from "lucide-react";

export interface EnsembleSettingsConfig {
  enabled: boolean;
  method: "bagging" | "stacking";
  bagging_params: {
    n_estimators: number;
    bootstrap_fraction: number;
    base_model_type: string;
    mixed_models?: string[];
    random_state?: number;
  };
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

const AVAILABLE_MODELS = [
  {
    value: "lightgbm",
    label: "LightGBM",
    description: "高速で高精度な勾配ブースティング",
  },
  {
    value: "xgboost",
    label: "XGBoost",
    description: "強力な勾配ブースティング",
  },
  {
    value: "catboost",
    label: "CatBoost",
    description: "カテゴリ特徴量に強い勾配ブースティング",
  },
  {
    value: "tabnet",
    label: "TabNet",
    description: "表形式データ用ニューラルネットワーク",
  },
  {
    value: "random_forest",
    label: "Random Forest",
    description: "アンサンブル決定木",
  },
];

const META_MODELS = [
  { value: "random_forest", label: "Random Forest" },
  { value: "lightgbm", label: "LightGBM" },
];

import SingleModelSettings, {
  SingleModelSettingsConfig as ExtractedSingleModelSettingsConfig,
} from "./SingleModelSettings";

export default function EnsembleSettings({
  settings,
  onChange,
  singleModelSettings = { model_type: "lightgbm" },
  onSingleModelChange,
  availableModels = ["lightgbm", "xgboost", "catboost", "tabnet"],
}: EnsembleSettingsProps) {
  const updateSettings = (updates: Partial<EnsembleSettingsConfig>) => {
    onChange({ ...settings, ...updates });
  };

  const updateBaggingParams = (
    updates: Partial<EnsembleSettingsConfig["bagging_params"]>
  ) => {
    updateSettings({
      bagging_params: { ...settings.bagging_params, ...updates },
    });
  };

  const updateStackingParams = (
    updates: Partial<EnsembleSettingsConfig["stacking_params"]>
  ) => {
    updateSettings({
      stacking_params: { ...settings.stacking_params, ...updates },
    });
  };

  const toggleMixedModel = (modelType: string) => {
    const currentMixed = settings.bagging_params.mixed_models || [];
    const newMixed = currentMixed.includes(modelType)
      ? currentMixed.filter((m) => m !== modelType)
      : [...currentMixed, modelType];

    updateBaggingParams({
      mixed_models: newMixed.length > 0 ? newMixed : undefined,
    });
  };

  const toggleStackingModel = (modelType: string) => {
    const current = settings.stacking_params.base_models;
    const updated = current.includes(modelType)
      ? current.filter((m) => m !== modelType)
      : [...current, modelType];

    updateStackingParams({ base_models: updated });
  };

  const isMixedBagging =
    settings.bagging_params.mixed_models &&
    settings.bagging_params.mixed_models.length > 0;

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
            {/* アンサンブル手法選択 */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">アンサンブル手法</Label>
              <Select
                value={settings.method}
                onValueChange={(method: "bagging" | "stacking") =>
                  updateSettings({ method })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="bagging">
                    <div className="flex items-center space-x-2">
                      <Shuffle className="h-4 w-4" />
                      <div>
                        <div className="font-medium">バギング</div>
                        <div className="text-xs text-muted-foreground">
                          同じモデルを複数学習して平均化
                        </div>
                      </div>
                    </div>
                  </SelectItem>
                  <SelectItem value="stacking">
                    <div className="flex items-center space-x-2">
                      <BarChart3 className="h-4 w-4" />
                      <div>
                        <div className="font-medium">スタッキング</div>
                        <div className="text-xs text-muted-foreground">
                          異なるモデルの予測をメタモデルで統合
                        </div>
                      </div>
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* バギング設定 */}
            {settings.method === "bagging" && (
              <div className="space-y-4 p-4 bg-muted/50 border rounded-lg">
                <h4 className="font-medium flex items-center space-x-2">
                  <Shuffle className="h-4 w-4" />
                  <span>バギング設定</span>
                </h4>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <InputField
                    label="ベースモデル数"
                    type="number"
                    min={2}
                    max={20}
                    value={settings.bagging_params.n_estimators}
                    onChange={(value) =>
                      updateBaggingParams({ n_estimators: Number(value) })
                    }
                  />
                  <InputField
                    label="ブートストラップ比率"
                    type="number"
                    step={0.1}
                    min={0.1}
                    max={1.0}
                    value={settings.bagging_params.bootstrap_fraction}
                    onChange={(value) =>
                      updateBaggingParams({ bootstrap_fraction: Number(value) })
                    }
                  />
                </div>

                {/* 混合バギング設定 */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm font-medium">
                      混合バギング（多様性重視）
                    </Label>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        if (isMixedBagging) {
                          updateBaggingParams({ mixed_models: undefined });
                        } else {
                          updateBaggingParams({
                            mixed_models: ["lightgbm", "xgboost"],
                          });
                        }
                      }}
                    >
                      {isMixedBagging ? "無効化" : "有効化"}
                    </Button>
                  </div>

                  {isMixedBagging ? (
                    <div className="space-y-2">
                      <p className="text-xs text-muted-foreground">
                        使用するモデルを選択してください：
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {AVAILABLE_MODELS.map((model) => (
                          <Badge
                            key={model.value}
                            variant={
                              settings.bagging_params.mixed_models?.includes(
                                model.value
                              )
                                ? "default"
                                : "outline"
                            }
                            className="cursor-pointer"
                            onClick={() => toggleMixedModel(model.value)}
                          >
                            {model.label}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <Label className="text-sm">ベースモデルタイプ</Label>
                      <Select
                        value={settings.bagging_params.base_model_type}
                        onValueChange={(value) =>
                          updateBaggingParams({ base_model_type: value })
                        }
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {AVAILABLE_MODELS.map((model) => (
                            <SelectItem key={model.value} value={model.value}>
                              <div>
                                <div className="font-medium">{model.label}</div>
                                <div className="text-xs text-muted-foreground">
                                  {model.description}
                                </div>
                              </div>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* スタッキング設定 */}
            {settings.method === "stacking" && (
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
                      {AVAILABLE_MODELS.map((model) => (
                        <Badge
                          key={model.value}
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
                      <Label className="text-sm">メタモデル</Label>
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
                          {META_MODELS.map((model) => (
                            <SelectItem key={model.value} value={model.value}>
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
            )}

            {/* 設定サマリー */}
            <div className="p-3 bg-primary/10 rounded-lg">
              <h5 className="font-medium text-primary/90 mb-2">設定サマリー</h5>
              <div className="text-sm text-primary/80 space-y-1">
                <p>
                  手法:{" "}
                  {settings.method === "bagging" ? "バギング" : "スタッキング"}
                </p>
                {settings.method === "bagging" && (
                  <>
                    <p>モデル数: {settings.bagging_params.n_estimators}</p>
                    <p>
                      使用モデル:{" "}
                      {isMixedBagging
                        ? settings.bagging_params.mixed_models?.join(", ")
                        : settings.bagging_params.base_model_type}
                    </p>
                  </>
                )}
                {settings.method === "stacking" && (
                  <>
                    <p>
                      ベースモデル:{" "}
                      {settings.stacking_params.base_models.join(", ")}
                    </p>
                    <p>メタモデル: {settings.stacking_params.meta_model}</p>
                  </>
                )}
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
