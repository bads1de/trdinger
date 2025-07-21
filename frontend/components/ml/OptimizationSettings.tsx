"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { SelectField } from "@/components/common/SelectField";
import { InputField } from "@/components/common/InputField";
import { Badge } from "@/components/ui/badge";
import {
  Trash2,
  Plus,
  Settings,
  Zap,
  Grid,
  Shuffle,
  Star,
  Clock,
  Target,
} from "lucide-react";
import { useApiCall } from "@/hooks/useApiCall";

export interface ParameterSpaceConfig {
  type: "real" | "integer" | "categorical";
  low?: number;
  high?: number;
  categories?: string[];
}

export interface OptimizationSettingsConfig {
  enabled: boolean;
  method: "bayesian" | "grid" | "random";
  n_calls: number;
  parameter_space: Record<string, ParameterSpaceConfig>;
}

export interface OptimizationPreset {
  name: string;
  description: string;
  method: string;
  n_calls: number;
  parameter_space: Record<string, ParameterSpaceConfig>;
  estimated_time_minutes: number;
}

export type OptimizationLevel = "high" | "medium" | "low";
export type OptimizationMethod = "bayesian" | "grid" | "random";

export interface MethodLevelInfo {
  value: string;
  name: string;
  description: string;
  estimated_time_minutes: number;
  n_calls: number;
}

export interface MethodInfo {
  value: OptimizationMethod;
  name: string;
  description: string;
  levels: MethodLevelInfo[];
}

interface OptimizationSettingsProps {
  settings: OptimizationSettingsConfig;
  onChange: (settings: OptimizationSettingsConfig) => void;
}

const OPTIMIZATION_METHODS = [
  {
    value: "bayesian",
    label: "ベイジアン最適化",
    icon: Zap,
    description: "効率的な最適化",
  },
  {
    value: "grid",
    label: "グリッドサーチ",
    icon: Grid,
    description: "網羅的な探索",
  },
  {
    value: "random",
    label: "ランダムサーチ",
    icon: Shuffle,
    description: "ランダムな探索",
  },
];

const PARAMETER_TYPES = [
  { value: "real", label: "実数" },
  { value: "integer", label: "整数" },
  { value: "categorical", label: "カテゴリ" },
];

const DEFAULT_PARAMETERS = {
  lightgbm: {
    num_leaves: { type: "integer" as const, low: 10, high: 100 },
    learning_rate: { type: "real" as const, low: 0.01, high: 0.3 },
    feature_fraction: { type: "real" as const, low: 0.5, high: 1.0 },
    bagging_fraction: { type: "real" as const, low: 0.5, high: 1.0 },
    min_data_in_leaf: { type: "integer" as const, low: 5, high: 50 },
  },
};

export default function OptimizationSettings({
  settings,
  onChange,
}: OptimizationSettingsProps) {
  const [newParamName, setNewParamName] = useState("");
  const [newParamType, setNewParamType] = useState<
    "real" | "integer" | "categorical"
  >("real");
  const [selectedPresetMethod, setSelectedPresetMethod] =
    useState<OptimizationMethod>("bayesian");
  const [selectedPresetLevel, setSelectedPresetLevel] =
    useState<OptimizationLevel>("medium");
  const [usePreset, setUsePreset] = useState(true);
  const [methodsData, setMethodsData] = useState<MethodInfo[]>([]);

  const { execute: fetchPreset, loading: presetLoading } = useApiCall<{
    success: boolean;
    preset: OptimizationPreset;
  }>();

  const { execute: fetchMethods, loading: methodsLoading } = useApiCall<{
    success: boolean;
    methods: MethodInfo[];
  }>();

  // コンポーネントマウント時にメソッド一覧を取得
  React.useEffect(() => {
    const loadMethods = async () => {
      try {
        const response = await fetchMethods(
          "/api/ml-training/optimization/presets/methods"
        );
        if (response?.success && response.methods) {
          setMethodsData(response.methods);
        }
      } catch (error) {
        console.error("メソッド一覧取得エラー:", error);
      }
    };

    loadMethods();
  }, []);

  const handleEnabledChange = (enabled: boolean) => {
    onChange({ ...settings, enabled });
  };

  const handleMethodChange = (method: string) => {
    onChange({ ...settings, method: method as "bayesian" | "grid" | "random" });
  };

  const handleNCallsChange = (n_calls: string) => {
    const value = parseInt(n_calls);
    if (!isNaN(value) && value > 0) {
      onChange({ ...settings, n_calls: value });
    }
  };

  const addParameter = () => {
    if (!newParamName.trim()) return;

    const newParam: ParameterSpaceConfig = {
      type: newParamType,
      ...(newParamType === "real" && { low: 0.01, high: 1.0 }),
      ...(newParamType === "integer" && { low: 1, high: 100 }),
      ...(newParamType === "categorical" && {
        categories: ["option1", "option2"],
      }),
    };

    onChange({
      ...settings,
      parameter_space: {
        ...settings.parameter_space,
        [newParamName]: newParam,
      },
    });

    setNewParamName("");
  };

  const removeParameter = (paramName: string) => {
    const newParameterSpace = { ...settings.parameter_space };
    delete newParameterSpace[paramName];
    onChange({ ...settings, parameter_space: newParameterSpace });
  };

  const updateParameter = (
    paramName: string,
    field: keyof ParameterSpaceConfig,
    value: any
  ) => {
    const updatedParam = { ...settings.parameter_space[paramName] };

    if (field === "categories" && typeof value === "string") {
      updatedParam.categories = value
        .split(",")
        .map((s) => s.trim())
        .filter((s) => s);
    } else {
      (updatedParam as any)[field] = value;
    }

    onChange({
      ...settings,
      parameter_space: {
        ...settings.parameter_space,
        [paramName]: updatedParam,
      },
    });
  };

  const loadDefaultParameters = () => {
    onChange({
      ...settings,
      parameter_space: { ...DEFAULT_PARAMETERS.lightgbm },
    });
  };

  const handlePresetSelect = async (
    method: OptimizationMethod,
    level: OptimizationLevel
  ) => {
    try {
      const response = await fetchPreset(
        `/api/ml-training/optimization/presets/${method}/${level}`
      );
      if (response?.success && response.preset) {
        const preset = response.preset;
        setSelectedPresetMethod(method);
        setSelectedPresetLevel(level);
        onChange({
          ...settings,
          method: preset.method as "bayesian" | "grid" | "random",
          n_calls: preset.n_calls,
          parameter_space: preset.parameter_space,
        });
      }
    } catch (error) {
      console.error("プリセット取得エラー:", error);
    }
  };

  const selectedMethod = OPTIMIZATION_METHODS.find(
    (m) => m.value === settings.method
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="h-5 w-5" />
          ハイパーパラメータ最適化設定
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* 最適化有効/無効 */}
        <div className="flex items-center space-x-2">
          <Switch
            id="optimization-enabled"
            checked={settings.enabled}
            onCheckedChange={handleEnabledChange}
          />
          <Label htmlFor="optimization-enabled">
            ハイパーパラメータ自動最適化を有効にする
          </Label>
        </div>

        {settings.enabled && (
          <>
            {/* プリセット選択 */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label>最適化プリセット</Label>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="use-preset"
                    checked={usePreset}
                    onCheckedChange={setUsePreset}
                  />
                  <Label htmlFor="use-preset" className="text-sm">
                    プリセットを使用
                  </Label>
                </div>
              </div>

              {usePreset && (
                <div className="space-y-6">
                  {methodsData.map((method) => {
                    const MethodIcon =
                      OPTIMIZATION_METHODS.find((m) => m.value === method.value)
                        ?.icon || Zap;
                    return (
                      <div key={method.value} className="space-y-3">
                        <div className="flex items-center gap-2">
                          <MethodIcon className="h-5 w-5" />
                          <h3 className="font-medium">{method.name}</h3>
                          <span className="text-sm text-muted-foreground">
                            - {method.description}
                          </span>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                          {method.levels.map((level) => {
                            const isSelected =
                              selectedPresetMethod === method.value &&
                              selectedPresetLevel === level.value;

                            const levelConfig = {
                              high: { icon: Star, color: "bg-purple-500" },
                              medium: { icon: Target, color: "bg-blue-500" },
                              low: { icon: Clock, color: "bg-green-500" },
                            }[level.value as OptimizationLevel];

                            const LevelIcon = levelConfig?.icon || Target;

                            return (
                              <Button
                                key={`${method.value}-${level.value}`}
                                variant={isSelected ? "default" : "outline"}
                                className="h-auto p-4 flex flex-col items-center gap-2"
                                onClick={() =>
                                  handlePresetSelect(
                                    method.value as OptimizationMethod,
                                    level.value as OptimizationLevel
                                  )
                                }
                                disabled={presetLoading}
                              >
                                <div
                                  className={`w-3 h-3 rounded-full ${levelConfig?.color} mb-1`}
                                />
                                <LevelIcon className="h-5 w-5" />
                                <div className="text-center">
                                  <div className="font-medium text-sm">
                                    {level.name}
                                  </div>
                                  <div className="text-xs text-muted-foreground mt-1">
                                    {level.n_calls}回試行
                                  </div>
                                  <div className="text-xs text-muted-foreground font-mono">
                                    約{level.estimated_time_minutes}分
                                  </div>
                                </div>
                              </Button>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* 最適化手法選択 */}
            {!usePreset && (
              <div className="space-y-3">
                <Label>最適化手法</Label>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {OPTIMIZATION_METHODS.map((method) => {
                    const Icon = method.icon;
                    return (
                      <Button
                        key={method.value}
                        variant={
                          settings.method === method.value
                            ? "default"
                            : "outline"
                        }
                        className="h-auto p-4 flex flex-col items-center gap-2"
                        onClick={() => handleMethodChange(method.value)}
                      >
                        <Icon className="h-5 w-5" />
                        <div className="text-center">
                          <div className="font-medium">{method.label}</div>
                          <div className="text-xs text-muted-foreground">
                            {method.description}
                          </div>
                        </div>
                      </Button>
                    );
                  })}
                </div>
                {selectedMethod && (
                  <div className="text-sm text-muted-foreground">
                    選択中: {selectedMethod.label} -{" "}
                    {selectedMethod.description}
                  </div>
                )}
              </div>
            )}

            {/* 試行回数 */}
            {!usePreset && (
              <InputField
                label="最適化試行回数"
                type="number"
                value={settings.n_calls.toString()}
                onChange={handleNCallsChange}
                placeholder="50"
                min={1}
                max={1000}
                description="最適化の試行回数を設定します。多いほど良い結果が得られる可能性がありますが、時間がかかります。"
              />
            )}

            {/* パラメータ空間設定 */}
            {!usePreset && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label>パラメータ空間設定</Label>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={loadDefaultParameters}
                    className="text-xs"
                  >
                    デフォルト設定を読み込み
                  </Button>
                </div>

                {/* 既存パラメータ */}
                <div className="space-y-3">
                  {Object.entries(settings.parameter_space).map(
                    ([paramName, paramConfig]) => (
                      <Card key={paramName} className="p-4">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <Badge variant="secondary">{paramName}</Badge>
                            <Badge variant="outline">{paramConfig.type}</Badge>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => removeParameter(paramName)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          {paramConfig.type === "real" ||
                          paramConfig.type === "integer" ? (
                            <>
                              <InputField
                                label="最小値"
                                type="number"
                                value={paramConfig.low?.toString() || ""}
                                onChange={(value) =>
                                  updateParameter(
                                    paramName,
                                    "low",
                                    parseFloat(value)
                                  )
                                }
                                step={paramConfig.type === "real" ? 0.01 : 1}
                              />
                              <InputField
                                label="最大値"
                                type="number"
                                value={paramConfig.high?.toString() || ""}
                                onChange={(value) =>
                                  updateParameter(
                                    paramName,
                                    "high",
                                    parseFloat(value)
                                  )
                                }
                                step={paramConfig.type === "real" ? 0.01 : 1}
                              />
                            </>
                          ) : (
                            <div className="md:col-span-2">
                              <InputField
                                label="カテゴリ (カンマ区切り)"
                                value={paramConfig.categories?.join(", ") || ""}
                                onChange={(value) =>
                                  updateParameter(
                                    paramName,
                                    "categories",
                                    value
                                  )
                                }
                                placeholder="option1, option2, option3"
                              />
                            </div>
                          )}
                        </div>
                      </Card>
                    )
                  )}
                </div>

                {/* 新しいパラメータ追加 */}
                <Card className="p-4 border-dashed">
                  <div className="space-y-3">
                    <Label>新しいパラメータを追加</Label>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      <InputField
                        label="パラメータ名"
                        value={newParamName}
                        onChange={setNewParamName}
                        placeholder="parameter_name"
                      />
                      <SelectField
                        label="型"
                        value={newParamType}
                        onChange={(value) => setNewParamType(value as any)}
                        options={PARAMETER_TYPES}
                      />
                      <div className="flex items-end">
                        <Button
                          onClick={addParameter}
                          disabled={!newParamName.trim()}
                          className="w-full"
                        >
                          <Plus className="h-4 w-4 mr-2" />
                          追加
                        </Button>
                      </div>
                    </div>
                  </div>
                </Card>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
