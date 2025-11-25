"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { InputField } from "@/components/common/InputField";
import { SelectField } from "@/components/common/SelectField";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Info, Tag } from "lucide-react";
import {
  LABEL_PRESETS,
  SUPPORTED_TIMEFRAMES,
  THRESHOLD_METHODS,
  THRESHOLD_METHOD_LABELS,
  THRESHOLD_METHOD_DESCRIPTIONS,
  TIMEFRAME_LABELS,
  getPresetNames,
  getPresetsByCategory,
  PRESET_CATEGORIES,
} from "@/constants/ml-config-constants";
import type { LabelGenerationConfig } from "@/types/ml-config";

interface LabelGenerationSettingsProps {
  config: LabelGenerationConfig;
  onChange: (key: keyof LabelGenerationConfig, value: any) => void;
}

/**
 * ラベル生成設定UIコンポーネント
 *
 * プリセット選択とカスタム設定の切り替え、各種パラメータの設定を行います。
 */
export const LabelGenerationSettings: React.FC<LabelGenerationSettingsProps> = ({
  config,
  onChange,
}) => {
  const selectedPreset = config.usePreset
    ? LABEL_PRESETS[config.defaultPreset]
    : null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Tag className="h-5 w-5" />
          <span>ラベル生成設定</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* プリセット使用チェックボックス */}
        <div className="flex items-center space-x-2">
          <Checkbox
            id="use-preset"
            checked={config.usePreset}
            onCheckedChange={(checked) => onChange("usePreset", checked)}
          />
          <Label
            htmlFor="use-preset"
            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
          >
            プリセットを使用
          </Label>
        </div>

        {/* プリセット選択（use_preset=trueの場合） */}
        {config.usePreset && (
          <div className="space-y-3">
            <SelectField
              label="プリセット"
              value={config.defaultPreset}
              options={Object.entries(PRESET_CATEGORIES).flatMap(
                ([category, presets]) =>
                  presets.map((presetName) => ({
                    value: presetName,
                    label: `${category} - ${LABEL_PRESETS[presetName]?.description || presetName}`,
                  }))
              )}
              onChange={(value) => onChange("defaultPreset", value)}
            />

            {/* 選択されたプリセットの詳細表示 */}
            {selectedPreset && (
              <Alert className="bg-blue-950 border-blue-800">
                <Info className="h-4 w-4 text-blue-400" />
                <AlertDescription className="text-blue-300">
                  <div className="space-y-1 text-sm">
                    <p>
                      <strong>時間足:</strong>{" "}
                      {TIMEFRAME_LABELS[selectedPreset.timeframe]}
                    </p>
                    <p>
                      <strong>ホライズン:</strong> {selectedPreset.horizonN}本先
                    </p>
                    <p>
                      <strong>閾値:</strong> {selectedPreset.threshold * 100}%
                    </p>
                    {selectedPreset.pt !== undefined && (
                      <p>
                        <strong>PT (利確):</strong> {selectedPreset.pt}σ
                      </p>
                    )}
                    {selectedPreset.sl !== undefined && (
                      <p>
                        <strong>SL (損切):</strong> {selectedPreset.sl}σ
                      </p>
                    )}
                    <p>
                      <strong>閾値計算方法:</strong>{" "}
                      {THRESHOLD_METHOD_LABELS[selectedPreset.thresholdMethod]}
                    </p>
                  </div>
                </AlertDescription>
              </Alert>
            )}
          </div>
        )}

        {/* カスタム設定（use_preset=falseの場合） */}
        {!config.usePreset && (
          <div className="space-y-4 pl-4 border-l-2 border-gray-600">
            <SelectField
              label="時間足"
              value={config.timeframe}
              options={SUPPORTED_TIMEFRAMES.map((tf) => ({
                value: tf,
                label: TIMEFRAME_LABELS[tf],
              }))}
              onChange={(value) => onChange("timeframe", value)}
            />

            <InputField
              label="ホライズン（N本先）"
              type="number"
              value={config.horizonN}
              onChange={(value) => onChange("horizonN", parseInt(value))}
              min={1}
              max={100}
              description="何本先の価格を予測するかを指定します（例: 4本先）"
            />

            <InputField
              label="閾値"
              type="number"
              value={config.threshold}
              onChange={(value) => onChange("threshold", parseFloat(value))}
              step={0.001}
              min={0}
              max={1}
              description="価格変動の閾値を指定します（例: 0.002 = 0.2%）"
            />

            <SelectField
              label="閾値計算方法"
              value={config.thresholdMethod}
              options={THRESHOLD_METHODS.map((method) => ({
                value: method,
                label: THRESHOLD_METHOD_LABELS[method],
              }))}
              onChange={(value) => onChange("thresholdMethod", value)}
            />

            {/* 閾値計算方法の説明 */}
            <Alert className="bg-gray-800 border-gray-700">
              <Info className="h-4 w-4" />
              <AlertDescription className="text-sm text-gray-300">
                {THRESHOLD_METHOD_DESCRIPTIONS[config.thresholdMethod]}
              </AlertDescription>
            </Alert>

            <InputField
              label="価格カラム"
              type="text"
              value={config.priceColumn}
              onChange={(value) => onChange("priceColumn", value)}
              description="使用する価格カラムを指定します（通常は'close'）"
            />
          </div>
        )}

        {/* 説明文 */}
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription className="text-sm">
            ラベル生成設定は、機械学習モデルが予測する目的変数（ラベル）の定義方法を指定します。
            プリセットを使用すると、推奨される設定が自動的に適用されます。
          </AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  );
};