"use client";

import React, { useMemo } from "react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Cpu } from "lucide-react";

export interface SingleModelSettingsConfig {
  model_type: string;
}

interface SingleModelSettingsProps {
  singleModelSettings: SingleModelSettingsConfig;
  onSingleModelChange?: (settings: SingleModelSettingsConfig) => void;
  availableModels?: string[];
}

/**
 * 単一モデル（シングルモード）設定コンポーネント
 *
 */
export default function SingleModelSettings({
  singleModelSettings,
  onSingleModelChange,
  availableModels = ["lightgbm", "xgboost", "catboost", "tabnet"],
}: SingleModelSettingsProps) {
  const normalizedModels = useMemo(() => {
    const fallback = ["lightgbm", "xgboost", "catboost", "tabnet"];
    const fromProps = (availableModels || []).map((m) =>
      String(m).toLowerCase().trim()
    );
    const union = new Set<string>([...fallback, ...fromProps]);

    return Array.from(union);
  }, [availableModels]);

  return (
    <div className="space-y-3 p-4 bg-muted/50 rounded-lg border">
      <div className="flex items-center space-x-2">
        <Cpu className="h-4 w-4 text-primary" />
        <Label className="text-sm font-medium">単一モデル設定</Label>
      </div>
      <div className="space-y-2">
        <Label className="text-xs text-muted-foreground">使用するモデル</Label>
        <Select
          value={singleModelSettings.model_type}
          onValueChange={(model_type) => onSingleModelChange?.({ model_type })}
        >
          <SelectTrigger>
            <SelectValue placeholder="モデルを選択" />
          </SelectTrigger>
          <SelectContent>
            {normalizedModels.map((model) => (
              <SelectItem key={model} value={model}>
                {model.toUpperCase()}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          選択したモデルで単独トレーニングを実行します
        </p>
      </div>
    </div>
  );
}
