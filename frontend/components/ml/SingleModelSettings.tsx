"use client";

import React, { useMemo, useState } from "react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Cpu, Info } from "lucide-react";
import InfoModal from "@/components/common/InfoModal";
import { getModelDescription } from "@/constants/modelDescriptions";

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
  availableModels = ["lightgbm", "xgboost"],
}: SingleModelSettingsProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const normalizedModels = useMemo(() => {
    const fallback = ["lightgbm", "xgboost"]; 
    const fromProps = (availableModels || []).map((m) =>
      String(m).toLowerCase().trim()
    );
    const union = new Set<string>([...fallback, ...fromProps]);

    return Array.from(union);
  }, [availableModels]);

  const currentDesc = getModelDescription(singleModelSettings.model_type);

  return (
    <div className="space-y-3 p-4 bg-muted/50 rounded-lg border">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Cpu className="h-4 w-4 text-primary" />
          <Label className="text-sm font-medium">単一モデル設定</Label>
        </div>

        <div>
          <button
            type="button"
            onClick={() => setIsModalOpen(true)}
            className="inline-flex items-center p-1 rounded hover:bg-muted text-muted-foreground"
            aria-label="モデル説明を表示"
            title="モデル説明"
          >
            <Info className="h-4 w-4" />
          </button>
        </div>
      </div>

      <InfoModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title={
          currentDesc?.title || singleModelSettings.model_type || "モデル情報"
        }
      >
        <div className="text-sm text-muted-foreground whitespace-pre-line">
          {currentDesc?.description ||
            "モデルを選択すると、そのモデルの簡単な説明が表示されます。"}
        </div>
      </InfoModal>

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
