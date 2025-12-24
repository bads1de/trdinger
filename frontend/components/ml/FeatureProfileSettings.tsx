"use client";

import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Info, Layers } from "lucide-react";
import {
  FEATURE_PROFILES,
  FEATURE_PROFILE_LABELS,
  FEATURE_PROFILE_DESCRIPTIONS,
} from "@/constants/ml-config-constants";
import type { FeatureEngineeringConfig, FeatureProfile } from "@/types/ml-config";

interface FeatureProfileSettingsProps {
  config: FeatureEngineeringConfig;
  onChange: (key: keyof FeatureEngineeringConfig, value: any) => void;
}

/**
 * 特徴量プロファイル設定UIコンポーネント
 *
 * 特徴量プロファイル（research/production）の選択とカスタムallowlistの設定を行います。
 */
export const FeatureProfileSettings: React.FC<FeatureProfileSettingsProps> = ({
  config,
  onChange,
}) => {
  const [allowlistText, setAllowlistText] = useState<string>(
    JSON.stringify(config.customAllowlist || [], null, 2)
  );
  const [allowlistError, setAllowlistError] = useState<string | null>(null);

  const handleAllowlistChange = (value: string) => {
    setAllowlistText(value);
    setAllowlistError(null);

    // 空文字列の場合はnullに設定
    if (value.trim() === "" || value.trim() === "[]") {
      onChange("customAllowlist", null);
      return;
    }

    try {
      const parsed = JSON.parse(value);
      if (Array.isArray(parsed)) {
        onChange("customAllowlist", parsed);
      } else {
        setAllowlistError("配列形式で入力してください");
      }
    } catch (error) {
      setAllowlistError("無効なJSON形式です");
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Layers className="h-5 w-5" />
          <span>特徴量エンジニアリング設定</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* プロファイル選択 */}
        <div className="space-y-3">
          <Label className="text-base font-semibold">特徴量プロファイル</Label>
          <RadioGroup
            value={config.profile}
            onValueChange={(value) => onChange("profile", value)}
          >
            {FEATURE_PROFILES.map((profile) => (
              <div
                key={profile}
                className="flex items-start space-x-3 space-y-0"
              >
                <RadioGroupItem value={profile} id={`profile-${profile}`} />
                <div className="space-y-1 leading-none">
                  <Label
                    htmlFor={`profile-${profile}`}
                    className="font-medium cursor-pointer"
                  >
                    {FEATURE_PROFILE_LABELS[profile]}
                  </Label>
                  <p className="text-sm text-gray-400">
                    {FEATURE_PROFILE_DESCRIPTIONS[profile]}
                  </p>
                </div>
              </div>
            ))}
          </RadioGroup>
        </div>

        {/* カスタムallowlist */}
        <div className="space-y-2">
          <Label htmlFor="custom-allowlist" className="text-base font-semibold">
            カスタム特徴量allowlist（オプション）
          </Label>
          <p className="text-sm text-gray-400">
            特定の特徴量のみを使用したい場合は、JSON配列形式で指定してください。
            空欄の場合はプロファイルのデフォルト設定が使用されます。
          </p>
          <Textarea
            id="custom-allowlist"
            placeholder='["RSI_14", "MACD_Signal", "BB_Position", ...]'
            value={allowlistText}
            onChange={(e) => handleAllowlistChange(e.target.value)}
            rows={6}
            className="font-mono text-sm bg-gray-800 border-gray-700 text-white"
          />
          {allowlistError && (
            <p className="text-sm text-red-400">{allowlistError}</p>
          )}
          {!allowlistError && config.customAllowlist && (
            <p className="text-sm text-green-400">
              ✓ {config.customAllowlist.length}個の特徴量が指定されています
            </p>
          )}
        </div>

        {/* プロファイル情報 */}
        <Alert className="bg-blue-950 border-blue-800">
          <Info className="h-4 w-4 text-blue-400" />
          <AlertDescription className="text-blue-300">
            <div className="space-y-2 text-sm">
              <p>
                <strong>選択中:</strong> {FEATURE_PROFILE_LABELS[(config.profile as FeatureProfile) || "production"]}
              </p>
              {config.profile === "research" && (
                <p>
                  全ての生成可能な特徴量（約108個）を使用します。
                  研究・実験用途に適していますが、計算時間が長くなります。
                </p>
              )}
              {config.profile === "production" && (
                <p>
                  厳選された重要な特徴量（約40個）のみを使用します。
                  本番運用に適しており、計算が高速で過学習のリスクも低減されます。
                </p>
              )}
              {config.customAllowlist && config.customAllowlist.length > 0 && (
                <p className="text-yellow-300">
                  ⚠️ カスタムallowlistが設定されているため、プロファイルの
                  デフォルト設定は上書きされます。
                </p>
              )}
            </div>
          </AlertDescription>
        </Alert>

        {/* 説明文 */}
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription className="text-sm">
            特徴量エンジニアリング設定は、機械学習モデルで使用する特徴量の範囲を制御します。
            researchプロファイルは全特徴量、productionプロファイルは厳選された特徴量を使用します。
          </AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  );
};