"use client";

import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Info, Layers } from "lucide-react";
import type { FeatureEngineeringConfig } from "@/types/ml-config";

interface FeatureAllowlistSettingsProps {
  config: FeatureEngineeringConfig;
  onChange: (key: keyof FeatureEngineeringConfig, value: any) => void;
}

/**
 * 特徴量allowlist設定UIコンポーネント（簡素化版）
 *
 * 研究目的専用のため、プロファイル選択を削除しallowlistのみで管理します。
 */
export const FeatureAllowlistSettings: React.FC<FeatureAllowlistSettingsProps> = ({
  config,
  onChange,
}) => {
  const [allowlistText, setAllowlistText] = useState<string>(
    JSON.stringify(config.featureAllowlist || [], null, 2)
  );
  const [allowlistError, setAllowlistError] = useState<string | null>(null);

  const handleAllowlistChange = (value: string) => {
    setAllowlistText(value);
    setAllowlistError(null);

    // 空文字列の場合はnullに設定（デフォルト35個を使用）
    if (value.trim() === "" || value.trim() === "[]") {
      onChange("featureAllowlist", null);
      return;
    }

    try {
      const parsed = JSON.parse(value);
      if (Array.isArray(parsed)) {
        onChange("featureAllowlist", parsed);
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
          <span>特徴量設定</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* 特徴量allowlist */}
        <div className="space-y-2">
          <Label htmlFor="feature-allowlist" className="text-base font-semibold">
            使用する特徴量リスト
          </Label>
          <p className="text-sm text-gray-400">
            使用する特徴量をJSON配列形式で指定してください。
            空の場合は全ての特徴量が使用されます（研究目的用）。
          </p>
          <Textarea
            id="feature-allowlist"
            placeholder='["RSI_14", "MACD", "BB_Position", ...] または空にしてデフォルトを使用'
            value={allowlistText}
            onChange={(e) => handleAllowlistChange(e.target.value)}
            rows={8}
            className="font-mono text-sm bg-gray-800 border-gray-700 text-white"
          />
          {allowlistError && (
            <p className="text-sm text-red-400">{allowlistError}</p>
          )}
          {!allowlistError && config.featureAllowlist && config.featureAllowlist.length > 0 && (
            <p className="text-sm text-green-400">
              ✓ {config.featureAllowlist.length}個の特徴量が指定されています
            </p>
          )}
          {!allowlistError && !config.featureAllowlist && (
            <p className="text-sm text-blue-400">
              ℹ 全ての特徴量が使用されます（研究用デフォルト）
            </p>
          )}
        </div>

        {/* 情報表示 */}
        <Alert className="bg-blue-950 border-blue-800">
          <Info className="h-4 w-4 text-blue-400" />
          <AlertDescription className="text-blue-300">
            <div className="space-y-2 text-sm">
              <p>
                <strong>研究目的用設定:</strong>
              </p>
              <p>
                デフォルトでは全ての特徴量が使用されます。
                特徴量を制限したい場合は、上記のフィールドでカスタムリストを指定してください。
              </p>
              <p className="text-xs text-blue-200 mt-2">
                参考: テクニカル指標（RSI_14, MACD, BB_*など）、価格関連（Price_Change_Pct, 
                High_Low_Rangeなど）、市場レジーム（Market_Regime, Trend_Strengthなど）、
                建玉残高（OI_*）、ファンディングレート（FR_*, Funding_Rate_Impact）などが自動生成されます。
              </p>
            </div>
          </AlertDescription>
        </Alert>

        {/* 説明文 */}
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription className="text-sm">
            特徴量設定は、機械学習モデルで使用する特徴量の範囲を制御します。
            研究目的のため、デフォルトでは全ての生成可能な特徴量が使用されます。
            過学習が懸念される場合は、カスタムリストで特徴量を絞り込むことができます。
          </AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  );
};
