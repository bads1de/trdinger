"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select";
import { 
  Database, 
  Filter, 
  TrendingUp, 
  AlertTriangle,
  CheckCircle 
} from "lucide-react";

interface DataPreprocessingConfig {
  imputation_strategy: "mean" | "median" | "most_frequent";
  scale_features: boolean;
  remove_outliers: boolean;
  outlier_threshold: number;
  handle_infinite_values: boolean;
}

interface DataPreprocessingSettingsProps {
  settings: DataPreprocessingConfig;
  onChange: (settings: DataPreprocessingConfig) => void;
  isLoading?: boolean;
}

export default function DataPreprocessingSettings({
  settings,
  onChange,
  isLoading = false,
}: DataPreprocessingSettingsProps) {
  const updateSetting = <K extends keyof DataPreprocessingConfig>(
    key: K,
    value: DataPreprocessingConfig[K]
  ) => {
    onChange({
      ...settings,
      [key]: value,
    });
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Database className="h-5 w-5" />
          データ前処理設定
          <Badge variant="outline" className="ml-auto">
            高品質補完
          </Badge>
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          統計的手法による高品質なデータ前処理設定
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* 欠損値補完戦略 */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4" />
            <Label className="text-sm font-medium">欠損値補完戦略</Label>
          </div>
          <Select
            value={settings.imputation_strategy}
            onValueChange={(value: "mean" | "median" | "most_frequent") =>
              updateSetting("imputation_strategy", value)
            }
            disabled={isLoading}
          >
            <SelectTrigger>
              <SelectValue placeholder="補完戦略を選択" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="median">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <div>
                    <div className="font-medium">中央値 (推奨)</div>
                    <div className="text-xs text-muted-foreground">
                      外れ値に頑健な補完
                    </div>
                  </div>
                </div>
              </SelectItem>
              <SelectItem value="mean">
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-blue-500" />
                  <div>
                    <div className="font-medium">平均値</div>
                    <div className="text-xs text-muted-foreground">
                      正規分布データに適用
                    </div>
                  </div>
                </div>
              </SelectItem>
              <SelectItem value="most_frequent">
                <div className="flex items-center gap-2">
                  <Database className="h-4 w-4 text-purple-500" />
                  <div>
                    <div className="font-medium">最頻値</div>
                    <div className="text-xs text-muted-foreground">
                      カテゴリカルデータ向け
                    </div>
                  </div>
                </div>
              </SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            fillna(0)の代わりに統計的手法で高品質な補完を実行
          </p>
        </div>

        {/* 外れ値除去 */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              <Label className="text-sm font-medium">外れ値除去</Label>
            </div>
            <Switch
              checked={settings.remove_outliers}
              onCheckedChange={(checked) =>
                updateSetting("remove_outliers", checked)
              }
              disabled={isLoading}
            />
          </div>
          
          {settings.remove_outliers && (
            <div className="space-y-2">
              <Label className="text-sm">
                外れ値閾値: {settings.outlier_threshold}σ
              </Label>
              <Slider
                value={[settings.outlier_threshold]}
                onValueChange={([value]) =>
                  updateSetting("outlier_threshold", value)
                }
                max={5}
                min={1}
                step={0.5}
                className="mt-2"
                disabled={isLoading}
              />
              <p className="text-xs text-muted-foreground">
                標準偏差の何倍を外れ値とするか (Z-score基準)
              </p>
            </div>
          )}
        </div>

        {/* 特徴量スケーリング */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              <Label className="text-sm font-medium">特徴量スケーリング</Label>
            </div>
            <Switch
              checked={settings.scale_features}
              onCheckedChange={(checked) =>
                updateSetting("scale_features", checked)
              }
              disabled={isLoading}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            StandardScalerによる標準化 (平均0, 標準偏差1)
          </p>
        </div>

        {/* 無限値処理 */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4" />
              <Label className="text-sm font-medium">無限値処理</Label>
            </div>
            <Switch
              checked={settings.handle_infinite_values}
              onCheckedChange={(checked) =>
                updateSetting("handle_infinite_values", checked)
              }
              disabled={isLoading}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            ±∞をNaNに変換してから統計的補完を適用
          </p>
        </div>

        {/* 設定サマリー */}
        <div className="mt-6 p-4 bg-muted/50 rounded-lg">
          <h4 className="text-sm font-medium mb-2">現在の設定</h4>
          <div className="space-y-1 text-xs text-muted-foreground">
            <div>• 補完戦略: {
              settings.imputation_strategy === "median" ? "中央値" :
              settings.imputation_strategy === "mean" ? "平均値" : "最頻値"
            }</div>
            <div>• 外れ値除去: {settings.remove_outliers ? `有効 (${settings.outlier_threshold}σ)` : "無効"}</div>
            <div>• スケーリング: {settings.scale_features ? "有効" : "無効"}</div>
            <div>• 無限値処理: {settings.handle_infinite_values ? "有効" : "無効"}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// デフォルト設定
export const defaultDataPreprocessingConfig: DataPreprocessingConfig = {
  imputation_strategy: "median",
  scale_features: false,
  remove_outliers: true,
  outlier_threshold: 3.0,
  handle_infinite_values: true,
};
