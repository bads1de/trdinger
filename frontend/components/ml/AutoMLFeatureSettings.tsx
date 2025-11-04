"use client";

import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import {
  Bot,
  Zap,
  TrendingUp,
  BarChart3,
  Clock,
  Info,
  AlertTriangle,
  CheckCircle,
} from "lucide-react";

// useMLTrainingフックから型定義をインポート
import { AutoMLFeatureConfig } from "@/hooks/useMLTraining";

// 型エイリアスを作成
type AutoFeatSettings = AutoMLFeatureConfig["autofeat"];

interface AutoMLFeatureSettingsProps {
  settings: AutoMLFeatureConfig;
  onChange: (settings: AutoMLFeatureConfig) => void;
  onValidate?: (settings: AutoMLFeatureConfig) => Promise<any>;
  isLoading?: boolean;
}

export default function AutoMLFeatureSettings({
  settings,
  onChange,
  onValidate,
  isLoading = false,
}: AutoMLFeatureSettingsProps) {
  const [validationResult, setValidationResult] = useState<any>(null);
  const [activeTab, setActiveTab] = useState("autofeat");


  const updateAutoFeatSettings = (key: keyof AutoFeatSettings, value: any) => {
    const newSettings = {
      ...settings,
      autofeat: {
        ...settings.autofeat,
        [key]: value,
      },
    };
    onChange(newSettings);
  };

  const getEstimatedProcessingTime = () => {
    let baseTime = 0;

    if (settings.autofeat.enabled) {
      baseTime += settings.autofeat.generations * 0.2;
    }

    return Math.max(baseTime, 1);
  };

  const getComputationalCost = () => {
    let cost = 0;

    if (settings.autofeat.enabled) {
      cost += settings.autofeat.population_size / 10;
      cost += settings.autofeat.generations / 5;
    }

    return Math.min(cost, 100);
  };

  const estimatedTime = getEstimatedProcessingTime();
  const computationalCost = getComputationalCost();

  return (
    <div className="space-y-6">
      {/* ヘッダー情報 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bot className="h-5 w-5" />
            AutoML特徴量エンジニアリング設定
          </CardTitle>
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-1">
              <Clock className="h-4 w-4" />
              予想処理時間: {estimatedTime.toFixed(1)}分
            </div>
            <div className="flex items-center gap-1">
              <BarChart3 className="h-4 w-4" />
              計算コスト: {computationalCost.toFixed(0)}%
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>計算コスト</span>
                <span>{computationalCost.toFixed(0)}%</span>
              </div>
              <Progress value={computationalCost} className="h-2" />
            </div>

            {validationResult && (
              <Alert
                variant={validationResult.valid ? "default" : "destructive"}
              >
                <AlertDescription>
                  {validationResult.valid ? (
                    <div className="flex items-center gap-2">
                      <CheckCircle className="h-4 w-4" />
                      設定は有効です
                    </div>
                  ) : (
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4" />
                        設定に問題があります
                      </div>
                      {validationResult.errors?.map(
                        (error: string, index: number) => (
                          <div key={index} className="text-xs">
                            • {error}
                          </div>
                        )
                      )}
                    </div>
                  )}
                </AlertDescription>
              </Alert>
            )}
          </div>
        </CardContent>
      </Card>

      {/* 設定タブ */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-1">
          <TabsTrigger value="autofeat" className="flex items-center gap-2">
            <Zap className="h-4 w-4" />
            AutoFeat
          </TabsTrigger>
        </TabsList>

        {/* AutoFeat設定 */}
        <TabsContent value="autofeat">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                AutoFeat遺伝的特徴量選択
                <Badge
                  variant={settings.autofeat.enabled ? "default" : "secondary"}
                >
                  {settings.autofeat.enabled ? "有効" : "無効"}
                </Badge>
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                遺伝的アルゴリズムによる最適特徴量選択
              </p>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center space-x-2">
                <Switch
                  id="autofeat-enabled"
                  checked={settings.autofeat.enabled}
                  onCheckedChange={(enabled) =>
                    updateAutoFeatSettings("enabled", enabled)
                  }
                  disabled={isLoading}
                />
                <Label htmlFor="autofeat-enabled">
                  AutoFeat選択を有効にする
                </Label>
              </div>

              {settings.autofeat.enabled && (
                <div className="space-y-4">
                  <div>
                    <Label>
                      最大特徴量数: {settings.autofeat.max_features}個
                    </Label>
                    <Slider
                      value={[settings.autofeat.max_features]}
                      onValueChange={([value]) =>
                        updateAutoFeatSettings("max_features", value)
                      }
                      max={200}
                      min={10}
                      step={5}
                      className="mt-2"
                    />
                  </div>

                  <div>
                    <Label>世代数: {settings.autofeat.generations}</Label>
                    <Slider
                      value={[settings.autofeat.generations]}
                      onValueChange={([value]) =>
                        updateAutoFeatSettings("generations", value)
                      }
                      max={50}
                      min={5}
                      step={5}
                      className="mt-2"
                    />
                  </div>

                  <div>
                    <Label>
                      集団サイズ: {settings.autofeat.population_size}
                    </Label>
                    <Slider
                      value={[settings.autofeat.population_size]}
                      onValueChange={([value]) =>
                        updateAutoFeatSettings("population_size", value)
                      }
                      max={200}
                      min={20}
                      step={10}
                      className="mt-2"
                    />
                  </div>

                  <div>
                    <Label>
                      トーナメントサイズ: {settings.autofeat.tournament_size}
                    </Label>
                    <Slider
                      value={[settings.autofeat.tournament_size]}
                      onValueChange={([value]) =>
                        updateAutoFeatSettings("tournament_size", value)
                      }
                      max={10}
                      min={2}
                      step={1}
                      className="mt-2"
                    />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* 情報パネル */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Info className="h-5 w-5" />
            AutoML特徴量について
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-4 text-sm">
            <div className="space-y-2">
              <h4 className="font-medium">AutoFeat</h4>
              <ul className="text-muted-foreground space-y-1">
                <li>• 遺伝的アルゴリズム</li>
                <li>• 最適特徴量組み合わせ</li>
                <li>• 自動特徴量選択</li>
                <li>• 性能ベース最適化</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
