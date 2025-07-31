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
  Brain,
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
type TSFreshSettings = AutoMLFeatureConfig["tsfresh"];
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
  const [activeTab, setActiveTab] = useState("tsfresh");

  const updateTSFreshSettings = (key: keyof TSFreshSettings, value: any) => {
    const newSettings = {
      ...settings,
      tsfresh: {
        ...settings.tsfresh,
        [key]: value,
      },
    };
    onChange(newSettings);

    // 自動検証
    if (onValidate) {
      onValidate(newSettings).then(setValidationResult);
    }
  };



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

    if (settings.tsfresh.enabled) {
      baseTime += settings.tsfresh.feature_count_limit * 0.1;
      if (settings.tsfresh.performance_mode === "comprehensive") {
        baseTime *= 2;
      } else if (settings.tsfresh.performance_mode === "fast") {
        baseTime *= 0.5;
      }
    }

    if (settings.autofeat.enabled) {
      baseTime += settings.autofeat.generations * 0.2;
    }

    return Math.max(baseTime, 1);
  };

  const getComputationalCost = () => {
    let cost = 0;

    if (settings.tsfresh.enabled) {
      cost += settings.tsfresh.feature_count_limit / 10;
      cost += settings.tsfresh.parallel_jobs * 2;
    }

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
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="tsfresh" className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            TSFresh
          </TabsTrigger>
          <TabsTrigger value="autofeat" className="flex items-center gap-2">
            <Zap className="h-4 w-4" />
            AutoFeat
          </TabsTrigger>
        </TabsList>

        {/* TSFresh設定 */}
        <TabsContent value="tsfresh">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                TSFresh時系列特徴量生成
                <Badge
                  variant={settings.tsfresh.enabled ? "default" : "secondary"}
                >
                  {settings.tsfresh.enabled ? "有効" : "無効"}
                </Badge>
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                時系列データから100以上の統計的特徴量を自動生成
              </p>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center space-x-2">
                <Switch
                  id="tsfresh-enabled"
                  checked={settings.tsfresh.enabled}
                  onCheckedChange={(enabled) =>
                    updateTSFreshSettings("enabled", enabled)
                  }
                  disabled={isLoading}
                />
                <Label htmlFor="tsfresh-enabled">
                  TSFresh特徴量生成を有効にする
                </Label>
              </div>

              {settings.tsfresh.enabled && (
                <div className="space-y-4">
                  <div>
                    <Label>パフォーマンスモード</Label>
                    <Select
                      value={settings.tsfresh.performance_mode}
                      onValueChange={(value) =>
                        updateTSFreshSettings("performance_mode", value)
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="fast">
                          高速 (基本特徴量のみ)
                        </SelectItem>
                        <SelectItem value="balanced">
                          バランス (推奨)
                        </SelectItem>
                        <SelectItem value="financial_optimized">
                          金融最適化
                        </SelectItem>
                        <SelectItem value="comprehensive">
                          包括的 (全特徴量)
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label>
                      特徴量数制限: {settings.tsfresh.feature_count_limit}個
                    </Label>
                    <Slider
                      value={[settings.tsfresh.feature_count_limit]}
                      onValueChange={([value]) =>
                        updateTSFreshSettings("feature_count_limit", value)
                      }
                      max={500}
                      min={20}
                      step={10}
                      className="mt-2"
                    />
                  </div>

                  <div>
                    <Label>並列処理数: {settings.tsfresh.parallel_jobs}</Label>
                    <Slider
                      value={[settings.tsfresh.parallel_jobs]}
                      onValueChange={([value]) =>
                        updateTSFreshSettings("parallel_jobs", value)
                      }
                      max={8}
                      min={1}
                      step={1}
                      className="mt-2"
                    />
                  </div>

                  <div className="flex items-center space-x-2">
                    <Switch
                      id="tsfresh-selection"
                      checked={settings.tsfresh.feature_selection}
                      onCheckedChange={(enabled) =>
                        updateTSFreshSettings("feature_selection", enabled)
                      }
                    />
                    <Label htmlFor="tsfresh-selection">
                      統計的特徴量選択を有効にする
                    </Label>
                  </div>

                  {settings.tsfresh.feature_selection && (
                    <div>
                      <Label>FDRレベル: {settings.tsfresh.fdr_level}</Label>
                      <Slider
                        value={[settings.tsfresh.fdr_level]}
                        onValueChange={([value]) =>
                          updateTSFreshSettings("fdr_level", value)
                        }
                        max={0.1}
                        min={0.001}
                        step={0.001}
                        className="mt-2"
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        偽発見率制御レベル (低いほど厳格)
                      </p>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>



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
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="space-y-2">
              <h4 className="font-medium">TSFresh</h4>
              <ul className="text-muted-foreground space-y-1">
                <li>• 統計的特徴量 (平均、分散等)</li>
                <li>• 周波数領域特徴量</li>
                <li>• エントロピー・複雑性</li>
                <li>• 自己相関・トレンド</li>
              </ul>
            </div>
            <div className="space-y-2">
              <h4 className="font-medium">Featuretools</h4>
              <ul className="text-muted-foreground space-y-1">
                <li>• 特徴量の相互作用</li>
                <li>• 集約・変換操作</li>
                <li>• 時間窓ベース特徴量</li>
                <li>• カスタム関数適用</li>
              </ul>
            </div>
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
