"use client";

import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import LoadingSpinner from "@/components/common/LoadingSpinner";
import ErrorDisplay from "@/components/common/ErrorDisplay";
import {
  Bot,
  Zap,
  TrendingUp,
  Target,
  Database,
  RefreshCw,
  CheckCircle,
  Info,
} from "lucide-react";
import {
  useAutoMLPresets,
  getPresetDisplayName,
  getPresetDescription,
} from "@/hooks/useAutoMLPresets";
import {
  MARKET_CONDITION_LABELS,
  TRADING_STRATEGY_LABELS,
  DATA_SIZE_LABELS,
} from "@/constants/automl-presets-constants";
import { AutoMLPreset, AutoMLFeatureConfig } from "@/hooks/useMLTraining";

interface AutoMLPresetSelectorProps {
  /** 現在のAutoML設定 */
  currentConfig: AutoMLFeatureConfig;
  /** 設定変更時のコールバック */
  onConfigChange: (config: AutoMLFeatureConfig) => void;
  /** ローディング状態 */
  isLoading?: boolean;
  /** カスタムクラス名 */
  className?: string;
}

export default function AutoMLPresetSelector({
  currentConfig,
  onConfigChange,
  isLoading = false,
  className = "",
}: AutoMLPresetSelectorProps) {
  const {
    presets,
    loading: presetsLoading,
    error: presetsError,
    recommendPreset,
  } = useAutoMLPresets();

  const [selectedPreset, setSelectedPreset] = useState<AutoMLPreset | null>(
    null
  );
  const [marketCondition, setMarketCondition] = useState<string>("");
  const [tradingStrategy, setTradingStrategy] = useState<string>("");
  const [dataSize, setDataSize] = useState<string>("");
  const [recommending, setRecommending] = useState(false);

  const handlePresetSelect = (presetName: string) => {
    const preset = presets.find((p) => p.name === presetName);
    if (preset) {
      setSelectedPreset(preset);
      onConfigChange(preset.config);
    }
  };

  const handleRecommendPreset = async () => {
    setRecommending(true);
    try {
      const recommended = await recommendPreset({
        market_condition: marketCondition || undefined,
        trading_strategy: tradingStrategy || undefined,
        data_size: dataSize || undefined,
      });

      if (recommended) {
        setSelectedPreset(recommended);
        onConfigChange(recommended.config);
      }
    } catch (error) {
      console.error("プリセット推奨エラー:", error);
    } finally {
      setRecommending(false);
    }
  };

  if (presetsLoading) {
    return (
      <Card className={`bg-gray-900/50 border-gray-800 ${className}`}>
        <CardContent className="flex items-center justify-center py-8">
          <LoadingSpinner size="lg" />
        </CardContent>
      </Card>
    );
  }

  if (presetsError) {
    return (
      <Card className={`bg-gray-900/50 border-gray-800 ${className}`}>
        <CardContent className="py-8">
          <ErrorDisplay message={presetsError} />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={`bg-gray-900/50 border-gray-800 ${className}`}>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Bot className="h-5 w-5 text-cyan-400" />
          <span>AutoMLプリセット選択</span>
          {selectedPreset && (
            <Badge variant="outline" className="ml-2">
              {selectedPreset.name}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* プリセット直接選択 */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-white flex items-center gap-2">
            <Target className="h-4 w-4" />
            プリセット選択
          </h4>
          <Select onValueChange={handlePresetSelect} disabled={isLoading}>
            <SelectTrigger className="bg-gray-800 border-gray-700">
              <SelectValue placeholder="プリセットを選択..." />
            </SelectTrigger>
            <SelectContent>
              {presets.map((preset) => (
                <SelectItem key={preset.name} value={preset.name}>
                  <div className="flex items-center space-x-2">
                    <span>{getPresetDisplayName(preset)}</span>
                    <Badge variant="outline" className="text-xs">
                      {TRADING_STRATEGY_LABELS[preset.trading_strategy] ||
                        preset.trading_strategy}
                    </Badge>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* 条件ベース推奨 */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-white flex items-center gap-2">
            <Zap className="h-4 w-4" />
            条件ベース推奨
          </h4>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {/* 市場条件 */}
            <div>
              <label className="text-xs text-gray-400 mb-1 block">
                市場条件
              </label>
              <Select onValueChange={setMarketCondition} disabled={isLoading}>
                <SelectTrigger className="bg-gray-800 border-gray-700">
                  <SelectValue placeholder="選択..." />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(MARKET_CONDITION_LABELS).map(
                    ([value, label]) => (
                      <SelectItem key={value} value={value}>
                        {label}
                      </SelectItem>
                    )
                  )}
                </SelectContent>
              </Select>
            </div>

            {/* 取引戦略 */}
            <div>
              <label className="text-xs text-gray-400 mb-1 block">
                取引戦略
              </label>
              <Select onValueChange={setTradingStrategy} disabled={isLoading}>
                <SelectTrigger className="bg-gray-800 border-gray-700">
                  <SelectValue placeholder="選択..." />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(TRADING_STRATEGY_LABELS).map(
                    ([value, label]) => (
                      <SelectItem key={value} value={value}>
                        {label}
                      </SelectItem>
                    )
                  )}
                </SelectContent>
              </Select>
            </div>

            {/* データサイズ */}
            <div>
              <label className="text-xs text-gray-400 mb-1 block">
                データサイズ
              </label>
              <Select onValueChange={setDataSize} disabled={isLoading}>
                <SelectTrigger className="bg-gray-800 border-gray-700">
                  <SelectValue placeholder="選択..." />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(DATA_SIZE_LABELS).map(([value, label]) => (
                    <SelectItem key={value} value={value}>
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <Button
            onClick={handleRecommendPreset}
            disabled={
              isLoading ||
              recommending ||
              (!marketCondition && !tradingStrategy && !dataSize)
            }
            className="w-full"
            variant="outline"
          >
            {recommending ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                推奨中...
              </>
            ) : (
              <>
                <TrendingUp className="h-4 w-4 mr-2" />
                プリセットを推奨
              </>
            )}
          </Button>
        </div>

        {/* 選択されたプリセットの詳細 */}
        {selectedPreset && (
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-white flex items-center gap-2">
              <Info className="h-4 w-4" />
              プリセット詳細
            </h4>

            <div className="bg-gray-800/50 rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h5 className="font-medium text-white">
                  {selectedPreset.name}
                </h5>
                <div className="flex gap-2">
                  <Badge variant="outline">
                    {MARKET_CONDITION_LABELS[selectedPreset.market_condition] ||
                      selectedPreset.market_condition}
                  </Badge>
                  <Badge variant="outline">
                    {TRADING_STRATEGY_LABELS[selectedPreset.trading_strategy] ||
                      selectedPreset.trading_strategy}
                  </Badge>
                  <Badge variant="outline">
                    {DATA_SIZE_LABELS[selectedPreset.data_size] ||
                      selectedPreset.data_size}
                  </Badge>
                </div>
              </div>

              <p className="text-sm text-gray-300">
                {selectedPreset.description}
              </p>

              <div className="space-y-2">
                <h6 className="text-xs font-medium text-gray-400">
                  パフォーマンス特性
                </h6>
                <Textarea
                  value={selectedPreset.performance_notes}
                  readOnly
                  className="bg-gray-700/50 border-gray-600 text-xs resize-none"
                  rows={3}
                />
              </div>

              {/* 設定サマリー */}
              <div className="grid grid-cols-3 gap-4 pt-2 border-t border-gray-700">
                <div className="text-center">
                  <div className="text-xs text-gray-400">TSFresh</div>
                  <div className="flex items-center justify-center gap-1">
                    {selectedPreset.config.tsfresh.enabled ? (
                      <CheckCircle className="h-3 w-3 text-green-400" />
                    ) : (
                      <div className="h-3 w-3 rounded-full bg-gray-600" />
                    )}
                    <span className="text-xs text-white">
                      {selectedPreset.config.tsfresh.enabled ? "有効" : "無効"}
                    </span>
                  </div>
                </div>

                <div className="text-center">
                  <div className="text-xs text-gray-400">Featuretools</div>
                  <div className="flex items-center justify-center gap-1">
                    {selectedPreset.config.featuretools.enabled ? (
                      <CheckCircle className="h-3 w-3 text-green-400" />
                    ) : (
                      <div className="h-3 w-3 rounded-full bg-gray-600" />
                    )}
                    <span className="text-xs text-white">
                      {selectedPreset.config.featuretools.enabled
                        ? "有効"
                        : "無効"}
                    </span>
                  </div>
                </div>

                <div className="text-center">
                  <div className="text-xs text-gray-400">AutoFeat</div>
                  <div className="flex items-center justify-center gap-1">
                    {selectedPreset.config.autofeat.enabled ? (
                      <CheckCircle className="h-3 w-3 text-green-400" />
                    ) : (
                      <div className="h-3 w-3 rounded-full bg-gray-600" />
                    )}
                    <span className="text-xs text-white">
                      {selectedPreset.config.autofeat.enabled ? "有効" : "無効"}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
