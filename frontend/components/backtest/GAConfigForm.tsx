/**
 * GA設定フォームコンポーネント
 *
 * 遺伝的アルゴリズムによる自動戦略生成の設定を行います。
 */

"use client";

import React, { useState, useEffect } from "react";
import { InputField } from "@/components/common/InputField";
import { SelectField } from "@/components/common/SelectField";
import ActionButton from "@/components/common/ActionButton";
import ApiButton from "@/components/button/ApiButton";
import { GAConfig as GAConfigType } from "@/types/optimization";
import { BacktestConfig as BacktestConfigType } from "@/types/backtest";
import { BaseBacktestConfigForm } from "./BaseBacktestConfigForm";
import { GA_INFO_MESSAGES } from "@/constants/info";
import { ObjectiveSelection } from "./optimization/ObjectiveSelection";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronDown } from "lucide-react";

// 指標モードの選択肢
const INDICATOR_MODE_OPTIONS = [
  { value: "technical_only", label: "TA" },
  { value: "ml_only", label: "ML" },
  { value: "mixed", label: "混合" },
];

interface GAConfigFormProps {
  onSubmit: (config: GAConfigType) => void;
  onClose?: () => void;
  initialConfig?: Partial<GAConfigType>;
  isLoading?: boolean;
  currentBacktestConfig?: BacktestConfigType | null;
}

const GAConfigForm: React.FC<GAConfigFormProps> = ({
  onSubmit,
  onClose,
  initialConfig = {},
  isLoading = false,
  currentBacktestConfig = null,
}) => {
  // Collapsibleの開閉状態（デフォルトは閉じている）
  const [isOpen, setIsOpen] = useState<boolean>(false);

  const [config, setConfig] = useState<GAConfigType>(() => {
    const baseBacktestConfig: BacktestConfigType = {
      strategy_name: "GA_STRATEGY",
      symbol: "BTC/USDT:USDT",
      timeframe: "1h",
      start_date: "2025-01-01",
      end_date: "2025-03-01",
      initial_capital: 1000000,
      commission_rate: 0.00055,
      strategy_config: {
        strategy_type: "",
        parameters: {},
      },
    };

    const effectiveBaseConfig = currentBacktestConfig || baseBacktestConfig;

    return {
      experiment_name: `GA_${new Date()
        .toISOString()
        .slice(0, 10)}_${effectiveBaseConfig.symbol.replace("/", "_")}`,
      base_config: effectiveBaseConfig,
      ga_config: {
        population_size: initialConfig.ga_config?.population_size || 10,
        generations: initialConfig.ga_config?.generations || 10,
        mutation_rate: initialConfig.ga_config?.mutation_rate || 0.1,
        crossover_rate: initialConfig.ga_config?.crossover_rate || 0.8,
        elite_size: initialConfig.ga_config?.elite_size || 5,
        max_indicators: initialConfig.ga_config?.max_indicators || 5,
        allowed_indicators: initialConfig.ga_config?.allowed_indicators || [],
        // 指標モード設定
        indicator_mode:
          initialConfig.ga_config?.indicator_mode || "technical_only",
        fitness_weights: initialConfig.ga_config?.fitness_weights || {
          total_return: 0.3,
          sharpe_ratio: 0.4,
          max_drawdown: 0.2,
          win_rate: 0.1,
        },
        fitness_constraints: initialConfig.ga_config?.fitness_constraints || {
          min_trades: 10,
          max_drawdown_limit: 0.3,
          min_sharpe_ratio: 0.5,
        },
        // 多目的最適化設定
        enable_multi_objective:
          initialConfig.ga_config?.enable_multi_objective ?? true,
        objectives: initialConfig.ga_config?.objectives || [
          "win_rate",
          "max_drawdown",
        ],
        objective_weights: initialConfig.ga_config?.objective_weights || [
          1.0, -1.0,
        ],
        regime_adaptation_enabled:
          initialConfig.ga_config?.regime_adaptation_enabled ?? false,
      },
    };
  });

  useEffect(() => {
    if (currentBacktestConfig) {
      setConfig((prev) => ({
        ...prev,
        base_config: currentBacktestConfig,
      }));
    }
  }, [currentBacktestConfig]);

  const handleBaseConfigChange = (updates: Partial<BacktestConfigType>) => {
    setConfig((prev) => ({
      ...prev,
      base_config: { ...prev.base_config, ...updates },
    }));
  };

  const handleGAConfigChange = (
    updates: Partial<GAConfigType["ga_config"]>
  ) => {
    setConfig((prev) => ({
      ...prev,
      ga_config: { ...prev.ga_config, ...updates },
    }));
  };

  const handleSubmit = () => {
    onSubmit(config);
  };

  return (
    <div className="flex flex-col lg:flex-row min-h-0">
      {/* Left Column: Main Settings */}
      <div className="flex-1 p-6 space-y-6 overflow-y-auto">
        <BaseBacktestConfigForm
          config={config.base_config}
          onConfigChange={handleBaseConfigChange}
          isOptimization={true}
        />
        <InputField
          label="個体数 (population_size)"
          type="number"
          value={config.ga_config.population_size}
          onChange={(value) =>
            setConfig((prev) => ({
              ...prev,
              ga_config: { ...prev.ga_config, population_size: value },
            }))
          }
          min={10}
          step={10}
          required
          description={GA_INFO_MESSAGES.population_size}
        />
        <InputField
          label="世代数 (generations)"
          type="number"
          value={config.ga_config.generations}
          onChange={(value) =>
            setConfig((prev) => ({
              ...prev,
              ga_config: { ...prev.ga_config, generations: value },
            }))
          }
          min={1}
          step={1}
          required
          description={GA_INFO_MESSAGES.generations}
        />
        <InputField
          label="突然変異率 (mutation_rate)"
          type="number"
          value={config.ga_config.mutation_rate}
          onChange={(value) =>
            setConfig((prev) => ({
              ...prev,
              ga_config: { ...prev.ga_config, mutation_rate: value },
            }))
          }
          min={0}
          max={1}
          step={0.01}
          required
          description={GA_INFO_MESSAGES.mutation_rate}
        />
        <InputField
          label="交叉率 (crossover_rate)"
          type="number"
          value={config.ga_config.crossover_rate}
          onChange={(value) =>
            setConfig((prev) => ({
              ...prev,
              ga_config: { ...prev.ga_config, crossover_rate: value },
            }))
          }
          min={0}
          max={1}
          step={0.01}
          required
          description={GA_INFO_MESSAGES.crossover_rate}
        />
        <SelectField
          label="指標モード (indicator_mode)"
          value={config.ga_config.indicator_mode}
          onChange={(value) =>
            setConfig((prev) => ({
              ...prev,
              ga_config: {
                ...prev.ga_config,
                indicator_mode: value as "technical_only" | "ml_only" | "mixed",
              },
            }))
          }
          options={INDICATOR_MODE_OPTIONS}
          required
        />
      </div>

      {/* Right Column: Advanced GA Settings */}
      <div className="flex-1 p-6 space-y-4 bg-secondary-900 border-l border-secondary-700 overflow-y-auto">
        <h3 className="text-lg font-semibold text-secondary-100 mb-3">
          🧬 GA詳細設定
        </h3>

        {/* 自動最適化設定説明（Collapsible） */}
        <Collapsible open={isOpen} onOpenChange={setIsOpen}>
          <CollapsibleTrigger className="w-full">
            <div className="flex items-center justify-between p-3 bg-secondary-800/50 border border-secondary-600/30 rounded-md hover:bg-secondary-700/50 transition-colors">
              <h4 className="font-medium text-secondary-200">
                📋 自動最適化設定説明
              </h4>
              <ChevronDown
                className={`w-5 h-5 text-secondary-400 transition-transform duration-200 ${
                  isOpen ? "rotate-180" : ""
                }`}
              />
            </div>
          </CollapsibleTrigger>

          <CollapsibleContent className="space-y-4 mt-4">
            {/* 指標モードの説明 */}
            <div className="p-3 bg-purple-900/30 border border-purple-500/30 rounded-md">
              <h4 className="font-medium text-purple-300 mb-2">
                📊 指標モード選択
              </h4>
              <div className="text-sm text-purple-200 space-y-1">
                <div>
                  <strong className="text-purple-100">TA:</strong>{" "}
                  従来のテクニカル指標のみを使用
                </div>
                <div>
                  <strong className="text-purple-100">ML:</strong>{" "}
                  ML予測指標のみを使用
                </div>
                <div>
                  <strong className="text-purple-100">混合:</strong>{" "}
                  テクニカル指標とML予測指標を組み合わせ
                </div>
              </div>
            </div>

            {/* リスク管理自動最適化 */}
            <div className="p-3 bg-blue-900/30 border border-blue-500/30 rounded-md">
              <h4 className="font-medium text-blue-300 mb-2">
                🤖 リスク管理自動最適化
              </h4>
              <p className="text-sm text-blue-200">
                TP/SLとポジションサイズはGAが自動最適化します。
                <strong className="text-blue-100">
                  手動でのイグジット条件は無視されます。
                </strong>
              </p>
            </div>

            {/* TP/SL自動最適化 */}
            <div className="p-3 bg-pink-900/30 border border-pink-500/30 rounded-md">
              <h4 className="font-medium text-pink-300 mb-2">
                📈 TP/SL自動最適化
              </h4>
              <div className="text-xs text-pink-200 space-y-1">
                <div>
                  • <strong>決定方式</strong>:
                  固定値、リスクリワード比、ボラティリティベース等
                </div>
                <div>
                  • <strong>リスクリワード比</strong>: 1:1.2 ～ 1:4.0
                </div>
                <div>
                  • <strong>パーセンテージ範囲</strong>: SL: 1%-8%, TP: 2%-20%
                </div>
              </div>
            </div>

            {/* ポジションサイジング自動最適化 */}
            <div className="p-3 bg-emerald-900/30 border border-emerald-500/30 rounded-md">
              <h4 className="font-medium text-emerald-300 mb-2">
                💰 ポジションサイジング自動最適化
              </h4>
              <div className="text-xs text-emerald-200 space-y-1">
                <div>
                  • <strong>方式</strong>: ハーフオプティマルF,
                  ボラティリティベース, 固定比率/枚数
                </div>
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>

        {/* 高度なGA設定 */}
        <div className="p-3 bg-indigo-900/30 border border-indigo-500/30 rounded-md">
          <h4 className="font-medium text-indigo-300 mb-3">⚙️ 高度なGA設定</h4>

          {/* フィットネス共有 */}
          <div className="mb-2">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={config.ga_config.enable_fitness_sharing ?? true}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    ga_config: {
                      ...config.ga_config,
                      enable_fitness_sharing: e.target.checked,
                    },
                  })
                }
                className="rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
              />
              <span className="text-sm text-indigo-200">
                フィットネス共有 (戦略の多様性向上)
              </span>
            </label>
            <div className="pt-1">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.ga_config.regime_adaptation_enabled ?? false}
                  onChange={(e) =>
                    handleGAConfigChange({
                      regime_adaptation_enabled: e.target.checked,
                    })
                  }
                  className="rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
                />
                <span className="text-sm text-indigo-200">
                  レジーム適応を有効化
                </span>
              </label>
            </div>
          </div>

          {/* 多目的最適化設定 */}
          <ObjectiveSelection
            gaConfig={config.ga_config}
            onGAConfigChange={handleGAConfigChange}
          />
        </div>

        {/* Action Buttons */}
        <div className="pt-6 flex justify-end items-center space-x-4 border-t border-secondary-700 mt-6">
          <ActionButton
            type="button"
            onClick={onClose}
            disabled={isLoading}
            variant="secondary"
          >
            キャンセル
          </ActionButton>
          <ApiButton onClick={handleSubmit} loading={isLoading}>
            {config.ga_config.enable_multi_objective
              ? "多目的GA戦略を生成"
              : "GA戦略を生成"}
          </ApiButton>
        </div>
      </div>
    </div>
  );
};

export default GAConfigForm;
