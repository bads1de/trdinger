/**
 * 最適化フォームコンポーネント
 *
 * 拡張バックテスト最適化の設定フォームです。
 * SAMBO最適化、マルチ目的最適化、ロバストネステストの設定を提供します。
 */

"use client";

import React, { useState } from "react";
import { useOptimizationForm } from "@/hooks/useOptimizationForm";
import { BacktestConfig, BacktestResult } from "@/types/backtest";
import {
  OptimizationConfig,
  MultiObjectiveConfig,
  RobustnessConfig,
  GAConfig,
} from "@/types/optimization";

import { BaseBacktestConfigForm } from "./BaseBacktestConfigForm";
import TabButton from "../common/TabButton";
import { InputField } from "../common/InputField";

// Dynamic imports for tab components
import EnhancedOptimizationTab from "./optimization/EnhancedOptimizationTab";
import MultiObjectiveTab from "./optimization/MultiObjectiveTab";
import RobustnessTestTab from "./optimization/RobustnessTestTab";
import GATab from "./optimization/GATab";

interface OptimizationFormProps {
  onEnhancedOptimization: (config: OptimizationConfig) => void;
  onMultiObjectiveOptimization: (config: MultiObjectiveConfig) => void;
  onRobustnessTest: (config: RobustnessConfig) => void;
  onGAGeneration?: (config: GAConfig) => void;

  isLoading?: boolean;
  initialConfig?: BacktestResult | null;
  currentBacktestConfig?: BacktestConfig | null;
}

export default function OptimizationForm({
  onEnhancedOptimization,
  onMultiObjectiveOptimization,
  onRobustnessTest,
  onGAGeneration,
  isLoading = false,
  initialConfig = null,
  currentBacktestConfig = null,
}: OptimizationFormProps) {
  const [activeTab, setActiveTab] = useState<
    "enhanced" | "multi" | "robustness" | "ga"
  >("enhanced");

  const { strategies, selectedStrategy, baseConfig, setBaseConfig } =
    useOptimizationForm(initialConfig, currentBacktestConfig);

  const fullBaseConfig: BacktestConfig = {
    ...baseConfig,
    strategy_config: {
      strategy_type: selectedStrategy,
      parameters: currentBacktestConfig?.strategy_config?.parameters || {},
    },
  };

  const {
    start_date: startDate,
    end_date: endDate,
    ...robustnessBaseConfig
  } = fullBaseConfig;

  return (
    <div className="w-full">
      <div className="mb-4 border-b border-secondary-700">
        <nav className="flex -mb-px space-x-4" aria-label="Tabs">
          <TabButton
            label="拡張最適化"
            isActive={activeTab === "enhanced"}
            onClick={() => setActiveTab("enhanced")}
          />
          <TabButton
            label="マルチ目的最適化"
            isActive={activeTab === "multi"}
            onClick={() => setActiveTab("multi")}
          />
          <TabButton
            label="ロバストネステスト"
            isActive={activeTab === "robustness"}
            onClick={() => setActiveTab("robustness")}
          />
          {onGAGeneration && (
            <TabButton
              label="遺伝的アルゴリズム"
              isActive={activeTab === "ga"}
              onClick={() => setActiveTab("ga")}
            />
          )}
        </nav>
      </div>

      <form
        onSubmit={(e) => e.preventDefault()}
        className="space-y-6 p-4 rounded-lg bg-secondary-900"
      >
        <h2 className="text-xl font-semibold text-white">基本設定</h2>
        <BaseBacktestConfigForm
          config={fullBaseConfig}
          onConfigChange={(updates) =>
            setBaseConfig((prev) => ({ ...prev, ...updates }))
          }
          isOptimization={true}
        />

        {selectedStrategy && strategies[selectedStrategy] && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white">戦略パラメータ</h3>
            {Object.entries(strategies[selectedStrategy].parameters).map(
              ([key, param]: [string, any]) => (
                <InputField
                  key={key}
                  label={param.description}
                  type="number"
                  value={
                    currentBacktestConfig?.strategy_config?.parameters[key] ??
                    param.default
                  }
                  onChange={() => {}}
                  min={param.min}
                  max={param.max}
                  step={param.step}
                  required
                  className="cursor-not-allowed opacity-70"
                />
              )
            )}
          </div>
        )}

        {activeTab === "enhanced" && (
          <EnhancedOptimizationTab
            baseConfig={fullBaseConfig}
            selectedStrategy={selectedStrategy}
            onRun={onEnhancedOptimization}
            isLoading={isLoading}
          />
        )}
        {activeTab === "multi" && (
          <MultiObjectiveTab
            baseConfig={fullBaseConfig}
            selectedStrategy={selectedStrategy}
            onRun={onMultiObjectiveOptimization}
            isLoading={isLoading}
          />
        )}
        {activeTab === "robustness" && (
          <RobustnessTestTab
            baseConfig={robustnessBaseConfig}
            selectedStrategy={selectedStrategy}
            onRun={onRobustnessTest}
            isLoading={isLoading}
          />
        )}
        {activeTab === "ga" && onGAGeneration && (
          <GATab
            baseConfig={fullBaseConfig}
            selectedStrategy={selectedStrategy}
            onRun={onGAGeneration}
            isLoading={isLoading}
          />
        )}
      </form>
    </div>
  );
}
