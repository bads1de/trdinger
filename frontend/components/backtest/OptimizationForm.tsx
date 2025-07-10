/**
 * 最適化フォームコンポーネント
 *
 * 拡張バックテスト最適化の設定フォームです。
 * SAMBO最適化、マルチ目的最適化、ロバストネステストの設定を提供します。
 */

"use client";

import React, { useState, useEffect } from "react";
import { useApiCall } from "@/hooks/useApiCall";
import { BacktestConfig, BacktestResult } from "@/types/backtest";
import {
  OptimizationConfig,
  MultiObjectiveConfig,
  RobustnessConfig,
  GAConfig,
} from "@/types/optimization";
import { BaseBacktestConfigForm } from "./BaseBacktestConfigForm";
import { UnifiedStrategy } from "@/types/auto-strategy";
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
  const [strategies, setStrategies] = useState<Record<string, UnifiedStrategy>>(
    {}
  );
  const [selectedStrategy, setSelectedStrategy] = useState<string>("");

  const [baseConfig, setBaseConfig] = useState<Omit<BacktestConfig, 'strategy_config'>>({
    strategy_name: "OPTIMIZED_STRATEGY",
    symbol: "BTC/USDT",
    timeframe: "1d",
    start_date: "2024-01-01",
    end_date: "2024-12-31",
    initial_capital: 10000000,
    commission_rate: 0.00055,
  });

  const { execute: fetchStrategies } = useApiCall();

  useEffect(() => {
    const loadStrategies = async () => {
      try {
        const response = await fetchStrategies("/api/backtest/strategies");
        if (response?.success && Object.keys(response.strategies).length > 0) {
          setStrategies(response.strategies);
          // Set default strategy if not set by currentBacktestConfig
          if (!selectedStrategy) {
            setSelectedStrategy(Object.keys(response.strategies)[0]);
          }
        }
      } catch (error) {
        console.error("Failed to load strategies:", error);
      }
    };

    loadStrategies();
  }, [selectedStrategy]);

  useEffect(() => {
    if (currentBacktestConfig) {
      setBaseConfig({
        strategy_name: `${currentBacktestConfig.strategy_name}_OPTIMIZED`,
        symbol: currentBacktestConfig.symbol,
        timeframe: currentBacktestConfig.timeframe,
        start_date: currentBacktestConfig.start_date,
        end_date: currentBacktestConfig.end_date,
        initial_capital: currentBacktestConfig.initial_capital,
        commission_rate: currentBacktestConfig.commission_rate,
      });
      setSelectedStrategy(currentBacktestConfig.strategy_config.strategy_type);
    } else if (initialConfig) {
      setBaseConfig({
        strategy_name: `${initialConfig.strategy_name}_OPTIMIZED`,
        symbol: initialConfig.symbol,
        timeframe: initialConfig.timeframe,
        start_date: initialConfig.start_date,
        end_date: initialConfig.end_date,
        initial_capital: initialConfig.initial_capital,
        commission_rate: initialConfig.commission_rate,
      });
    }
  }, [currentBacktestConfig, initialConfig]);

  const fullBaseConfig: BacktestConfig = {
    ...baseConfig,
    strategy_config: {
      strategy_type: selectedStrategy,
      parameters: currentBacktestConfig?.strategy_config?.parameters || {},
    },
  };

  const { 'start_date': startDate, 'end_date': endDate, ...robustnessBaseConfig } = fullBaseConfig;


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
