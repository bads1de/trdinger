"use client";

import React, { useState, useEffect } from "react";
import { useApiCall } from "@/hooks/useApiCall";
import { StrategyParameter, Strategy } from "@/types/strategy";
import { BacktestConfig } from "@/types/backtest";
import { InputField } from "@/components/common/InputField";
import { SelectField } from "@/components/common/SelectField";
import { BaseBacktestConfigForm } from "./BaseBacktestConfigForm";

interface BacktestFormProps {
  onSubmit: (config: BacktestConfig) => void;
  onConfigChange?: (config: BacktestConfig) => void;
  isLoading?: boolean;
}

export default function BacktestForm({
  onSubmit,
  onConfigChange,
  isLoading = false,
}: BacktestFormProps) {
  const [strategies, setStrategies] = useState<Record<string, Strategy>>({});
  const [selectedStrategy, setSelectedStrategy] = useState<string>("");
  const [config, setConfig] = useState<BacktestConfig>({
    strategy_name: "",
    symbol: "BTC/USDT",
    timeframe: "1h",
    start_date: "2024-01-01",
    end_date: "2024-12-31",
    initial_capital: 100000,
    commission_rate: 0.00055,
    strategy_config: {
      strategy_type: "",
      parameters: {},
    },
  });

  const { execute: fetchStrategies } = useApiCall();

  // 設定更新用のヘルパー関数
  const updateConfig = (updates: Partial<BacktestConfig>) => {
    const newConfig = { ...config, ...updates };
    setConfig(newConfig);
    onConfigChange?.(newConfig);
  };

  useEffect(() => {
    const loadStrategies = async () => {
      try {
        const response = await fetchStrategies("/api/backtest/strategies");
        if (response?.success) {
          setStrategies(response.strategies);
          // デフォルトで最初の戦略を選択
          const firstStrategy = Object.keys(response.strategies)[0];
          if (firstStrategy) {
            setSelectedStrategy(firstStrategy);
            handleStrategyChange(firstStrategy, response.strategies);
          }
        }
      } catch (error) {
        console.error("Failed to load strategies:", error);
      }
    };

    loadStrategies();
  }, []);

  const handleStrategyChange = (
    strategyKey: string,
    strategiesData?: Record<string, Strategy>
  ) => {
    const strategiesSource = strategiesData || strategies;
    const strategy = strategiesSource[strategyKey];

    if (strategy) {
      const newConfig = {
        ...config,
        strategy_name: strategyKey,
        strategy_config: {
          strategy_type: strategyKey,
          parameters: Object.fromEntries(
            Object.entries(strategy.parameters).map(([key, param]) => [
              key,
              param.default,
            ])
          ),
        },
      };
      setConfig(newConfig);
      onConfigChange?.(newConfig);
    }
  };

  const handleParameterChange = (paramName: string, value: number) => {
    const newConfig = {
      ...config,
      strategy_config: {
        ...config.strategy_config,
        parameters: {
          ...config.strategy_config.parameters,
          [paramName]: value,
        },
      },
    };
    setConfig(newConfig);
    onConfigChange?.(newConfig);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const modifiedConfig = { ...config };
    if (modifiedConfig.symbol === "BTC/USDT") {
      modifiedConfig.symbol = "BTC/USDT:USDT";
    }
    onSubmit(modifiedConfig);
  };

  const selectedStrategyData = strategies[selectedStrategy];

  return (
    <div className="w-full">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* 戦略選択 */}
        <div>
          <SelectField
            label="戦略"
            value={selectedStrategy}
            onChange={(value) => {
              setSelectedStrategy(value);
              handleStrategyChange(value);
            }}
            options={Object.entries(strategies).map(([key, strategy]) => ({
              value: key,
              label: strategy.name,
            }))}
            required
          />
          {selectedStrategyData && (
            <p className="mt-2 text-sm text-gray-400">
              {selectedStrategyData.description}
            </p>
          )}
        </div>

        {/* 基本設定 */}
        <BaseBacktestConfigForm config={config} onConfigChange={updateConfig} />

        {/* 戦略パラメータ */}
        {selectedStrategyData &&
          Object.entries(selectedStrategyData.parameters).map(
            ([key, param]) => (
              <InputField
                key={key}
                label={param.description}
                type="number"
                value={config.strategy_config.parameters[key] || param.default}
                onChange={(value) => handleParameterChange(key, value)}
                min={param.min}
                max={param.max}
                step={param.step}
                required
              />
            )
          )}

        <div className="mt-6">
          <button
            type="submit"
            className={`w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 rounded-md text-white font-semibold transition duration-200 ${
              isLoading ? "opacity-50 cursor-not-allowed" : ""
            }`}
            disabled={isLoading}
          >
            {isLoading ? "バックテスト実行中..." : "バックテストを実行"}
          </button>
        </div>
      </form>
    </div>
  );
}
