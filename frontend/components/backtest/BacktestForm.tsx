"use client";

import React, { useState, useEffect } from "react";
import { useApiCall } from "@/hooks/useApiCall";

interface StrategyParameter {
  type: string;
  default: number;
  min?: number;
  max?: number;
  description: string;
}

interface Strategy {
  name: string;
  description: string;
  parameters: Record<string, StrategyParameter>;
  constraints?: string[];
}

interface BacktestConfig {
  strategy_name: string;
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  commission_rate: number;
  strategy_config: {
    strategy_type: string;
    parameters: Record<string, number>;
  };
}

interface BacktestFormProps {
  onSubmit: (config: BacktestConfig) => void;
  isLoading?: boolean;
}

export default function BacktestForm({
  onSubmit,
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
    commission_rate: 0.001,
    strategy_config: {
      strategy_type: "",
      parameters: {},
    },
  });

  const { execute: fetchStrategies } = useApiCall();

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
      setConfig((prev) => ({
        ...prev,
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
      }));
    }
  };

  const handleParameterChange = (paramName: string, value: number) => {
    setConfig((prev) => ({
      ...prev,
      strategy_config: {
        ...prev.strategy_config,
        parameters: {
          ...prev.strategy_config.parameters,
          [paramName]: value,
        },
      },
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(config);
  };

  const selectedStrategyData = strategies[selectedStrategy];

  return (
    <div className="w-full">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* 戦略選択 */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            戦略
          </label>
          <select
            value={selectedStrategy}
            onChange={(e) => {
              setSelectedStrategy(e.target.value);
              handleStrategyChange(e.target.value);
            }}
            className="w-full p-3 bg-gray-800 border border-gray-700 text-white rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            required
          >
            <option value="">戦略を選択してください</option>
            {Object.entries(strategies).map(([key, strategy]) => (
              <option key={key} value={key}>
                {strategy.name}
              </option>
            ))}
          </select>
          {selectedStrategyData && (
            <p className="mt-2 text-sm text-gray-400">
              {selectedStrategyData.description}
            </p>
          )}
        </div>

        {/* 基本設定 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              取引ペア
            </label>
            <select
              value={config.symbol}
              onChange={(e) =>
                setConfig((prev) => ({ ...prev, symbol: e.target.value }))
              }
              className="w-full p-3 bg-gray-800 border border-gray-700 text-white rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              required
            >
              <option value="BTC/USDT">BTC/USDT</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              時間軸
            </label>
            <select
              value={config.timeframe}
              onChange={(e) =>
                setConfig((prev) => ({ ...prev, timeframe: e.target.value }))
              }
              className="w-full p-3 bg-gray-800 border border-gray-700 text-white rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              required
            >
              <option value="1h">1時間</option>
              <option value="4h">4時間</option>
              <option value="1d">1日</option>
            </select>
          </div>
        </div>

        {/* 期間設定 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              開始日
            </label>
            <input
              type="date"
              value={config.start_date}
              onChange={(e) =>
                setConfig((prev) => ({ ...prev, start_date: e.target.value }))
              }
              className="w-full p-3 bg-gray-800 border border-gray-700 text-white rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              終了日
            </label>
            <input
              type="date"
              value={config.end_date}
              onChange={(e) =>
                setConfig((prev) => ({ ...prev, end_date: e.target.value }))
              }
              className="w-full p-3 bg-gray-800 border border-gray-700 text-white rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              required
            />
          </div>
        </div>

        {/* 資金設定 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              初期資金 (USDT)
            </label>
            <input
              type="number"
              value={config.initial_capital}
              onChange={(e) =>
                setConfig((prev) => ({
                  ...prev,
                  initial_capital: Number(e.target.value),
                }))
              }
              className="w-full p-3 bg-gray-800 border border-gray-700 text-white rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              min="1000"
              step="1000"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              手数料率 (%)
            </label>
            <input
              type="number"
              value={config.commission_rate * 100}
              onChange={(e) =>
                setConfig((prev) => ({
                  ...prev,
                  commission_rate: Number(e.target.value) / 100,
                }))
              }
              className="w-full p-3 bg-gray-800 border border-gray-700 text-white rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              min="0"
              max="1"
              step="0.01"
              required
            />
          </div>
        </div>

        {/* 戦略パラメータ */}
        {selectedStrategyData && (
          <div>
            <h3 className="text-lg font-medium text-white mb-4">
              戦略パラメータ
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(selectedStrategyData.parameters).map(
                ([paramName, param]) => (
                  <div key={paramName}>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      {paramName} - {param.description}
                    </label>
                    <input
                      type="number"
                      value={
                        config.strategy_config.parameters[paramName] ||
                        param.default
                      }
                      onChange={(e) =>
                        handleParameterChange(paramName, Number(e.target.value))
                      }
                      className="w-full p-3 bg-gray-800 border border-gray-700 text-white rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      min={param.min}
                      max={param.max}
                      required
                    />
                    {param.min !== undefined && param.max !== undefined && (
                      <p className="mt-1 text-xs text-gray-400">
                        範囲: {param.min} - {param.max}
                      </p>
                    )}
                  </div>
                )
              )}
            </div>

            {selectedStrategyData.constraints && (
              <div className="mt-4 p-3 bg-yellow-900/20 border border-yellow-600/30 rounded-md">
                <h4 className="text-sm font-medium text-yellow-400 mb-2">
                  制約条件:
                </h4>
                <ul className="text-sm text-yellow-300">
                  {selectedStrategyData.constraints.map((constraint, index) => (
                    <li key={index}>• {constraint}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* 実行ボタン */}
        <div className="pt-4">
          <button
            type="submit"
            disabled={isLoading || !selectedStrategy}
            className={`w-full py-3 px-4 rounded-md font-medium text-white transition-colors ${
              isLoading || !selectedStrategy
                ? "bg-gray-700 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
            }`}
          >
            {isLoading ? "バックテスト実行中..." : "バックテストを実行"}
          </button>
        </div>
      </form>
    </div>
  );
}
