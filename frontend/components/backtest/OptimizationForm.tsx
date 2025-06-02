/**
 * 最適化フォームコンポーネント
 *
 * 拡張バックテスト最適化の設定フォームです。
 * SAMBO最適化、マルチ目的最適化、ロバストネステストの設定を提供します。
 */

"use client";

import React, { useState, useEffect } from "react";
import ApiButton from "@/components/button/ApiButton";
import { useApiCall } from "@/hooks/useApiCall";

interface OptimizationConfig {
  base_config: {
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
  };
  optimization_params: {
    method: "grid" | "sambo";
    max_tries?: number;
    maximize: string;
    return_heatmap: boolean;
    return_optimization?: boolean;
    random_state?: number;
    constraint?: string;
    parameters: Record<string, number[]>;
  };
}

interface MultiObjectiveConfig {
  base_config: {
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
  };
  objectives: string[];
  weights?: number[];
  optimization_params?: {
    method: "grid" | "sambo";
    max_tries?: number;
    parameters: Record<string, number[]>;
  };
}

interface RobustnessConfig {
  base_config: {
    strategy_name: string;
    symbol: string;
    timeframe: string;
    initial_capital: number;
    commission_rate: number;
    strategy_config: {
      strategy_type: string;
      parameters: Record<string, number>;
    };
  };
  test_periods: string[][];
  optimization_params: {
    method: "grid" | "sambo";
    max_tries?: number;
    maximize: string;
    parameters: Record<string, number[]>;
  };
}

interface BacktestResult {
  id: string;
  strategy_name: string;
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  commission_rate: number;
}

interface OptimizationFormProps {
  onEnhancedOptimization: (config: OptimizationConfig) => void;
  onMultiObjectiveOptimization: (config: MultiObjectiveConfig) => void;
  onRobustnessTest: (config: RobustnessConfig) => void;
  isLoading?: boolean;
  initialConfig?: BacktestResult | null;
}

export default function OptimizationForm({
  onEnhancedOptimization,
  onMultiObjectiveOptimization,
  onRobustnessTest,
  isLoading = false,
  initialConfig = null,
}: OptimizationFormProps) {
  const [activeTab, setActiveTab] = useState<
    "enhanced" | "multi" | "robustness"
  >("enhanced");
  const [strategies, setStrategies] = useState<Record<string, any>>({});
  const [selectedStrategy, setSelectedStrategy] = useState<string>("SMA_CROSS");

  // 基本設定
  const [baseConfig, setBaseConfig] = useState({
    strategy_name: "OPTIMIZED_STRATEGY",
    symbol: "BTC/USDT",
    timeframe: "1d",
    start_date: "2024-01-01",
    end_date: "2024-12-31",
    initial_capital: 10000000,
    commission_rate: 0.001,
  });

  // 拡張最適化設定
  const [enhancedConfig, setEnhancedConfig] = useState({
    method: "grid" as "grid" | "sambo",
    max_tries: 100,
    maximize: "Sharpe Ratio",
    return_heatmap: true,
    return_optimization: false,
    random_state: 42,
    constraint: "sma_cross",
    n1_range: [10, 30, 5],
    n2_range: [30, 80, 10],
  });

  // マルチ目的最適化設定
  const [multiConfig, setMultiConfig] = useState({
    objectives: ["Sharpe Ratio", "Return [%]", "-Max. Drawdown [%]"],
    weights: [0.4, 0.4, 0.2],
    method: "grid" as "grid" | "sambo",
    max_tries: 80,
    n1_range: [10, 25, 5],
    n2_range: [30, 70, 10],
  });

  // ロバストネステスト設定
  const [robustnessConfig, setRobustnessConfig] = useState({
    test_periods: [
      ["2024-01-01", "2024-03-31"],
      ["2024-04-01", "2024-06-30"],
      ["2024-07-01", "2024-09-30"],
      ["2024-10-01", "2024-12-31"],
    ],
    method: "grid" as "grid" | "sambo",
    max_tries: 50,
    maximize: "Sharpe Ratio",
    n1_range: [10, 25, 5],
    n2_range: [30, 60, 10],
  });

  const { execute: fetchStrategies } = useApiCall();

  useEffect(() => {
    const loadStrategies = async () => {
      try {
        const response = await fetchStrategies("/api/backtest/strategies");
        if (response?.success) {
          setStrategies(response.strategies);
        }
      } catch (error) {
        console.error("Failed to load strategies:", error);
      }
    };

    loadStrategies();
  }, []);

  // 初期設定が渡された場合、基本設定を自動入力
  useEffect(() => {
    if (initialConfig) {
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
  }, [initialConfig]);

  const createParameterRange = (rangeConfig: number[]) => {
    const [start, end, step] = rangeConfig;
    const range = [];
    for (let i = start; i <= end; i += step) {
      range.push(i);
    }
    return range;
  };

  const handleEnhancedSubmit = () => {
    const config: OptimizationConfig = {
      base_config: {
        ...baseConfig,
        strategy_config: {
          strategy_type: selectedStrategy,
          parameters: {},
        },
      },
      optimization_params: {
        method: enhancedConfig.method,
        ...(enhancedConfig.method === "sambo" && {
          max_tries: enhancedConfig.max_tries,
        }),
        maximize: enhancedConfig.maximize,
        return_heatmap: enhancedConfig.return_heatmap,
        ...(enhancedConfig.method === "sambo" && {
          return_optimization: enhancedConfig.return_optimization,
          random_state: enhancedConfig.random_state,
        }),
        constraint: enhancedConfig.constraint,
        parameters: {
          n1: createParameterRange(enhancedConfig.n1_range),
          n2: createParameterRange(enhancedConfig.n2_range),
        },
      },
    };
    onEnhancedOptimization(config);
  };

  const handleMultiObjectiveSubmit = () => {
    const config: MultiObjectiveConfig = {
      base_config: {
        ...baseConfig,
        strategy_config: {
          strategy_type: selectedStrategy,
          parameters: {},
        },
      },
      objectives: multiConfig.objectives,
      weights: multiConfig.weights,
      optimization_params: {
        method: multiConfig.method,
        ...(multiConfig.method === "sambo" && {
          max_tries: multiConfig.max_tries,
        }),
        parameters: {
          n1: createParameterRange(multiConfig.n1_range),
          n2: createParameterRange(multiConfig.n2_range),
        },
      },
    };
    onMultiObjectiveOptimization(config);
  };

  const handleRobustnessSubmit = () => {
    const config: RobustnessConfig = {
      base_config: {
        strategy_name: baseConfig.strategy_name,
        symbol: baseConfig.symbol,
        timeframe: baseConfig.timeframe,
        initial_capital: baseConfig.initial_capital,
        commission_rate: baseConfig.commission_rate,
        strategy_config: {
          strategy_type: selectedStrategy,
          parameters: {},
        },
      },
      test_periods: robustnessConfig.test_periods,
      optimization_params: {
        method: robustnessConfig.method,
        ...(robustnessConfig.method === "sambo" && {
          max_tries: robustnessConfig.max_tries,
        }),
        maximize: robustnessConfig.maximize,
        parameters: {
          n1: createParameterRange(robustnessConfig.n1_range),
          n2: createParameterRange(robustnessConfig.n2_range),
        },
      },
    };
    onRobustnessTest(config);
  };

  const TabButton = ({
    id,
    label,
    isActive,
    onClick,
  }: {
    id: string;
    label: string;
    isActive: boolean;
    onClick: () => void;
  }) => (
    <button
      onClick={onClick}
      className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
        isActive
          ? "bg-blue-600 text-white border-b-2 border-blue-600"
          : "bg-gray-800 text-gray-300 hover:bg-gray-700"
      }`}
    >
      {label}
    </button>
  );

  const InputField = ({
    label,
    value,
    onChange,
    type = "text",
    min,
    max,
    step,
  }: {
    label: string;
    value: any;
    onChange: (value: any) => void;
    type?: string;
    min?: number;
    max?: number;
    step?: number;
  }) => (
    <div className="space-y-1">
      <label className="block text-sm font-medium text-gray-300">{label}</label>
      <input
        type={type}
        value={value}
        onChange={(e) =>
          onChange(type === "number" ? Number(e.target.value) : e.target.value)
        }
        min={min}
        max={max}
        step={step}
        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      />
    </div>
  );

  const SelectField = ({
    label,
    value,
    onChange,
    options,
  }: {
    label: string;
    value: string;
    onChange: (value: string) => void;
    options: { value: string; label: string }[];
  }) => (
    <div className="space-y-1">
      <label className="block text-sm font-medium text-gray-300">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold mb-4">最適化設定</h2>

      {/* タブナビゲーション */}
      <div className="flex space-x-1 mb-6">
        <TabButton
          id="enhanced"
          label="拡張最適化"
          isActive={activeTab === "enhanced"}
          onClick={() => setActiveTab("enhanced")}
        />
        <TabButton
          id="multi"
          label="マルチ目的最適化"
          isActive={activeTab === "multi"}
          onClick={() => setActiveTab("multi")}
        />
        <TabButton
          id="robustness"
          label="ロバストネステスト"
          isActive={activeTab === "robustness"}
          onClick={() => setActiveTab("robustness")}
        />
      </div>

      {/* 基本設定 */}
      <div className="mb-6 p-4 bg-gray-700 rounded-lg">
        <h3 className="text-lg font-medium mb-3">基本設定</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <InputField
            label="戦略名"
            value={baseConfig.strategy_name}
            onChange={(value) =>
              setBaseConfig({ ...baseConfig, strategy_name: value })
            }
          />
          <SelectField
            label="シンボル"
            value={baseConfig.symbol}
            onChange={(value) =>
              setBaseConfig({ ...baseConfig, symbol: value })
            }
            options={[{ value: "BTC/USDT", label: "BTC/USDT" }]}
          />
          <SelectField
            label="時間軸"
            value={baseConfig.timeframe}
            onChange={(value) =>
              setBaseConfig({ ...baseConfig, timeframe: value })
            }
            options={[
              { value: "1h", label: "1時間" },
              { value: "4h", label: "4時間" },
              { value: "1d", label: "1日" },
            ]}
          />
          <InputField
            label="開始日"
            value={baseConfig.start_date}
            onChange={(value) =>
              setBaseConfig({ ...baseConfig, start_date: value })
            }
            type="date"
          />
          <InputField
            label="終了日"
            value={baseConfig.end_date}
            onChange={(value) =>
              setBaseConfig({ ...baseConfig, end_date: value })
            }
            type="date"
          />
          <InputField
            label="初期資金"
            value={baseConfig.initial_capital}
            onChange={(value) =>
              setBaseConfig({ ...baseConfig, initial_capital: value })
            }
            type="number"
            min={1000000}
            step={1000000}
          />
        </div>
      </div>

      {/* 拡張最適化タブ */}
      {activeTab === "enhanced" && (
        <div className="space-y-4">
          <h3 className="text-lg font-medium">拡張最適化設定</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <SelectField
              label="最適化手法"
              value={enhancedConfig.method}
              onChange={(value) =>
                setEnhancedConfig({
                  ...enhancedConfig,
                  method: value as "grid" | "sambo",
                })
              }
              options={[
                { value: "grid", label: "Grid Search" },
                { value: "sambo", label: "SAMBO (ベイズ最適化)" },
              ]}
            />
            {enhancedConfig.method === "sambo" && (
              <InputField
                label="最大試行回数"
                value={enhancedConfig.max_tries}
                onChange={(value) =>
                  setEnhancedConfig({ ...enhancedConfig, max_tries: value })
                }
                type="number"
                min={10}
                max={500}
              />
            )}
            <SelectField
              label="最大化指標"
              value={enhancedConfig.maximize}
              onChange={(value) =>
                setEnhancedConfig({ ...enhancedConfig, maximize: value })
              }
              options={[
                { value: "Sharpe Ratio", label: "シャープレシオ" },
                { value: "Return [%]", label: "総リターン" },
                { value: "Profit Factor", label: "プロフィットファクター" },
              ]}
            />
            <SelectField
              label="制約条件"
              value={enhancedConfig.constraint}
              onChange={(value) =>
                setEnhancedConfig({ ...enhancedConfig, constraint: value })
              }
              options={[
                { value: "sma_cross", label: "SMAクロス制約" },
                { value: "rsi", label: "RSI制約" },
                { value: "macd", label: "MACD制約" },
                { value: "risk_management", label: "リスク管理制約" },
              ]}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                n1パラメータ範囲 [開始, 終了, ステップ]
              </label>
              <div className="grid grid-cols-3 gap-2">
                <input
                  type="number"
                  value={enhancedConfig.n1_range[0]}
                  onChange={(e) =>
                    setEnhancedConfig({
                      ...enhancedConfig,
                      n1_range: [
                        Number(e.target.value),
                        enhancedConfig.n1_range[1],
                        enhancedConfig.n1_range[2],
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="開始"
                />
                <input
                  type="number"
                  value={enhancedConfig.n1_range[1]}
                  onChange={(e) =>
                    setEnhancedConfig({
                      ...enhancedConfig,
                      n1_range: [
                        enhancedConfig.n1_range[0],
                        Number(e.target.value),
                        enhancedConfig.n1_range[2],
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="終了"
                />
                <input
                  type="number"
                  value={enhancedConfig.n1_range[2]}
                  onChange={(e) =>
                    setEnhancedConfig({
                      ...enhancedConfig,
                      n1_range: [
                        enhancedConfig.n1_range[0],
                        enhancedConfig.n1_range[1],
                        Number(e.target.value),
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="ステップ"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                n2パラメータ範囲 [開始, 終了, ステップ]
              </label>
              <div className="grid grid-cols-3 gap-2">
                <input
                  type="number"
                  value={enhancedConfig.n2_range[0]}
                  onChange={(e) =>
                    setEnhancedConfig({
                      ...enhancedConfig,
                      n2_range: [
                        Number(e.target.value),
                        enhancedConfig.n2_range[1],
                        enhancedConfig.n2_range[2],
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="開始"
                />
                <input
                  type="number"
                  value={enhancedConfig.n2_range[1]}
                  onChange={(e) =>
                    setEnhancedConfig({
                      ...enhancedConfig,
                      n2_range: [
                        enhancedConfig.n2_range[0],
                        Number(e.target.value),
                        enhancedConfig.n2_range[2],
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="終了"
                />
                <input
                  type="number"
                  value={enhancedConfig.n2_range[2]}
                  onChange={(e) =>
                    setEnhancedConfig({
                      ...enhancedConfig,
                      n2_range: [
                        enhancedConfig.n2_range[0],
                        enhancedConfig.n2_range[1],
                        Number(e.target.value),
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="ステップ"
                />
              </div>
            </div>
          </div>

          <div className="pt-4">
            <ApiButton
              onClick={handleEnhancedSubmit}
              loading={isLoading}
              loadingText="最適化実行中..."
              variant="primary"
              size="lg"
              className="w-full"
            >
              拡張最適化を実行
            </ApiButton>
          </div>
        </div>
      )}

      {/* マルチ目的最適化タブ */}
      {activeTab === "multi" && (
        <div className="space-y-4">
          <h3 className="text-lg font-medium">マルチ目的最適化設定</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                目的関数
              </label>
              <div className="space-y-2">
                {[
                  "Sharpe Ratio",
                  "Return [%]",
                  "-Max. Drawdown [%]",
                  "Profit Factor",
                  "-Volatility",
                ].map((objective, index) => (
                  <label key={objective} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={multiConfig.objectives.includes(objective)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setMultiConfig({
                            ...multiConfig,
                            objectives: [...multiConfig.objectives, objective],
                          });
                        } else {
                          setMultiConfig({
                            ...multiConfig,
                            objectives: multiConfig.objectives.filter(
                              (obj) => obj !== objective
                            ),
                          });
                        }
                      }}
                      className="mr-2"
                    />
                    <span className="text-sm text-gray-300">{objective}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                重み設定
              </label>
              <div className="space-y-2">
                {multiConfig.objectives.map((objective, index) => (
                  <div key={objective} className="flex items-center space-x-2">
                    <span className="text-sm text-gray-300 w-32">
                      {objective}:
                    </span>
                    <input
                      type="number"
                      value={multiConfig.weights[index] || 0}
                      onChange={(e) => {
                        const newWeights = [...multiConfig.weights];
                        newWeights[index] = Number(e.target.value);
                        setMultiConfig({ ...multiConfig, weights: newWeights });
                      }}
                      min={0}
                      max={1}
                      step={0.1}
                      className="px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm"
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <SelectField
              label="最適化手法"
              value={multiConfig.method}
              onChange={(value) =>
                setMultiConfig({
                  ...multiConfig,
                  method: value as "grid" | "sambo",
                })
              }
              options={[
                { value: "grid", label: "Grid Search" },
                { value: "sambo", label: "SAMBO (ベイズ最適化)" },
              ]}
            />
            {multiConfig.method === "sambo" && (
              <InputField
                label="最大試行回数"
                value={multiConfig.max_tries}
                onChange={(value) =>
                  setMultiConfig({ ...multiConfig, max_tries: value })
                }
                type="number"
                min={10}
                max={300}
              />
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                n1パラメータ範囲 [開始, 終了, ステップ]
              </label>
              <div className="grid grid-cols-3 gap-2">
                <input
                  type="number"
                  value={multiConfig.n1_range[0]}
                  onChange={(e) =>
                    setMultiConfig({
                      ...multiConfig,
                      n1_range: [
                        Number(e.target.value),
                        multiConfig.n1_range[1],
                        multiConfig.n1_range[2],
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="開始"
                />
                <input
                  type="number"
                  value={multiConfig.n1_range[1]}
                  onChange={(e) =>
                    setMultiConfig({
                      ...multiConfig,
                      n1_range: [
                        multiConfig.n1_range[0],
                        Number(e.target.value),
                        multiConfig.n1_range[2],
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="終了"
                />
                <input
                  type="number"
                  value={multiConfig.n1_range[2]}
                  onChange={(e) =>
                    setMultiConfig({
                      ...multiConfig,
                      n1_range: [
                        multiConfig.n1_range[0],
                        multiConfig.n1_range[1],
                        Number(e.target.value),
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="ステップ"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                n2パラメータ範囲 [開始, 終了, ステップ]
              </label>
              <div className="grid grid-cols-3 gap-2">
                <input
                  type="number"
                  value={multiConfig.n2_range[0]}
                  onChange={(e) =>
                    setMultiConfig({
                      ...multiConfig,
                      n2_range: [
                        Number(e.target.value),
                        multiConfig.n2_range[1],
                        multiConfig.n2_range[2],
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="開始"
                />
                <input
                  type="number"
                  value={multiConfig.n2_range[1]}
                  onChange={(e) =>
                    setMultiConfig({
                      ...multiConfig,
                      n2_range: [
                        multiConfig.n2_range[0],
                        Number(e.target.value),
                        multiConfig.n2_range[2],
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="終了"
                />
                <input
                  type="number"
                  value={multiConfig.n2_range[2]}
                  onChange={(e) =>
                    setMultiConfig({
                      ...multiConfig,
                      n2_range: [
                        multiConfig.n2_range[0],
                        multiConfig.n2_range[1],
                        Number(e.target.value),
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="ステップ"
                />
              </div>
            </div>
          </div>

          <div className="pt-4">
            <ApiButton
              onClick={handleMultiObjectiveSubmit}
              loading={isLoading}
              loadingText="マルチ目的最適化実行中..."
              variant="success"
              size="lg"
              className="w-full"
            >
              マルチ目的最適化を実行
            </ApiButton>
          </div>
        </div>
      )}

      {/* ロバストネステストタブ */}
      {activeTab === "robustness" && (
        <div className="space-y-4">
          <h3 className="text-lg font-medium">ロバストネステスト設定</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <SelectField
              label="最適化手法"
              value={robustnessConfig.method}
              onChange={(value) =>
                setRobustnessConfig({
                  ...robustnessConfig,
                  method: value as "grid" | "sambo",
                })
              }
              options={[
                { value: "grid", label: "Grid Search" },
                { value: "sambo", label: "SAMBO (ベイズ最適化)" },
              ]}
            />
            {robustnessConfig.method === "sambo" && (
              <InputField
                label="最大試行回数"
                value={robustnessConfig.max_tries}
                onChange={(value) =>
                  setRobustnessConfig({ ...robustnessConfig, max_tries: value })
                }
                type="number"
                min={10}
                max={200}
              />
            )}
            <SelectField
              label="最大化指標"
              value={robustnessConfig.maximize}
              onChange={(value) =>
                setRobustnessConfig({ ...robustnessConfig, maximize: value })
              }
              options={[
                { value: "Sharpe Ratio", label: "シャープレシオ" },
                { value: "Return [%]", label: "総リターン" },
                { value: "Profit Factor", label: "プロフィットファクター" },
              ]}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              テスト期間設定
            </label>
            <div className="space-y-2">
              {robustnessConfig.test_periods.map((period, index) => (
                <div
                  key={index}
                  className="grid grid-cols-3 gap-2 items-center"
                >
                  <input
                    type="date"
                    value={period[0]}
                    onChange={(e) => {
                      const newPeriods = [...robustnessConfig.test_periods];
                      newPeriods[index][0] = e.target.value;
                      setRobustnessConfig({
                        ...robustnessConfig,
                        test_periods: newPeriods,
                      });
                    }}
                    className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  />
                  <input
                    type="date"
                    value={period[1]}
                    onChange={(e) => {
                      const newPeriods = [...robustnessConfig.test_periods];
                      newPeriods[index][1] = e.target.value;
                      setRobustnessConfig({
                        ...robustnessConfig,
                        test_periods: newPeriods,
                      });
                    }}
                    className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  />
                  <button
                    onClick={() => {
                      const newPeriods = robustnessConfig.test_periods.filter(
                        (_, i) => i !== index
                      );
                      setRobustnessConfig({
                        ...robustnessConfig,
                        test_periods: newPeriods,
                      });
                    }}
                    className="px-3 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
                  >
                    削除
                  </button>
                </div>
              ))}
              <button
                onClick={() => {
                  const newPeriods = [
                    ...robustnessConfig.test_periods,
                    ["2024-01-01", "2024-03-31"],
                  ];
                  setRobustnessConfig({
                    ...robustnessConfig,
                    test_periods: newPeriods,
                  });
                }}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                期間を追加
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                n1パラメータ範囲 [開始, 終了, ステップ]
              </label>
              <div className="grid grid-cols-3 gap-2">
                <input
                  type="number"
                  value={robustnessConfig.n1_range[0]}
                  onChange={(e) =>
                    setRobustnessConfig({
                      ...robustnessConfig,
                      n1_range: [
                        Number(e.target.value),
                        robustnessConfig.n1_range[1],
                        robustnessConfig.n1_range[2],
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="開始"
                />
                <input
                  type="number"
                  value={robustnessConfig.n1_range[1]}
                  onChange={(e) =>
                    setRobustnessConfig({
                      ...robustnessConfig,
                      n1_range: [
                        robustnessConfig.n1_range[0],
                        Number(e.target.value),
                        robustnessConfig.n1_range[2],
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="終了"
                />
                <input
                  type="number"
                  value={robustnessConfig.n1_range[2]}
                  onChange={(e) =>
                    setRobustnessConfig({
                      ...robustnessConfig,
                      n1_range: [
                        robustnessConfig.n1_range[0],
                        robustnessConfig.n1_range[1],
                        Number(e.target.value),
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="ステップ"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                n2パラメータ範囲 [開始, 終了, ステップ]
              </label>
              <div className="grid grid-cols-3 gap-2">
                <input
                  type="number"
                  value={robustnessConfig.n2_range[0]}
                  onChange={(e) =>
                    setRobustnessConfig({
                      ...robustnessConfig,
                      n2_range: [
                        Number(e.target.value),
                        robustnessConfig.n2_range[1],
                        robustnessConfig.n2_range[2],
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="開始"
                />
                <input
                  type="number"
                  value={robustnessConfig.n2_range[1]}
                  onChange={(e) =>
                    setRobustnessConfig({
                      ...robustnessConfig,
                      n2_range: [
                        robustnessConfig.n2_range[0],
                        Number(e.target.value),
                        robustnessConfig.n2_range[2],
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="終了"
                />
                <input
                  type="number"
                  value={robustnessConfig.n2_range[2]}
                  onChange={(e) =>
                    setRobustnessConfig({
                      ...robustnessConfig,
                      n2_range: [
                        robustnessConfig.n2_range[0],
                        robustnessConfig.n2_range[1],
                        Number(e.target.value),
                      ],
                    })
                  }
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  placeholder="ステップ"
                />
              </div>
            </div>
          </div>

          <div className="pt-4">
            <ApiButton
              onClick={handleRobustnessSubmit}
              loading={isLoading}
              loadingText="ロバストネステスト実行中..."
              variant="warning"
              size="lg"
              className="w-full"
            >
              ロバストネステストを実行
            </ApiButton>
          </div>
        </div>
      )}
    </div>
  );
}
