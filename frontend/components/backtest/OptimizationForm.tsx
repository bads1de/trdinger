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
import GAConfigForm from "./GAConfigForm";
import GAProgressDisplay from "./GAProgressDisplay";
import { useGAExecution } from "@/hooks/useGAProgress";
import { BacktestConfig, BacktestResult } from "@/types/backtest";
import {
  OptimizationConfig,
  MultiObjectiveConfig,
  RobustnessConfig,
  GAConfig,
} from "@/types/optimization";
import { InputField } from "@/components/common/InputField";
import { SelectField } from "@/components/common/SelectField";
import { BaseBacktestConfigForm } from "./BaseBacktestConfigForm";
import { Strategy } from "@/types/strategy";
import {
  OPTIMIZATION_METHODS,
  ENHANCED_OPTIMIZATION_OBJECTIVES,
  GA_OBJECTIVE_OPTIONS,
} from "@/constants/backtest";
import TabButton from "../common/TabButton";

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
  const [strategies, setStrategies] = useState<Record<string, Strategy>>({});
  const [selectedStrategy, setSelectedStrategy] = useState<string>("SMA_CROSS");

  // 基本設定
  const [baseConfig, setBaseConfig] = useState({
    strategy_name: "OPTIMIZED_STRATEGY",
    symbol: "BTC/USDT",
    timeframe: "1d",
    start_date: "2024-01-01",
    end_date: "2024-12-31",
    initial_capital: 10000000,
    commission_rate: 0.00055,
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

  // GA実行管理
  const gaExecution = useGAExecution();

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

  // 現在のバックテスト設定から基本設定を自動入力（優先）
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
      // フォールバック: 選択されたバックテスト結果から設定を引き継ぎ
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

  const createParameterRange = (rangeConfig: number[]) => {
    const [start, end, step] = rangeConfig;
    const range = [];
    for (let i = start; i <= end; i += step) {
      range.push(i);
    }
    return range;
  };

  const handleEnhancedSubmit = () => {
    // 現在のバックテスト設定から戦略パラメータを引き継ぐ
    const strategyParameters =
      currentBacktestConfig?.strategy_config?.parameters || {};

    const config: OptimizationConfig = {
      base_config: {
        ...baseConfig,
        strategy_config: {
          strategy_type: selectedStrategy,
          parameters: strategyParameters,
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
    // 現在のバックテスト設定から戦略パラメータを引き継ぐ
    const strategyParameters =
      currentBacktestConfig?.strategy_config?.parameters || {};

    const config: MultiObjectiveConfig = {
      base_config: {
        ...baseConfig,
        strategy_config: {
          strategy_type: selectedStrategy,
          parameters: strategyParameters,
        },
      },
      optimization_params: {
        objectives: multiConfig.objectives,
        weights: multiConfig.weights || [], // multiConfig.weights が undefined の場合のフォールバックを提供
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
    // 現在のバックテスト設定から戦略パラメータを引き継ぐ
    const strategyParameters =
      currentBacktestConfig?.strategy_config?.parameters || {};

    const config: RobustnessConfig = {
      base_config: {
        strategy_name: baseConfig.strategy_name,
        symbol: baseConfig.symbol,
        timeframe: baseConfig.timeframe,
        initial_capital: baseConfig.initial_capital,
        commission_rate: baseConfig.commission_rate,
        strategy_config: {
          strategy_type: selectedStrategy,
          parameters: strategyParameters,
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

  // GA実行ハンドラー
  const handleGAGeneration = (gaConfig: GAConfig) => {
    if (onGAGeneration) {
      const strategyParameters =
        currentBacktestConfig?.strategy_config?.parameters || {};
      const baseBacktestConfig: BacktestConfig = {
        ...baseConfig,
        strategy_config: {
          strategy_type: selectedStrategy,
          parameters: strategyParameters,
        },
      };
      onGAGeneration({ ...gaConfig, base_config: baseBacktestConfig });
    }
  };

  // GAProgressDisplay に渡す onComplete と onError のダミー関数
  const handleGAProgressComplete = (result: any) => {
    console.log("GA completed in OptimizationForm:", result);
  };

  const handleGAProgressError = (error: string) => {
    console.error("GA error in OptimizationForm:", error);
  };

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
        {/* 基本設定はすべての最適化で共通 */}
        <h2 className="text-xl font-semibold text-white">基本設定</h2>
        <BaseBacktestConfigForm
          config={{
            strategy_name: baseConfig.strategy_name,
            symbol: baseConfig.symbol,
            timeframe: baseConfig.timeframe,
            start_date: baseConfig.start_date,
            end_date: baseConfig.end_date,
            initial_capital: baseConfig.initial_capital,
            commission_rate: baseConfig.commission_rate,
            strategy_config: {
              strategy_type: selectedStrategy,
              parameters:
                currentBacktestConfig?.strategy_config?.parameters || {},
            },
          }}
          onConfigChange={(updates) =>
            setBaseConfig((prev) => ({ ...prev, ...updates }))
          }
          isOptimization={true}
        />

        {/* 戦略パラメータの表示と編集 */}
        {selectedStrategy && strategies[selectedStrategy] && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white">戦略パラメータ</h3>
            {Object.entries(strategies[selectedStrategy].parameters).map(
              ([key, param]) => (
                <InputField
                  key={key}
                  label={param.description}
                  type="number"
                  value={
                    currentBacktestConfig?.strategy_config?.parameters[key] ||
                    param.default
                  }
                  onChange={(value) => {
                    // 最適化フォームでは直接パラメータを変更しないため、変更を無視
                    console.log(
                      `OptimizationForm: Parameter ${key} changed to ${value}, but not applied.`
                    );
                  }}
                  min={param.min}
                  max={param.max}
                  step={param.step}
                  required
                  className="cursor-not-allowed opacity-70"
                />
              )
            )}
            {strategies[selectedStrategy].constraints && (
              <div className="mt-4 p-3 bg-yellow-900/20 border border-yellow-600/30 rounded-md">
                <h4 className="text-sm font-medium text-yellow-400 mb-2">
                  制約条件:
                </h4>
                <ul className="text-sm text-yellow-300">
                  {strategies[selectedStrategy].constraints.map(
                    (constraint: string, index: number) => (
                      <li key={index}>• {constraint}</li>
                    )
                  )}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* タブコンテンツ */}
        {activeTab === "enhanced" && (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-white">拡張最適化設定</h2>
            <SelectField
              label="最適化手法"
              value={enhancedConfig.method}
              onChange={(value) =>
                setEnhancedConfig((prev) => ({
                  ...prev,
                  method: value as "grid" | "sambo",
                }))
              }
              options={OPTIMIZATION_METHODS}
            />
            {enhancedConfig.method === "sambo" && (
              <InputField
                label="最大試行回数 (max_tries)"
                type="number"
                value={enhancedConfig.max_tries}
                onChange={(value) =>
                  setEnhancedConfig((prev) => ({ ...prev, max_tries: value }))
                }
                min={10}
                step={10}
              />
            )}
            <SelectField
              label="最大化する指標"
              value={enhancedConfig.maximize}
              onChange={(value) =>
                setEnhancedConfig((prev) => ({ ...prev, maximize: value }))
              }
              options={ENHANCED_OPTIMIZATION_OBJECTIVES}
            />
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="return_heatmap"
                checked={enhancedConfig.return_heatmap}
                onChange={(e) =>
                  setEnhancedConfig((prev) => ({
                    ...prev,
                    return_heatmap: e.target.checked,
                  }))
                }
                className="form-checkbox h-5 w-5 text-blue-600 rounded border-gray-600 bg-gray-700"
              />
              <label htmlFor="return_heatmap" className="text-gray-300">
                ヒートマップを返す
              </label>
            </div>
            {enhancedConfig.method === "sambo" && (
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="return_optimization"
                  checked={enhancedConfig.return_optimization}
                  onChange={(e) =>
                    setEnhancedConfig((prev) => ({
                      ...prev,
                      return_optimization: e.target.checked,
                    }))
                  }
                  className="form-checkbox h-5 w-5 text-blue-600 rounded border-gray-600 bg-gray-700"
                />
                <label htmlFor="return_optimization" className="text-gray-300">
                  最適化結果を返す (Samboのみ)
                </label>
              </div>
            )}
            {enhancedConfig.method === "sambo" && (
              <InputField
                label="ランダムシード (random_state)"
                type="number"
                value={enhancedConfig.random_state}
                onChange={(value) =>
                  setEnhancedConfig((prev) => ({
                    ...prev,
                    random_state: value,
                  }))
                }
                min={0}
              />
            )}
            <InputField
              label="制約 (constraint)"
              value={enhancedConfig.constraint}
              onChange={(value) =>
                setEnhancedConfig((prev) => ({ ...prev, constraint: value }))
              }
            />
            <InputField
              label="N1範囲 [start, end, step]"
              value={enhancedConfig.n1_range.join(", ")}
              onChange={(value) =>
                setEnhancedConfig((prev) => ({
                  ...prev,
                  n1_range: value.split(",").map(Number),
                }))
              }
              placeholder="例: 10, 30, 5"
            />
            <InputField
              label="N2範囲 [start, end, step]"
              value={enhancedConfig.n2_range.join(", ")}
              onChange={(value) =>
                setEnhancedConfig((prev) => ({
                  ...prev,
                  n2_range: value.split(",").map(Number),
                }))
              }
              placeholder="例: 30, 80, 10"
            />
            <ApiButton onClick={handleEnhancedSubmit} loading={isLoading}>
              拡張最適化を実行
            </ApiButton>
          </div>
        )}

        {activeTab === "multi" && (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-white">
              マルチ目的最適化設定
            </h2>
            <SelectField
              label="最適化手法"
              value={multiConfig.method}
              onChange={(value) =>
                setMultiConfig((prev) => ({
                  ...prev,
                  method: value as "grid" | "sambo",
                }))
              }
              options={OPTIMIZATION_METHODS}
            />
            {multiConfig.method === "sambo" && (
              <InputField
                label="最大試行回数 (max_tries)"
                type="number"
                value={multiConfig.max_tries}
                onChange={(value) =>
                  setMultiConfig((prev) => ({ ...prev, max_tries: value }))
                }
                min={10}
                step={10}
              />
            )}
            <InputField
              label="目的関数 [指標1, 指標2,...]"
              value={multiConfig.objectives.join(", ")}
              onChange={(value) =>
                setMultiConfig((prev) => ({
                  ...prev,
                  objectives: value.split(",").map((s: string) => s.trim()),
                }))
              }
              placeholder="例: Sharpe Ratio, Return [%], -Max. Drawdown [%]"
            />
            <InputField
              label="重み [重み1, 重み2,...] (合計1.0)"
              value={multiConfig.weights.join(", ")}
              onChange={(value) =>
                setMultiConfig((prev) => ({
                  ...prev,
                  weights: value.split(",").map(Number),
                }))
              }
              placeholder="例: 0.4, 0.4, 0.2"
            />
            <InputField
              label="N1範囲 [start, end, step]"
              value={multiConfig.n1_range.join(", ")}
              onChange={(value) =>
                setMultiConfig((prev) => ({
                  ...prev,
                  n1_range: value.split(",").map(Number),
                }))
              }
              placeholder="例: 10, 25, 5"
            />
            <InputField
              label="N2範囲 [start, end, step]"
              value={multiConfig.n2_range.join(", ")}
              onChange={(value) =>
                setMultiConfig((prev) => ({
                  ...prev,
                  n2_range: value.split(",").map(Number),
                }))
              }
              placeholder="例: 30, 70, 10"
            />
            <ApiButton onClick={handleMultiObjectiveSubmit} loading={isLoading}>
              マルチ目的最適化を実行
            </ApiButton>
          </div>
        )}

        {activeTab === "robustness" && (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-white">
              ロバストネステスト設定
            </h2>
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                テスト期間 (test_periods)
              </label>
              {robustnessConfig.test_periods.map((period, index) => (
                <div key={index} className="flex gap-2">
                  <InputField
                    label={`期間 ${index + 1} 開始`}
                    type="date"
                    value={period[0]}
                    onChange={(value) => {
                      const newPeriods = [...robustnessConfig.test_periods];
                      newPeriods[index][0] = value;
                      setRobustnessConfig((prev) => ({
                        ...prev,
                        test_periods: newPeriods,
                      }));
                    }}
                  />
                  <InputField
                    label={`期間 ${index + 1} 終了`}
                    type="date"
                    value={period[1]}
                    onChange={(value) => {
                      const newPeriods = [...robustnessConfig.test_periods];
                      newPeriods[index][1] = value;
                      setRobustnessConfig((prev) => ({
                        ...prev,
                        test_periods: newPeriods,
                      }));
                    }}
                  />
                </div>
              ))}
              <button
                type="button"
                onClick={() =>
                  setRobustnessConfig((prev) => ({
                    ...prev,
                    test_periods: [...prev.test_periods, ["", ""]],
                  }))
                }
                className="mt-2 py-2 px-4 bg-gray-700 hover:bg-gray-600 rounded-md text-white text-sm"
              >
                期間を追加
              </button>
            </div>
            <SelectField
              label="最適化手法"
              value={robustnessConfig.method}
              onChange={(value) =>
                setRobustnessConfig((prev) => ({
                  ...prev,
                  method: value as "grid" | "sambo",
                }))
              }
              options={OPTIMIZATION_METHODS}
            />
            {robustnessConfig.method === "sambo" && (
              <InputField
                label="最大試行回数 (max_tries)"
                type="number"
                value={robustnessConfig.max_tries}
                onChange={(value) =>
                  setRobustnessConfig((prev) => ({ ...prev, max_tries: value }))
                }
                min={10}
                step={10}
              />
            )}
            <SelectField
              label="最大化する指標"
              value={robustnessConfig.maximize}
              onChange={(value) =>
                setRobustnessConfig((prev) => ({ ...prev, maximize: value }))
              }
              options={ENHANCED_OPTIMIZATION_OBJECTIVES}
            />
            <InputField
              label="N1範囲 [start, end, step]"
              value={robustnessConfig.n1_range.join(", ")}
              onChange={(value) =>
                setRobustnessConfig((prev) => ({
                  ...prev,
                  n1_range: value.split(",").map(Number),
                }))
              }
              placeholder="例: 10, 25, 5"
            />
            <InputField
              label="N2範囲 [start, end, step]"
              value={robustnessConfig.n2_range.join(", ")}
              onChange={(value) =>
                setRobustnessConfig((prev) => ({
                  ...prev,
                  n2_range: value.split(",").map(Number),
                }))
              }
              placeholder="例: 30, 60, 10"
            />
            <ApiButton onClick={handleRobustnessSubmit} loading={isLoading}>
              ロバストネステストを実行
            </ApiButton>
          </div>
        )}

        {activeTab === "ga" && onGAGeneration && (
          <div className="space-y-6">
            <GAConfigForm onSubmit={handleGAGeneration} isLoading={isLoading} />
            {gaExecution.experimentId && (
              <GAProgressDisplay
                experimentId={gaExecution.experimentId}
                onComplete={handleGAProgressComplete}
                onError={handleGAProgressError}
              />
            )}
          </div>
        )}
      </form>
    </div>
  );
}
