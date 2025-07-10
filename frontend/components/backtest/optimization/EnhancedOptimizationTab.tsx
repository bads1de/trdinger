/**
 * EnhancedOptimizationTab.tsx
 *
 * 拡張最適化設定タブのコンポーネント
 */
"use client";

import React, { useState } from "react";
import { InputField } from "@/components/common/InputField";
import { SelectField } from "@/components/common/SelectField";
import ApiButton from "@/components/button/ApiButton";
import {
  OPTIMIZATION_METHODS,
  ENHANCED_OPTIMIZATION_OBJECTIVES,
} from "@/constants/backtest";
import { OptimizationConfig } from "@/types/optimization";
import { BacktestConfig } from "@/types/backtest";

interface EnhancedOptimizationTabProps {
  baseConfig: BacktestConfig;
  selectedStrategy: string;
  onRun: (config: OptimizationConfig) => void;
  isLoading: boolean;
}

export default function EnhancedOptimizationTab({
  baseConfig,
  selectedStrategy,
  onRun,
  isLoading,
}: EnhancedOptimizationTabProps) {
  const [enhancedConfig, setEnhancedConfig] = useState({
    method: "grid" as "grid" | "sambo",
    max_tries: 100,
    maximize: "Sharpe Ratio",
    return_optimization: false,
    random_state: 42,
    constraint: "sma_cross",
    n1_range: [10, 30, 5],
    n2_range: [30, 80, 10],
  });

  const createParameterRange = (rangeConfig: number[]) => {
    const [start, end, step] = rangeConfig;
    const range = [];
    for (let i = start; i <= end; i += step) {
      range.push(i);
    }
    return range;
  };

  const handleSubmit = () => {
    const config: OptimizationConfig = {
      base_config: {
        ...baseConfig,
        strategy_config: {
          strategy_type: selectedStrategy,
          parameters: baseConfig.strategy_config.parameters,
        },
      },
      optimization_params: {
        method: enhancedConfig.method,
        ...(enhancedConfig.method === "sambo" && {
          max_tries: enhancedConfig.max_tries,
        }),
        maximize: enhancedConfig.maximize,
        return_optimization: enhancedConfig.return_optimization,
        random_state: enhancedConfig.random_state,
        constraint: enhancedConfig.constraint,
        parameters: {
          n1: createParameterRange(enhancedConfig.n1_range),
          n2: createParameterRange(enhancedConfig.n2_range),
        },
      },
    };
    onRun(config);
  };

  return (
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
      {enhancedConfig.method === "sambo" && (
        <>
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
        </>
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
      <ApiButton onClick={handleSubmit} loading={isLoading}>
        拡張最適化を実行
      </ApiButton>
    </div>
  );
}