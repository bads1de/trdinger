/**
 * MultiObjectiveTab.tsx
 *
 * マルチ目的最適化設定タブのコンポーネント
 */
"use client";

import React, { useState } from "react";
import { InputField } from "@/components/common/InputField";
import { SelectField } from "@/components/common/SelectField";
import ApiButton from "@/components/button/ApiButton";
import { OPTIMIZATION_METHODS } from "@/constants/backtest";
import { MultiObjectiveConfig } from "@/types/optimization";
import { BacktestConfig } from "@/types/backtest";

interface MultiObjectiveTabProps {
  baseConfig: BacktestConfig;
  selectedStrategy: string;
  onRun: (config: MultiObjectiveConfig) => void;
  isLoading: boolean;
}

export default function MultiObjectiveTab({
  baseConfig,
  selectedStrategy,
  onRun,
  isLoading,
}: MultiObjectiveTabProps) {
  const [multiConfig, setMultiConfig] = useState({
    objectives: ["Sharpe Ratio", "Return [%]", "-Max. Drawdown [%]"],
    weights: [0.4, 0.4, 0.2],
    method: "grid" as "grid" | "sambo",
    max_tries: 80,
    n1_range: [10, 25, 5],
    n2_range: [30, 70, 10],
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
    const config: MultiObjectiveConfig = {
      base_config: {
        ...baseConfig,
        strategy_config: {
          strategy_type: selectedStrategy,
          parameters: baseConfig.strategy_config.parameters,
        },
      },
      optimization_params: {
        objectives: multiConfig.objectives,
        weights: multiConfig.weights || [],
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
    onRun(config);
  };

  return (
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
        placeholder="例: Sharpe Ratio, Return [%]"
      />
      <InputField
        label="重み [重み1, 重み2,...]"
        value={multiConfig.weights.join(", ")}
        onChange={(value) =>
          setMultiConfig((prev) => ({
            ...prev,
            weights: value.split(",").map(Number),
          }))
        }
        placeholder="例: 0.5, 0.5"
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
      <ApiButton onClick={handleSubmit} loading={isLoading}>
        マルチ目的最適化を実行
      </ApiButton>
    </div>
  );
}