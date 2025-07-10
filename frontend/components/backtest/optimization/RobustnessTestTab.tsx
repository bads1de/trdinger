/**
 * RobustnessTestTab.tsx
 *
 * ロバストネステスト設定タブのコンポーネント
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
import { RobustnessConfig } from "@/types/optimization";
import { BacktestConfig } from "@/types/backtest";

interface RobustnessTestTabProps {
  baseConfig: Omit<BacktestConfig, 'start_date' | 'end_date'>;
  selectedStrategy: string;
  onRun: (config: RobustnessConfig) => void;
  isLoading: boolean;
}

export default function RobustnessTestTab({
  baseConfig,
  selectedStrategy,
  onRun,
  isLoading,
}: RobustnessTestTabProps) {
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

  const createParameterRange = (rangeConfig: number[]) => {
    const [start, end, step] = rangeConfig;
    const range = [];
    for (let i = start; i <= end; i += step) {
      range.push(i);
    }
    return range;
  };

  const handleSubmit = () => {
    const config: RobustnessConfig = {
      base_config: {
        ...baseConfig,
        strategy_config: {
          strategy_type: selectedStrategy,
          parameters: baseConfig.strategy_config.parameters,
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
    onRun(config);
  };

  const handlePeriodChange = (index: number, position: number, value: string) => {
    const newPeriods = [...robustnessConfig.test_periods];
    newPeriods[index][position] = value;
    setRobustnessConfig((prev) => ({ ...prev, test_periods: newPeriods }));
  };

  const addPeriod = () => {
    setRobustnessConfig((prev) => ({
      ...prev,
      test_periods: [...prev.test_periods, ["", ""]],
    }));
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold text-white">
        ロバストネステスト設定
      </h2>
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-300">
          テスト期間
        </label>
        {robustnessConfig.test_periods.map((period, index) => (
          <div key={index} className="flex items-center gap-2">
            <InputField
              label={`期間 ${index + 1} 開始`}
              type="date"
              value={period[0]}
              onChange={(value) => handlePeriodChange(index, 0, value)}
            />
            <InputField
              label={`期間 ${index + 1} 終了`}
              type="date"
              value={period[1]}
              onChange={(value) => handlePeriodChange(index, 1, value)}
            />
          </div>
        ))}
        <button
          type="button"
          onClick={addPeriod}
          className="text-sm text-blue-400 hover:text-blue-300"
        >
          + 期間を追加
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
      <ApiButton onClick={handleSubmit} loading={isLoading}>
        ロバストネステストを実行
      </ApiButton>
    </div>
  );
}