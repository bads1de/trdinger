/**
 * GA設定フォームコンポーネント
 *
 * 遺伝的アルゴリズムによる自動戦略生成の設定を行います。
 */

"use client";

import React, { useState, useEffect } from "react";
import { InputField } from "@/components/common/InputField";
import { SelectField } from "@/components/common/SelectField";
import ApiButton from "@/components/button/ApiButton";
import { GAConfig as GAConfigType } from "@/types/optimization";
import { BacktestConfig as BacktestConfigType } from "@/types/backtest";
import { BaseBacktestConfigForm } from "./BaseBacktestConfigForm";
import { GA_OBJECTIVE_OPTIONS } from "@/constants/backtest";

interface GAConfigFormProps {
  onSubmit: (config: GAConfigType) => void;
  initialConfig?: Partial<GAConfigType>;
  isLoading?: boolean;
  currentBacktestConfig?: BacktestConfigType | null;
}

const GAConfigForm: React.FC<GAConfigFormProps> = ({
  onSubmit,
  initialConfig = {},
  isLoading = false,
  currentBacktestConfig = null,
}) => {
  const [config, setConfig] = useState<GAConfigType>(() => {
    const baseBacktestConfig: BacktestConfigType = {
      strategy_name: "GA_STRATEGY",
      symbol: "BTC/USDT",
      timeframe: "1h",
      start_date: "2023-01-01",
      end_date: "2023-12-31",
      initial_capital: 100000,
      commission_rate: 0.00055,
      strategy_config: {
        strategy_type: "", // GAで生成されるため、初期値は空
        parameters: {},
      },
    };

    const effectiveBaseConfig = currentBacktestConfig || baseBacktestConfig;

    return {
      experiment_name:
        initialConfig.experiment_name ||
        `GA_${new Date()
          .toISOString()
          .slice(0, 10)}_${effectiveBaseConfig.symbol.replace("/", "_")}`,
      base_config: effectiveBaseConfig,
      ga_config: {
        population_size: initialConfig.ga_config?.population_size || 50, // 100→50に最適化
        generations: initialConfig.ga_config?.generations || 20, // 50→20に最適化
        mutation_rate: initialConfig.ga_config?.mutation_rate || 0.1,
        crossover_rate: initialConfig.ga_config?.crossover_rate || 0.8, // 0.7→0.8に調整
        elite_size: initialConfig.ga_config?.elite_size || 5,
        max_indicators: initialConfig.ga_config?.max_indicators || 5,
        allowed_indicators: initialConfig.ga_config?.allowed_indicators || [],
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
        ga_objective: initialConfig.ga_config?.ga_objective || "Sharpe Ratio",
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

  const handleSubmit = () => {
    onSubmit(config);
  };

  return (
    <form onSubmit={(e) => e.preventDefault()} className="space-y-6">
      <h2 className="text-xl font-semibold text-white">
        遺伝的アルゴリズム設定
      </h2>

      <BaseBacktestConfigForm
        config={config.base_config}
        onConfigChange={handleBaseConfigChange}
        isOptimization={true}
      />

      <InputField
        label="実験名 (experiment_name)"
        value={config.experiment_name}
        onChange={(value) => setConfig({ ...config, experiment_name: value })}
        required
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
      />

      <SelectField
        label="最適化目的 (ga_objective)"
        value={config.ga_config.ga_objective}
        onChange={(value) =>
          setConfig((prev) => ({
            ...prev,
            ga_config: { ...prev.ga_config, ga_objective: value },
          }))
        }
        options={GA_OBJECTIVE_OPTIONS}
        required
      />

      <ApiButton onClick={handleSubmit} loading={isLoading}>
        GA戦略を生成
      </ApiButton>
    </form>
  );
};

export default GAConfigForm;
