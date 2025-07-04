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
      experiment_name: `GA_${new Date()
        .toISOString()
        .slice(0, 10)}_${effectiveBaseConfig.symbol.replace("/", "_")}`,
      base_config: effectiveBaseConfig,
      ga_config: {
        population_size: initialConfig.ga_config?.population_size || 10, // 100→50→20に最適化
        generations: initialConfig.ga_config?.generations || 10, // 50→20→10に最適化
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
        // リスク管理パラメータ範囲設定
        position_size_range: initialConfig.ga_config?.position_size_range || [
          0.1, 0.5,
        ],
        stop_loss_range: initialConfig.ga_config?.stop_loss_range || [
          0.02, 0.05,
        ],
        take_profit_range: initialConfig.ga_config?.take_profit_range || [
          0.01, 0.15,
        ],
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

  const validateConfig = () => {
    const errors: string[] = [];

    // 取引量範囲のバリデーション
    if (
      config.ga_config.position_size_range[0] >=
      config.ga_config.position_size_range[1]
    ) {
      errors.push("取引量範囲: 最小値は最大値より小さくしてください");
    }
    if (
      config.ga_config.position_size_range[0] < 0.01 ||
      config.ga_config.position_size_range[1] > 0.5
    ) {
      errors.push("取引量範囲: 1%〜50%の範囲で設定してください");
    }

    // ストップロス範囲のバリデーション
    if (
      config.ga_config.stop_loss_range[0] >= config.ga_config.stop_loss_range[1]
    ) {
      errors.push("ストップロス範囲: 最小値は最大値より小さくしてください");
    }
    if (
      config.ga_config.stop_loss_range[0] < 0.005 ||
      config.ga_config.stop_loss_range[1] > 0.1
    ) {
      errors.push("ストップロス範囲: 0.5%〜10%の範囲で設定してください");
    }

    // テイクプロフィット範囲のバリデーション
    if (
      config.ga_config.take_profit_range[0] >=
      config.ga_config.take_profit_range[1]
    ) {
      errors.push(
        "テイクプロフィット範囲: 最小値は最大値より小さくしてください"
      );
    }
    if (
      config.ga_config.take_profit_range[0] < 0.005 ||
      config.ga_config.take_profit_range[1] > 0.2
    ) {
      errors.push("テイクプロフィット範囲: 0.5%〜20%の範囲で設定してください");
    }

    return errors;
  };

  const handleSubmit = () => {
    const errors = validateConfig();
    if (errors.length > 0) {
      alert("設定エラー:\n" + errors.join("\n"));
      return;
    }
    onSubmit(config);
  };

  return (
    <form onSubmit={(e) => e.preventDefault()} className="space-y-6">
      <BaseBacktestConfigForm
        config={config.base_config}
        onConfigChange={handleBaseConfigChange}
        isOptimization={true}
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

      {/* リスク管理設定セクション */}
      <div className="space-y-4 p-4 border border-secondary-600 rounded-lg bg-secondary-800">
        <h3 className="text-lg font-semibold text-white">リスク管理設定</h3>
        <p className="text-sm text-secondary-400">
          戦略生成時に使用するリスク管理パラメータの範囲を設定します
        </p>

        {/* 取引量範囲設定 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <InputField
            label="取引量範囲 - 最小値 (%)"
            type="number"
            value={config.ga_config.position_size_range[0] * 100}
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                ga_config: {
                  ...prev.ga_config,
                  position_size_range: [
                    value / 100,
                    prev.ga_config.position_size_range[1],
                  ],
                },
              }))
            }
            min={1}
            max={50}
            step={1}
            required
          />
          <InputField
            label="取引量範囲 - 最大値 (%)"
            type="number"
            value={config.ga_config.position_size_range[1] * 100}
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                ga_config: {
                  ...prev.ga_config,
                  position_size_range: [
                    prev.ga_config.position_size_range[0],
                    value / 100,
                  ],
                },
              }))
            }
            min={1}
            max={50}
            step={1}
            required
          />
        </div>

        {/* ストップロス範囲設定 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <InputField
            label="ストップロス範囲 - 最小値 (%)"
            type="number"
            value={config.ga_config.stop_loss_range[0] * 100}
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                ga_config: {
                  ...prev.ga_config,
                  stop_loss_range: [
                    value / 100,
                    prev.ga_config.stop_loss_range[1],
                  ],
                },
              }))
            }
            min={0.5}
            max={10}
            step={0.1}
            required
          />
          <InputField
            label="ストップロス範囲 - 最大値 (%)"
            type="number"
            value={config.ga_config.stop_loss_range[1] * 100}
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                ga_config: {
                  ...prev.ga_config,
                  stop_loss_range: [
                    prev.ga_config.stop_loss_range[0],
                    value / 100,
                  ],
                },
              }))
            }
            min={0.5}
            max={10}
            step={0.1}
            required
          />
        </div>

        {/* テイクプロフィット範囲設定 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <InputField
            label="テイクプロフィット範囲 - 最小値 (%)"
            type="number"
            value={config.ga_config.take_profit_range[0] * 100}
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                ga_config: {
                  ...prev.ga_config,
                  take_profit_range: [
                    value / 100,
                    prev.ga_config.take_profit_range[1],
                  ],
                },
              }))
            }
            min={0.5}
            max={20}
            step={0.1}
            required
          />
          <InputField
            label="テイクプロフィット範囲 - 最大値 (%)"
            type="number"
            value={config.ga_config.take_profit_range[1] * 100}
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                ga_config: {
                  ...prev.ga_config,
                  take_profit_range: [
                    prev.ga_config.take_profit_range[0],
                    value / 100,
                  ],
                },
              }))
            }
            min={0.5}
            max={20}
            step={0.1}
            required
          />
        </div>
      </div>

      <ApiButton onClick={handleSubmit} loading={isLoading}>
        GA戦略を生成
      </ApiButton>
    </form>
  );
};

export default GAConfigForm;
