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
      start_date: "2020-01-01",
      end_date: "2020-12-31",
      initial_capital: 100000,
      commission_rate: 0.00055,
      strategy_config: {
        strategy_type: "",
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
        // 従来のリスク管理パラメータ（Position Sizingシステムにより廃止予定）
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

    // 従来の取引量範囲バリデーションは削除（Position Sizingシステムにより不要）

    // TP/SL設定はGAが自動最適化するため、バリデーション不要
    // 従来のTP/SL範囲バリデーション（後方互換性のため保持）
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

      {/* TP/SL & ポジションサイジング自動最適化の説明 */}
      <div className="space-y-4 p-4 border border-blue-600 rounded-lg bg-blue-900/20">
        <h3 className="text-lg font-semibold text-blue-300">
          🤖 リスク管理自動最適化
        </h3>
        <p className="text-sm text-blue-200">
          TP/SL設定とポジションサイズは、テクニカル指標と同様にGAが自動で最適化します。
          <strong className="text-blue-100">従来のイグジット条件は自動的に無効化され、TP/SL機能が優先されます。</strong>
          手動設定は不要です。
        </p>

        {/* TP/SL自動最適化 */}
        <div className="p-3 bg-pink-900/30 border border-pink-500/30 rounded-md">
          <h4 className="font-medium text-pink-300 mb-2">📈 TP/SL自動最適化</h4>
          <div className="text-xs text-pink-200 space-y-1">
            <div>
              • <strong>TP/SL決定方式</strong>:
              固定値、リスクリワード比、ボラティリティベースなど
            </div>
            <div>
              • <strong>リスクリワード比</strong>: 1:1.2 ～ 1:4.0の範囲
            </div>
            <div>
              • <strong>具体的なパーセンテージ</strong>: SL: 1%-8%, TP: 2%-20%
            </div>
          </div>
        </div>

        {/* ポジションサイジング自動最適化 */}
        <div className="p-3 bg-emerald-900/30 border border-emerald-500/30 rounded-md">
          <h4 className="font-medium text-emerald-300 mb-2">
            � ポジションサイジング自動最適化
          </h4>
          <div className="text-xs text-emerald-200 space-y-1">
            <div>
              • <strong>ハーフオプティマルF</strong>:
              過去データ分析によるリスク最適化
            </div>
            <div>
              • <strong>ボラティリティベース</strong>: ATRを使用したリスク調整
            </div>
            <div>
              • <strong>固定比率</strong>: 口座残高に対する固定比率
            </div>
            <div>
              • <strong>固定枚数</strong>: 設定された固定枚数
            </div>
          </div>
        </div>

        {/* 高度な設定 */}
        <div className="p-3 bg-purple-900/30 border border-purple-500/30 rounded-md">
          <h4 className="font-medium text-purple-300 mb-3">
            🧬 高度なGA設定
          </h4>

          {/* フィットネス共有 */}
          <div className="mb-4">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={config.ga_config.enable_fitness_sharing ?? true}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    ga_config: {
                      ...config.ga_config,
                      enable_fitness_sharing: e.target.checked,
                    },
                  })
                }
                className="rounded border-purple-500 text-purple-600 focus:ring-purple-500"
              />
              <span className="text-sm text-purple-200">
                フィットネス共有を有効化（戦略の多様性向上）
              </span>
            </label>
          </div>

          {/* ショートバイアス突然変異 */}
          <div className="mb-4">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={config.ga_config.enable_short_bias_mutation ?? true}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    ga_config: {
                      ...config.ga_config,
                      enable_short_bias_mutation: e.target.checked,
                    },
                  })
                }
                className="rounded border-purple-500 text-purple-600 focus:ring-purple-500"
              />
              <span className="text-sm text-purple-200">
                ショートバイアス突然変異を有効化（ショート戦略強化）
              </span>
            </label>
          </div>

          {/* ショートバイアス率 */}
          {config.ga_config.enable_short_bias_mutation && (
            <div className="mb-4">
              <InputField
                label="ショートバイアス適用率"
                type="number"
                value={config.ga_config.short_bias_rate ?? 0.3}
                onChange={(value) =>
                  setConfig({
                    ...config,
                    ga_config: {
                      ...config.ga_config,
                      short_bias_rate: parseFloat(value) || 0.3,
                    },
                  })
                }
                min={0}
                max={1}
                step={0.1}
                className="text-sm"
              />
            </div>
          )}

          <div className="text-xs text-purple-200 space-y-1">
            <div>
              • <strong>フィットネス共有</strong>: 類似戦略の評価を下げ、多様な戦略を促進
            </div>
            <div>
              • <strong>ショートバイアス突然変異</strong>: ショート戦略の生成を強化
            </div>
          </div>
        </div>
      </div>

      <ApiButton onClick={handleSubmit} loading={isLoading}>
        GA戦略を生成
      </ApiButton>
    </form>
  );
};

export default GAConfigForm;
