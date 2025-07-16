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
import { GA_INFO_MESSAGES } from "@/constants/info";
import {
  AVAILABLE_OBJECTIVES,
  ObjectiveDefinition,
} from "@/types/optimization";

// 指標モードの選択肢
const INDICATOR_MODE_OPTIONS = [
  { value: "mixed", label: "混合 (テクニカル + ML)" },
  { value: "technical_only", label: "テクニカルオンリー" },
  { value: "ml_only", label: "MLオンリー" },
];

interface GAConfigFormProps {
  onSubmit: (config: GAConfigType) => void;
  onClose?: () => void;
  initialConfig?: Partial<GAConfigType>;
  isLoading?: boolean;
  currentBacktestConfig?: BacktestConfigType | null;
}

const GAConfigForm: React.FC<GAConfigFormProps> = ({
  onSubmit,
  onClose,
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
        // 指標モード設定
        indicator_mode: initialConfig.ga_config?.indicator_mode || "mixed",
        // 多目的最適化設定
        enable_multi_objective:
          initialConfig.ga_config?.enable_multi_objective || false,
        objectives: initialConfig.ga_config?.objectives || ["total_return"],
        objective_weights: initialConfig.ga_config?.objective_weights || [1.0],
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

    return errors;
  };

  const handleObjectiveChange = (objectiveName: string, checked: boolean) => {
    const currentObjectives = config.ga_config.objectives || [];
    const currentWeights = config.ga_config.objective_weights || [];

    if (checked) {
      // 目的を追加
      const objective = AVAILABLE_OBJECTIVES.find(
        (obj) => obj.name === objectiveName
      );
      if (objective && !currentObjectives.includes(objectiveName)) {
        const newObjectives = [...currentObjectives, objectiveName];
        const newWeights = [...currentWeights, objective.weight];
        setConfig({
          ...config,
          ga_config: {
            ...config.ga_config,
            objectives: newObjectives,
            objective_weights: newWeights,
          },
        });
      }
    } else {
      // 目的を削除
      const index = currentObjectives.indexOf(objectiveName);
      if (index > -1) {
        const newObjectives = currentObjectives.filter((_, i) => i !== index);
        const newWeights = currentWeights.filter((_, i) => i !== index);
        setConfig({
          ...config,
          ga_config: {
            ...config.ga_config,
            objectives: newObjectives,
            objective_weights: newWeights,
          },
        });
      }
    }
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
    <div className="grid grid-cols-1 md:grid-cols-2 h-full">
      {/* Left Column: Main Settings */}
      <div className="p-6 space-y-6 overflow-y-auto max-h-[calc(90vh-80px)]">
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
          description={GA_INFO_MESSAGES.population_size}
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
          description={GA_INFO_MESSAGES.generations}
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
          description={GA_INFO_MESSAGES.mutation_rate}
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
          description={GA_INFO_MESSAGES.crossover_rate}
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
        <SelectField
          label="指標モード (indicator_mode)"
          value={config.ga_config.indicator_mode}
          onChange={(value) =>
            setConfig((prev) => ({
              ...prev,
              ga_config: {
                ...prev.ga_config,
                indicator_mode: value as "technical_only" | "ml_only" | "mixed",
              },
            }))
          }
          options={INDICATOR_MODE_OPTIONS}
          required
        />
      </div>

      {/* Right Column: Advanced GA Settings */}
      <div className="p-6 space-y-4 overflow-y-auto max-h-[calc(90vh-80px)] bg-secondary-900 border-l border-secondary-700">
        <h3 className="text-lg font-semibold text-secondary-100 mb-3">
          🧬 GA詳細設定
        </h3>

        {/* 指標モードの説明 */}
        <div className="p-3 bg-purple-900/30 border border-purple-500/30 rounded-md">
          <h4 className="font-medium text-purple-300 mb-2">
            📊 指標モード選択
          </h4>
          <div className="text-sm text-purple-200 space-y-1">
            <div>
              <strong className="text-purple-100">混合 (推奨):</strong>{" "}
              テクニカル指標とML予測指標の両方を使用
            </div>
            <div>
              <strong className="text-purple-100">テクニカルオンリー:</strong>{" "}
              従来のテクニカル指標のみを使用
            </div>
            <div>
              <strong className="text-purple-100">MLオンリー:</strong>{" "}
              ML予測指標のみを使用
            </div>
          </div>
        </div>

        {/* リスク管理自動最適化 */}
        <div className="p-3 bg-blue-900/30 border border-blue-500/30 rounded-md">
          <h4 className="font-medium text-blue-300 mb-2">
            🤖 リスク管理自動最適化
          </h4>
          <p className="text-sm text-blue-200">
            TP/SLとポジションサイズはGAが自動最適化します。
            <strong className="text-blue-100">
              手動でのイグジット条件は無視されます。
            </strong>
          </p>
        </div>

        {/* TP/SL自動最適化 */}
        <div className="p-3 bg-pink-900/30 border border-pink-500/30 rounded-md">
          <h4 className="font-medium text-pink-300 mb-2">📈 TP/SL自動最適化</h4>
          <div className="text-xs text-pink-200 space-y-1">
            <div>
              • <strong>決定方式</strong>:
              固定値、リスクリワード比、ボラティリティベース等
            </div>
            <div>
              • <strong>リスクリワード比</strong>: 1:1.2 ～ 1:4.0
            </div>
            <div>
              • <strong>パーセンテージ範囲</strong>: SL: 1%-8%, TP: 2%-20%
            </div>
          </div>
        </div>

        {/* ポジションサイジング自動最適化 */}
        <div className="p-3 bg-emerald-900/30 border border-emerald-500/30 rounded-md">
          <h4 className="font-medium text-emerald-300 mb-2">
            💰 ポジションサイジング自動最適化
          </h4>
          <div className="text-xs text-emerald-200 space-y-1">
            <div>
              • <strong>方式</strong>: ハーフオプティマルF,
              ボラティリティベース, 固定比率/枚数
            </div>
          </div>
        </div>

        {/* 高度なGA設定 */}
        <div className="p-3 bg-indigo-900/30 border border-indigo-500/30 rounded-md">
          <h4 className="font-medium text-indigo-300 mb-3">⚙️ 高度なGA設定</h4>

          {/* フィットネス共有 */}
          <div className="mb-2">
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
                className="rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
              />
              <span className="text-sm text-indigo-200">
                フィットネス共有 (戦略の多様性向上)
              </span>
            </label>
          </div>

          {/* ショートバイアス突然変異 */}
          <div className="mb-3">
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
                className="rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
              />
              <span className="text-sm text-indigo-200">
                ショートバイアス突然変異 (ショート戦略強化)
              </span>
            </label>
          </div>

          {/* ショートバイアス率 */}
          {config.ga_config.enable_short_bias_mutation && (
            <div className="mt-2 pl-6">
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

          {/* 多目的最適化設定 */}
          <div className="border-t border-indigo-500/20 pt-3 mt-3">
            <div className="mb-3">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.ga_config.enable_multi_objective ?? false}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      ga_config: {
                        ...config.ga_config,
                        enable_multi_objective: e.target.checked,
                        objectives: e.target.checked
                          ? ["total_return", "max_drawdown"]
                          : ["total_return"],
                        objective_weights: e.target.checked
                          ? [1.0, -1.0]
                          : [1.0],
                      },
                    })
                  }
                  className="rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
                />
                <span className="text-sm text-indigo-200">
                  多目的最適化を有効にする
                </span>
              </label>
              <div className="text-xs text-indigo-300/70 mt-1 ml-6">
                複数の目的（リターンとリスクなど）を同時に最適化します
              </div>
            </div>

            {/* 最適化目的選択 */}
            {config.ga_config.enable_multi_objective && (
              <div className="ml-6 space-y-3">
                <h5 className="text-sm font-medium text-indigo-200">
                  最適化目的
                </h5>

                <div className="grid grid-cols-2 gap-3">
                  {/* 総リターン */}
                  <div>
                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={
                          config.ga_config.objectives?.includes(
                            "total_return"
                          ) ?? false
                        }
                        onChange={(e) =>
                          handleObjectiveChange(
                            "total_return",
                            e.target.checked
                          )
                        }
                        className="rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
                      />
                      <span className="text-sm text-indigo-200">
                        総リターン
                      </span>
                      <span className="text-xs text-green-400">最大化</span>
                    </label>
                    <div className="text-xs text-indigo-300/60 ml-6">
                      投資期間全体での総リターン率
                    </div>
                  </div>

                  {/* シャープレシオ */}
                  <div>
                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={
                          config.ga_config.objectives?.includes(
                            "sharpe_ratio"
                          ) ?? false
                        }
                        onChange={(e) =>
                          handleObjectiveChange(
                            "sharpe_ratio",
                            e.target.checked
                          )
                        }
                        className="rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
                      />
                      <span className="text-sm text-indigo-200">
                        シャープレシオ
                      </span>
                      <span className="text-xs text-green-400">最大化</span>
                    </label>
                    <div className="text-xs text-indigo-300/60 ml-6">
                      リスク調整後リターンの指標
                    </div>
                  </div>

                  {/* 最大ドローダウン */}
                  <div>
                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={
                          config.ga_config.objectives?.includes(
                            "max_drawdown"
                          ) ?? false
                        }
                        onChange={(e) =>
                          handleObjectiveChange(
                            "max_drawdown",
                            e.target.checked
                          )
                        }
                        className="rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
                      />
                      <span className="text-sm text-indigo-200">
                        最大ドローダウン
                      </span>
                      <span className="text-xs text-red-400">最小化</span>
                    </label>
                    <div className="text-xs text-indigo-300/60 ml-6">
                      最大の資産減少分（最小化したい）
                    </div>
                  </div>

                  {/* 勝率 */}
                  <div>
                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={
                          config.ga_config.objectives?.includes("win_rate") ??
                          false
                        }
                        onChange={(e) =>
                          handleObjectiveChange("win_rate", e.target.checked)
                        }
                        className="rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
                      />
                      <span className="text-sm text-indigo-200">勝率</span>
                      <span className="text-xs text-green-400">最大化</span>
                    </label>
                    <div className="text-xs text-indigo-300/60 ml-6">
                      勝ちトレードの割合
                    </div>
                  </div>

                  {/* プロフィットファクター */}
                  <div>
                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={
                          config.ga_config.objectives?.includes(
                            "profit_factor"
                          ) ?? false
                        }
                        onChange={(e) =>
                          handleObjectiveChange(
                            "profit_factor",
                            e.target.checked
                          )
                        }
                        className="rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
                      />
                      <span className="text-sm text-indigo-200">
                        プロフィットファクター
                      </span>
                      <span className="text-xs text-green-400">最大化</span>
                    </label>
                    <div className="text-xs text-indigo-300/60 ml-6">
                      総利益と総損失の比率
                    </div>
                  </div>

                  {/* ソルティーノレシオ */}
                  <div>
                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={
                          config.ga_config.objectives?.includes(
                            "sortino_ratio"
                          ) ?? false
                        }
                        onChange={(e) =>
                          handleObjectiveChange(
                            "sortino_ratio",
                            e.target.checked
                          )
                        }
                        className="rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
                      />
                      <span className="text-sm text-indigo-200">
                        ソルティーノレシオ
                      </span>
                      <span className="text-xs text-green-400">最大化</span>
                    </label>
                    <div className="text-xs text-indigo-300/60 ml-6">
                      下方リスク調整後リターンの指標
                    </div>
                  </div>
                </div>

                {/* 選択された目的の表示 */}
                {config.ga_config.objectives &&
                  config.ga_config.objectives.length > 0 && (
                    <div className="mt-3 p-2 bg-indigo-800/30 rounded border border-indigo-500/30">
                      <h6 className="text-xs font-medium text-indigo-200 mb-2">
                        選択された目的 ({config.ga_config.objectives.length}個)
                      </h6>
                      <div className="space-y-1">
                        {config.ga_config.objectives.map((objective, index) => (
                          <div
                            key={objective}
                            className="flex justify-between items-center text-xs"
                          >
                            <span className="text-indigo-200">
                              {AVAILABLE_OBJECTIVES.find(
                                (obj) => obj.name === objective
                              )?.display_name || objective}
                            </span>
                            <span className="text-indigo-300">
                              重み:{" "}
                              {config.ga_config.objective_weights?.[
                                index
                              ]?.toFixed(1) || "1.0"}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
              </div>
            )}
          </div>
        </div>
        {/* Action Buttons */}
        <div className="pt-6 flex justify-end items-center space-x-4">
          <button
            type="button"
            onClick={onClose}
            disabled={isLoading}
            className="px-4 py-2 text-secondary-400 hover:text-secondary-200 transition-colors disabled:opacity-50"
          >
            キャンセル
          </button>
          <ApiButton onClick={handleSubmit} loading={isLoading}>
            {config.ga_config.enable_multi_objective
              ? "多目的GA戦略を生成"
              : "GA戦略を生成"}
          </ApiButton>
        </div>
      </div>
    </div>
  );
};

export default GAConfigForm;
