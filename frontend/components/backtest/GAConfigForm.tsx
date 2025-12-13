/**
 * GA設定フォームコンポーネント
 *
 * 遺伝的アルゴリズムによる自動戦略生成の設定を行います。
 */

"use client";

import React, { useState, useEffect } from "react";
import { InputField } from "@/components/common/InputField";
import { SelectField } from "@/components/common/SelectField";
import ActionButton from "@/components/common/ActionButton";
import ApiButton from "@/components/button/ApiButton";
import { GAConfig as GAConfigType } from "@/types/optimization";
import { BacktestConfig as BacktestConfigType } from "@/types/backtest";
import { BaseBacktestConfigForm } from "./BaseBacktestConfigForm";
import { GA_INFO_MESSAGES } from "@/constants/info";
import { ObjectiveSelection } from "./optimization/ObjectiveSelection";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronDown } from "lucide-react";

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
  // Collapsibleの開閉状態（デフォルトは閉じている）
  const [isOpen, setIsOpen] = useState<boolean>(false);

  const [config, setConfig] = useState<GAConfigType>(() => {
    const baseBacktestConfig: BacktestConfigType = {
      strategy_name: "GA_STRATEGY",
      symbol: "BTC/USDT:USDT",
      timeframe: "1h",
      start_date: "2025-01-01",
      end_date: "2025-03-01",
      initial_capital: 1000000,
      commission_rate: 0.00055,
      strategy_config: {
        strategy_type: "",
        parameters: {},
      },
    };

    const effectiveBaseConfig = currentBacktestConfig || baseBacktestConfig;
    const initialGAConfig = (initialConfig.ga_config ?? {}) as any;
    const defaultFitnessWeights = {
      total_return: 0.3,
      sharpe_ratio: 0.4,
      max_drawdown: 0.2,
      win_rate: 0.1,
      ...(initialGAConfig.fitness_weights || {}),
    };
    const defaultFitnessConstraints = {
      min_trades: 10,
      max_drawdown_limit: 0.3,
      min_sharpe_ratio: 0.5,
      ...(initialGAConfig.fitness_constraints || {}),
    };

    const defaultExperimentName =
      initialConfig.experiment_name ??
      `GA_${new Date()
        .toISOString()
        .slice(0, 10)}_${effectiveBaseConfig.symbol.replace("/", "_")}`;

    return {
      experiment_name: defaultExperimentName,
      base_config: initialConfig.base_config ?? effectiveBaseConfig,
      ga_config: {
        population_size: initialGAConfig.population_size ?? 10,
        generations: initialGAConfig.generations ?? 10,
        mutation_rate: initialGAConfig.mutation_rate ?? 0.1,
        crossover_rate: initialGAConfig.crossover_rate ?? 0.8,
        elite_size: initialGAConfig.elite_size ?? 5,
        max_indicators: initialGAConfig.max_indicators ?? 5,
        fitness_weights: defaultFitnessWeights,
        fitness_constraints: defaultFitnessConstraints,
        enable_multi_objective: initialGAConfig.enable_multi_objective ?? true,
        objectives: initialGAConfig.objectives ?? ["win_rate", "max_drawdown"],
        objective_weights: initialGAConfig.objective_weights ?? [1.0, -1.0],
        dynamic_objective_reweighting:
          initialGAConfig.dynamic_objective_reweighting ?? false,
        hybrid_mode: initialGAConfig.hybrid_mode ?? false,
        hybrid_model_type: initialGAConfig.hybrid_model_type ?? "lightgbm",
        hybrid_model_types: initialGAConfig.hybrid_model_types,
        // 並列評価設定
        enable_parallel_evaluation:
          initialGAConfig.enable_parallel_evaluation ?? true,
        max_evaluation_workers: initialGAConfig.max_evaluation_workers ?? null,
        evaluation_timeout: initialGAConfig.evaluation_timeout ?? 300,

        // 制限設定
        min_indicators: initialGAConfig.min_indicators,
        min_conditions: initialGAConfig.min_conditions,
        max_conditions: initialGAConfig.max_conditions,

        // ペナルティ設定
        zero_trades_penalty: initialGAConfig.zero_trades_penalty,
        constraint_violation_penalty:
          initialGAConfig.constraint_violation_penalty,

        // 高度な設定 (Fitness Sharing)
        enable_fitness_sharing: initialGAConfig.enable_fitness_sharing ?? true,
        sharing_radius: initialGAConfig.sharing_radius,
        sharing_alpha: initialGAConfig.sharing_alpha,
        sampling_threshold: initialGAConfig.sampling_threshold,
        sampling_ratio: initialGAConfig.sampling_ratio,

        // OOS
        oos_split_ratio: initialGAConfig.oos_split_ratio,
        oos_fitness_weight: initialGAConfig.oos_fitness_weight,

        // TPSL設定
        tpsl_method_constraints: initialGAConfig.tpsl_method_constraints,
        tpsl_sl_range: initialGAConfig.tpsl_sl_range,
        tpsl_tp_range: initialGAConfig.tpsl_tp_range,
        tpsl_rr_range: initialGAConfig.tpsl_rr_range,
        tpsl_atr_multiplier_range: initialGAConfig.tpsl_atr_multiplier_range,

        // WFA
        enable_walk_forward: initialGAConfig.enable_walk_forward ?? false,
        wfa_n_folds: initialGAConfig.wfa_n_folds ?? 5,
        wfa_train_ratio: initialGAConfig.wfa_train_ratio ?? 0.7,
        wfa_anchored: initialGAConfig.wfa_anchored ?? false,

        // MTF
        enable_multi_timeframe: initialGAConfig.enable_multi_timeframe ?? false,
        available_timeframes: initialGAConfig.available_timeframes,
        mtf_indicator_probability:
          initialGAConfig.mtf_indicator_probability ?? 0.3,

        // ML Filter
        ml_filter_enabled: initialGAConfig.ml_filter_enabled ?? false,
        ml_model_path: initialGAConfig.ml_model_path,
        preprocess_features: initialGAConfig.preprocess_features ?? true,

        // Data Weights
        price_data_weight: initialGAConfig.price_data_weight,
        volume_data_weight: initialGAConfig.volume_data_weight,
        oi_fr_data_weight: initialGAConfig.oi_fr_data_weight,

        // Advanced Genetic Operators
        crossover_field_selection_probability:
          initialGAConfig.crossover_field_selection_probability,
        indicator_param_mutation_range:
          initialGAConfig.indicator_param_mutation_range,
        risk_param_mutation_range: initialGAConfig.risk_param_mutation_range,
        indicator_add_delete_probability:
          initialGAConfig.indicator_add_delete_probability,
        indicator_add_vs_delete_probability:
          initialGAConfig.indicator_add_vs_delete_probability,
        condition_change_probability_multiplier:
          initialGAConfig.condition_change_probability_multiplier,
        condition_selection_probability:
          initialGAConfig.condition_selection_probability,
        condition_operator_switch_probability:
          initialGAConfig.condition_operator_switch_probability,
        tpsl_gene_creation_probability_multiplier:
          initialGAConfig.tpsl_gene_creation_probability_multiplier,
        position_sizing_gene_creation_probability_multiplier:
          initialGAConfig.position_sizing_gene_creation_probability_multiplier,
        numeric_threshold_probability:
          initialGAConfig.numeric_threshold_probability,

        // パラメータ範囲プリセット
        parameter_range_preset: initialGAConfig.parameter_range_preset,
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

  const handleGAConfigChange = (
    updates: Partial<GAConfigType["ga_config"]>
  ) => {
    setConfig((prev) => ({
      ...prev,
      ga_config: { ...prev.ga_config, ...updates },
    }));
  };

  const handleSubmit = () => {
    onSubmit(config);
  };

  return (
    <div className="flex flex-col lg:flex-row min-h-0">
      {/* Left Column: Main Settings */}
      <div className="flex-1 p-6 space-y-6 overflow-y-auto">
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
      </div>

      {/* Right Column: Advanced GA Settings */}
      <div className="flex-1 p-6 space-y-4 bg-secondary-900 border-l border-secondary-700 overflow-y-auto">
        <h3 className="text-lg font-semibold text-secondary-100 mb-3">
          🧬 GA詳細設定
        </h3>

        {/* 自動最適化設定説明（Collapsible） */}
        <Collapsible open={isOpen} onOpenChange={setIsOpen}>
          <CollapsibleTrigger className="w-full">
            <div className="flex items-center justify-between p-3 bg-secondary-800/50 border border-secondary-600/30 rounded-md hover:bg-secondary-700/50 transition-colors">
              <h4 className="font-medium text-secondary-200">
                📋 自動最適化設定説明
              </h4>
              <ChevronDown
                className={`w-5 h-5 text-secondary-400 transition-transform duration-200 ${
                  isOpen ? "rotate-180" : ""
                }`}
              />
            </div>
          </CollapsibleTrigger>

          <CollapsibleContent className="space-y-4 mt-4">
            {/* 指標モードの説明 */}
            <div className="p-3 bg-purple-900/30 border border-purple-500/30 rounded-md">
              <h4 className="font-medium text-purple-300 mb-2">
                📊 指標モード選択
              </h4>
              <div className="text-sm text-purple-200 space-y-1">
                <div>
                  <strong className="text-purple-100">TA:</strong>{" "}
                  従来のテクニカル指標のみを使用
                </div>
                <div>
                  <strong className="text-purple-100">ML:</strong>{" "}
                  ML予測指標のみを使用
                </div>
                <div>
                  <strong className="text-purple-100">混合:</strong>{" "}
                  テクニカル指標とML予測指標を組み合わせ
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
              <h4 className="font-medium text-pink-300 mb-2">
                📈 TP/SL自動最適化
              </h4>
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
          </CollapsibleContent>
        </Collapsible>

        {/* 高度なGA設定 */}
        <div className="p-3 bg-indigo-900/30 border border-indigo-500/30 rounded-md space-y-4">
          <h4 className="font-medium text-indigo-300">⚙️ 高度なGA設定</h4>

          {/* 多様性維持・動的制御 */}
          <div className="space-y-2">
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
            <div>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={
                    config.ga_config.dynamic_objective_reweighting ?? false
                  }
                  onChange={(e) =>
                    handleGAConfigChange({
                      dynamic_objective_reweighting: e.target.checked,
                    })
                  }
                  className="rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
                />
                <span className="text-sm text-indigo-200">
                  動的重み付け (レジーム適応)
                </span>
              </label>
            </div>
          </div>

          <div className="border-t border-indigo-500/30 pt-3">
            <h5 className="text-sm font-medium text-indigo-200 mb-2">
              🛡️ 過学習対策
            </h5>

            {/* OOS設定 */}
            <InputField
              label="Out-of-Sample 分割比率"
              type="number"
              value={config.ga_config.oos_split_ratio ?? 0.0}
              onChange={(val) => handleGAConfigChange({ oos_split_ratio: val })}
              min={0}
              max={0.5}
              step={0.05}
              description="検証用データの割合 (0.0-0.5)"
            />

            {/* WFA設定 */}
            <div className="mt-3 space-y-2">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.ga_config.enable_walk_forward ?? false}
                  onChange={(e) =>
                    handleGAConfigChange({
                      enable_walk_forward: e.target.checked,
                    })
                  }
                  className="rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
                />
                <span className="text-sm text-indigo-200">
                  Walk-Forward Analysis (WFA)
                </span>
              </label>

              {config.ga_config.enable_walk_forward && (
                <div className="pl-6">
                  <InputField
                    label="WFAフォールド数"
                    type="number"
                    value={config.ga_config.wfa_n_folds ?? 5}
                    onChange={(val) =>
                      handleGAConfigChange({ wfa_n_folds: val })
                    }
                    min={2}
                    max={10}
                    step={1}
                    description="期間分割数"
                  />
                </div>
              )}
            </div>
          </div>

          {/* 多目的最適化設定 */}
          <ObjectiveSelection
            gaConfig={config.ga_config}
            onGAConfigChange={handleGAConfigChange}
          />
        </div>

        {/* ハイブリッドGA+MLモード設定 */}
        <div className="p-4 bg-indigo-900/30 border border-indigo-500/30 rounded-lg space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-indigo-200">
              🔬 ハイブリッドGA+MLモード
            </label>
            <input
              type="checkbox"
              checked={config.ga_config.hybrid_mode || false}
              onChange={(e) =>
                handleGAConfigChange({ hybrid_mode: e.target.checked })
              }
              className="w-5 h-5 rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
              aria-label="ハイブリッドGA+MLモードを有効化"
            />
          </div>

          {config.ga_config.hybrid_mode && (
            <>
              <SelectField
                label="MLモデル"
                value={config.ga_config.hybrid_model_type || "lightgbm"}
                onChange={(value) =>
                  handleGAConfigChange({ hybrid_model_type: value })
                }
                options={[
                  { value: "lightgbm", label: "LightGBM" },
                  { value: "xgboost", label: "XGBoost" },
                ]}
              />
              <InputField
                label="ML予測重み (prediction_score)"
                type="number"
                value={config.ga_config.fitness_weights.prediction_score || 0.1}
                onChange={(value) =>
                  handleGAConfigChange({
                    fitness_weights: {
                      ...config.ga_config.fitness_weights,
                      prediction_score: value,
                    },
                  })
                }
                min={0}
                max={1}
                step={0.05}
                description="ML予測スコアの重み（0-1）"
              />
              <p className="text-xs text-indigo-300">
                💡
                事前にMLモデルを学習しておく必要があります。未学習の場合はデフォルト予測を使用します。
              </p>
            </>
          )}
        </div>

        {/* 並列評価設定 */}
        <div className="p-4 bg-cyan-900/30 border border-cyan-500/30 rounded-lg space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-cyan-200">
              ⚡ 並列評価
            </label>
            <input
              type="checkbox"
              checked={config.ga_config.enable_parallel_evaluation ?? true}
              onChange={(e) =>
                handleGAConfigChange({
                  enable_parallel_evaluation: e.target.checked,
                })
              }
              className="w-5 h-5 rounded border-cyan-500 text-cyan-600 focus:ring-cyan-500"
              aria-label="並列評価を有効化"
            />
          </div>

          {config.ga_config.enable_parallel_evaluation && (
            <>
              <InputField
                label="最大ワーカー数"
                type="number"
                value={config.ga_config.max_evaluation_workers ?? ""}
                onChange={(value) =>
                  handleGAConfigChange({
                    max_evaluation_workers: value || null,
                  })
                }
                min={1}
                max={32}
                step={1}
                description={GA_INFO_MESSAGES.max_evaluation_workers}
                placeholder="自動（CPUコア数×2）"
              />
              <InputField
                label="評価タイムアウト（秒）"
                type="number"
                value={config.ga_config.evaluation_timeout ?? 300}
                onChange={(value) =>
                  handleGAConfigChange({ evaluation_timeout: value })
                }
                min={30}
                max={1800}
                step={30}
                description={GA_INFO_MESSAGES.evaluation_timeout}
              />
              <p className="text-xs text-cyan-300">
                💡
                並列評価は大規模な個体群（50以上）で効果的です。ワーカー数を増やすとメモリ使用量も増加します。
              </p>
            </>
          )}
        </div>

        {/* Action Buttons */}
        <div className="pt-6 flex justify-end items-center space-x-4 border-t border-secondary-700 mt-6">
          <ActionButton
            type="button"
            onClick={onClose}
            disabled={isLoading}
            variant="secondary"
          >
            キャンセル
          </ActionButton>
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
