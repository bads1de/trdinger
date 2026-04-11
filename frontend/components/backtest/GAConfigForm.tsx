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
import {
  EarlyTerminationSettingsConfig,
  GAConfig as GAConfigType,
  GAEvaluationConfig,
  GAHybridConfig,
  FitnessSharingConfig,
} from "@/types/optimization";
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
    const initialEvaluationConfig = (initialGAConfig.evaluation_config ??
      {}) as GAEvaluationConfig;
    const initialEarlyTerminationSettings = (initialEvaluationConfig.early_termination_settings ??
      {}) as EarlyTerminationSettingsConfig;
    const initialFitnessSharing = (initialGAConfig.fitness_sharing ??
      {}) as FitnessSharingConfig;
    const initialHybridConfig = (initialGAConfig.hybrid_config ??
      {}) as GAHybridConfig;
    const initialFitnessWeights = {
      ...(initialGAConfig.fitness_weights || {}),
    };
    delete initialFitnessWeights.prediction_score;
    const defaultFitnessWeights = {
      total_return: 0.3,
      sharpe_ratio: 0.4,
      max_drawdown: 0.2,
      win_rate: 0.1,
      ...initialFitnessWeights,
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
        fitness_sharing: {
          ...initialFitnessSharing,
          enable_fitness_sharing:
            initialFitnessSharing.enable_fitness_sharing ?? true,
        },
        enable_multi_objective: initialGAConfig.enable_multi_objective ?? false,
        objectives: initialGAConfig.objectives ?? ["total_return"],
        objective_weights: initialGAConfig.objective_weights ?? [1.0],
        dynamic_objective_reweighting:
          initialGAConfig.dynamic_objective_reweighting ?? false,
        hybrid_config: {
          ...initialHybridConfig,
          mode: initialHybridConfig.mode ?? false,
          model_type: initialHybridConfig.model_type ?? "lightgbm",
          model_types: initialHybridConfig.model_types,
          volatility_gate_enabled:
            initialHybridConfig.volatility_gate_enabled ?? false,
          volatility_model_path: initialHybridConfig.volatility_model_path,
          ml_filter_enabled: initialHybridConfig.ml_filter_enabled ?? false,
          ml_model_path: initialHybridConfig.ml_model_path,
          preprocess_features: initialHybridConfig.preprocess_features ?? true,
        },
        evaluation_config: {
          ...initialEvaluationConfig,
          enable_parallel: initialEvaluationConfig.enable_parallel ?? true,
          max_workers: initialEvaluationConfig.max_workers ?? 4,
          timeout: initialEvaluationConfig.timeout ?? 300,
          enable_multi_fidelity_evaluation:
            initialEvaluationConfig.enable_multi_fidelity_evaluation ?? false,
          multi_fidelity_window_ratio:
            initialEvaluationConfig.multi_fidelity_window_ratio ?? 0.3,
          multi_fidelity_oos_ratio:
            initialEvaluationConfig.multi_fidelity_oos_ratio ?? 0.2,
          multi_fidelity_candidate_ratio:
            initialEvaluationConfig.multi_fidelity_candidate_ratio ?? 0.25,
          multi_fidelity_min_candidates:
            initialEvaluationConfig.multi_fidelity_min_candidates ?? 3,
          early_termination_settings: {
            ...initialEarlyTerminationSettings,
            enabled: initialEarlyTerminationSettings.enabled ?? false,
            max_drawdown:
              initialEarlyTerminationSettings.max_drawdown === undefined
                ? null
                : initialEarlyTerminationSettings.max_drawdown,
            min_trades:
              initialEarlyTerminationSettings.min_trades === undefined
                ? null
                : initialEarlyTerminationSettings.min_trades,
            min_trade_check_progress:
              initialEarlyTerminationSettings.min_trade_check_progress ?? 0.5,
            trade_pace_tolerance:
              initialEarlyTerminationSettings.trade_pace_tolerance ?? 0.5,
            min_expectancy:
              initialEarlyTerminationSettings.min_expectancy === undefined
                ? null
                : initialEarlyTerminationSettings.min_expectancy,
            expectancy_min_trades:
              initialEarlyTerminationSettings.expectancy_min_trades ?? 5,
            expectancy_progress:
              initialEarlyTerminationSettings.expectancy_progress ?? 0.6,
          },
          oos_split_ratio: initialEvaluationConfig.oos_split_ratio ?? 0.0,
          oos_fitness_weight: initialEvaluationConfig.oos_fitness_weight ?? 0.5,
          enable_walk_forward:
            initialEvaluationConfig.enable_walk_forward ?? false,
          wfa_n_folds: initialEvaluationConfig.wfa_n_folds ?? 5,
          wfa_train_ratio: initialEvaluationConfig.wfa_train_ratio ?? 0.7,
          wfa_anchored: initialEvaluationConfig.wfa_anchored ?? false,
        },

        // 制限設定
        min_indicators: initialGAConfig.min_indicators,
        min_conditions: initialGAConfig.min_conditions,
        max_conditions: initialGAConfig.max_conditions,

        // ペナルティ設定
        zero_trades_penalty: initialGAConfig.zero_trades_penalty,
        constraint_violation_penalty:
          initialGAConfig.constraint_violation_penalty,

        // TPSL設定
        tpsl_method_constraints: initialGAConfig.tpsl_method_constraints,
        tpsl_sl_range: initialGAConfig.tpsl_sl_range,
        tpsl_tp_range: initialGAConfig.tpsl_tp_range,
        tpsl_rr_range: initialGAConfig.tpsl_rr_range,
        tpsl_atr_multiplier_range: initialGAConfig.tpsl_atr_multiplier_range,

        // MTF
        enable_multi_timeframe: initialGAConfig.enable_multi_timeframe ?? false,
        available_timeframes: initialGAConfig.available_timeframes,
        mtf_indicator_probability:
          initialGAConfig.mtf_indicator_probability ?? 0.3,

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
    updates: Partial<GAConfigType["ga_config"]>,
  ) => {
    setConfig((prev) => ({
      ...prev,
      ga_config: { ...prev.ga_config, ...updates },
    }));
  };

  const handleEvaluationConfigChange = (
    updates: Partial<GAEvaluationConfig>,
  ) => {
    setConfig((prev) => ({
      ...prev,
      ga_config: {
        ...prev.ga_config,
        evaluation_config: {
          ...(prev.ga_config.evaluation_config ?? {}),
          ...updates,
        },
      },
    }));
  };

  const handleFitnessSharingChange = (
    updates: Partial<FitnessSharingConfig>,
  ) => {
    setConfig((prev) => ({
      ...prev,
      ga_config: {
        ...prev.ga_config,
        fitness_sharing: {
          ...(prev.ga_config.fitness_sharing ?? {}),
          ...updates,
        },
      },
    }));
  };

  const handleHybridConfigChange = (updates: Partial<GAHybridConfig>) => {
    setConfig((prev) => ({
      ...prev,
      ga_config: {
        ...prev.ga_config,
        hybrid_config: {
          ...(prev.ga_config.hybrid_config ?? {}),
          ...updates,
        },
      },
    }));
  };

  const evaluationConfig: GAEvaluationConfig =
    config.ga_config.evaluation_config ?? {};
  const earlyTerminationSettings: EarlyTerminationSettingsConfig =
    evaluationConfig.early_termination_settings ?? {};
  const fitnessSharingConfig: FitnessSharingConfig =
    config.ga_config.fitness_sharing ?? {};
  const hybridConfig: GAHybridConfig = config.ga_config.hybrid_config ?? {};

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
                checked={fitnessSharingConfig.enable_fitness_sharing ?? true}
                onChange={(e) =>
                  handleFitnessSharingChange({
                    enable_fitness_sharing: e.target.checked,
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
              value={evaluationConfig.oos_split_ratio ?? 0.0}
              onChange={(val) =>
                handleEvaluationConfigChange({ oos_split_ratio: val })
              }
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
                  checked={evaluationConfig.enable_walk_forward ?? false}
                  onChange={(e) =>
                    handleEvaluationConfigChange({
                      enable_walk_forward: e.target.checked,
                    })
                  }
                  className="rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
                />
                <span className="text-sm text-indigo-200">
                  Walk-Forward Analysis (WFA)
                </span>
              </label>

              {evaluationConfig.enable_walk_forward && (
                <div className="pl-6">
                  <InputField
                    label="WFAフォールド数"
                    type="number"
                    value={evaluationConfig.wfa_n_folds ?? 5}
                    onChange={(val) =>
                      handleEvaluationConfigChange({ wfa_n_folds: val })
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
              checked={hybridConfig.mode ?? false}
              onChange={(e) =>
                handleHybridConfigChange({ mode: e.target.checked })
              }
              className="w-5 h-5 rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
              aria-label="ハイブリッドGA+MLモードを有効化"
            />
          </div>

          {hybridConfig.mode && (
            <>
              <SelectField
                label="MLモデル"
                value={hybridConfig.model_type || "lightgbm"}
                onChange={(value) =>
                  handleHybridConfigChange({ model_type: value })
                }
                options={[
                  { value: "lightgbm", label: "LightGBM" },
                  { value: "xgboost", label: "XGBoost" },
                ]}
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
              checked={evaluationConfig.enable_parallel ?? true}
              onChange={(e) =>
                handleEvaluationConfigChange({ enable_parallel: e.target.checked })
              }
              className="w-5 h-5 rounded border-cyan-500 text-cyan-600 focus:ring-cyan-500"
              aria-label="並列評価を有効化"
            />
          </div>

          {evaluationConfig.enable_parallel && (
            <>
              <InputField
                label="最大ワーカー数"
                type="number"
                value={evaluationConfig.max_workers ?? ""}
                onChange={(value) =>
                  handleEvaluationConfigChange({ max_workers: value || null })
                }
                min={1}
                max={32}
                step={1}
                description={GA_INFO_MESSAGES.max_workers}
                placeholder="自動（CPUコア数×2）"
              />
              <InputField
                label="評価タイムアウト（秒）"
                type="number"
                value={evaluationConfig.timeout ?? 300}
                onChange={(value) =>
                  handleEvaluationConfigChange({ timeout: value })
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

        <div className="p-4 bg-emerald-900/30 border border-emerald-500/30 rounded-lg space-y-3">
          <h4 className="text-sm font-medium text-emerald-200">🚀 高速化</h4>

          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={evaluationConfig.enable_multi_fidelity_evaluation ?? false}
              onChange={(e) =>
                handleEvaluationConfigChange({
                  enable_multi_fidelity_evaluation: e.target.checked,
                })
              }
              className="rounded border-emerald-500 text-emerald-600 focus:ring-emerald-500"
            />
            <span className="text-sm text-emerald-200">
              multi-fidelity 評価
            </span>
          </label>

          {evaluationConfig.enable_multi_fidelity_evaluation && (
            <>
              <InputField
                label="coarse 期間比率"
                type="number"
                value={evaluationConfig.multi_fidelity_window_ratio ?? 0.3}
                onChange={(value) =>
                  handleEvaluationConfigChange({
                    multi_fidelity_window_ratio: value,
                  })
                }
                min={0.05}
                max={1}
                step={0.05}
                description="最初の粗評価で使う直近データの割合"
              />
              <InputField
                label="coarse OOS 比率"
                type="number"
                value={evaluationConfig.multi_fidelity_oos_ratio ?? 0.2}
                onChange={(value) =>
                  handleEvaluationConfigChange({
                    multi_fidelity_oos_ratio: value,
                  })
                }
                min={0.05}
                max={0.9}
                step={0.05}
                description="粗評価時だけ使う OOS 分割比率"
              />
              <InputField
                label="full 評価候補比率"
                type="number"
                value={evaluationConfig.multi_fidelity_candidate_ratio ?? 0.25}
                onChange={(value) =>
                  handleEvaluationConfigChange({
                    multi_fidelity_candidate_ratio: value,
                  })
                }
                min={0.05}
                max={1}
                step={0.05}
                description="coarse 後に full 評価へ昇格させる候補の割合"
              />
              <InputField
                label="full 評価最小候補数"
                type="number"
                value={evaluationConfig.multi_fidelity_min_candidates ?? 3}
                onChange={(value) =>
                  handleEvaluationConfigChange({
                    multi_fidelity_min_candidates: value,
                  })
                }
                min={1}
                max={20}
                step={1}
                description="候補比率が小さくても最低限 full 評価する件数"
              />
            </>
          )}

          <label className="flex items-center space-x-2 pt-2">
            <input
              type="checkbox"
              checked={earlyTerminationSettings.enabled ?? false}
              onChange={(e) =>
                handleEvaluationConfigChange({
                  early_termination_settings: {
                    ...earlyTerminationSettings,
                    enabled: e.target.checked,
                  },
                })
              }
              className="rounded border-emerald-500 text-emerald-600 focus:ring-emerald-500"
            />
            <span className="text-sm text-emerald-200">
              早期打ち切り
            </span>
          </label>

          {earlyTerminationSettings.enabled && (
            <>
              <InputField
                label="最大DD打ち切り"
                type="number"
                value={earlyTerminationSettings.max_drawdown}
                onChange={(value) =>
                  handleEvaluationConfigChange({
                    early_termination_settings: {
                      ...earlyTerminationSettings,
                      max_drawdown: value,
                    },
                  })
                }
                allowEmptyNumber
                min={0.01}
                max={1}
                step={0.01}
                description="この DD を超えたら即終了"
              />
              <InputField
                label="最低トレード数"
                type="number"
                value={earlyTerminationSettings.min_trades}
                onChange={(value) =>
                  handleEvaluationConfigChange({
                    early_termination_settings: {
                      ...earlyTerminationSettings,
                      min_trades: value,
                    },
                  })
                }
                allowEmptyNumber
                min={1}
                max={1000}
                step={1}
                description="このペースを大きく下回る候補を早めに落とします"
              />
              <InputField
                label="トレード数判定開始進捗"
                type="number"
                value={earlyTerminationSettings.min_trade_check_progress ?? 0.5}
                onChange={(value) =>
                  handleEvaluationConfigChange({
                    early_termination_settings: {
                      ...earlyTerminationSettings,
                      min_trade_check_progress: value,
                    },
                  })
                }
                min={0.05}
                max={1}
                step={0.05}
                description="この進捗以降でトレード数不足を見始めます"
              />
              <InputField
                label="トレードペース許容係数"
                type="number"
                value={earlyTerminationSettings.trade_pace_tolerance ?? 0.5}
                onChange={(value) =>
                  handleEvaluationConfigChange({
                    early_termination_settings: {
                      ...earlyTerminationSettings,
                      trade_pace_tolerance: value,
                    },
                  })
                }
                min={0.1}
                max={1}
                step={0.05}
                description="目標ペースに対してどこまで不足を許すか"
              />
              <InputField
                label="最低期待値"
                type="number"
                value={earlyTerminationSettings.min_expectancy}
                onChange={(value) =>
                  handleEvaluationConfigChange({
                    early_termination_settings: {
                      ...earlyTerminationSettings,
                      min_expectancy: value,
                    },
                  })
                }
                allowEmptyNumber
                min={-1}
                max={1}
                step={0.01}
                description="期待値がこの値を下回る候補を打ち切ります"
              />
              <InputField
                label="期待値判定最小トレード数"
                type="number"
                value={earlyTerminationSettings.expectancy_min_trades ?? 5}
                onChange={(value) =>
                  handleEvaluationConfigChange({
                    early_termination_settings: {
                      ...earlyTerminationSettings,
                      expectancy_min_trades: value,
                    },
                  })
                }
                min={1}
                max={100}
                step={1}
                description="期待値判定に必要なクローズ済みトレード数"
              />
              <InputField
                label="期待値判定開始進捗"
                type="number"
                value={earlyTerminationSettings.expectancy_progress ?? 0.6}
                onChange={(value) =>
                  handleEvaluationConfigChange({
                    early_termination_settings: {
                      ...earlyTerminationSettings,
                      expectancy_progress: value,
                    },
                  })
                }
                min={0.05}
                max={1}
                step={0.05}
                description="この進捗以降で期待値不足を見始めます"
              />
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
