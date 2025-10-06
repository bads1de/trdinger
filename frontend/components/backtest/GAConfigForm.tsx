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

// 指標モードの選択肢
const INDICATOR_MODE_OPTIONS = [
  { value: "technical_only", label: "TA" },
  { value: "ml_only", label: "ML" },
  { value: "mixed", label: "混合" },
];

const DRL_POLICY_OPTIONS = [
  { value: "ppo", label: "PPO" },
  { value: "a2c", label: "A2C" },
  { value: "dqn", label: "DQN" },
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
    const initialGAConfig = initialConfig.ga_config ?? {};
    const defaultFitnessWeights = {
      total_return: 0.3,
      sharpe_ratio: 0.4,
      max_drawdown: 0.2,
      win_rate: 0.1,
      ...initialGAConfig.fitness_weights,
    };
    const defaultFitnessConstraints = {
      min_trades: 10,
      max_drawdown_limit: 0.3,
      min_sharpe_ratio: 0.5,
      ...initialGAConfig.fitness_constraints,
    };

    const defaultExperimentName =
      initialConfig.experiment_name ??
      `GA_${new Date().toISOString().slice(0, 10)}_${effectiveBaseConfig.symbol.replace("/", "_")}`;

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
        allowed_indicators: initialGAConfig.allowed_indicators ?? [],
        indicator_mode: initialGAConfig.indicator_mode ?? "technical_only",
        fitness_weights: defaultFitnessWeights,
        fitness_constraints: defaultFitnessConstraints,
        enable_multi_objective: initialGAConfig.enable_multi_objective ?? true,
        objectives: initialGAConfig.objectives ?? ["win_rate", "max_drawdown"],
        objective_weights: initialGAConfig.objective_weights ?? [1.0, -1.0],
        regime_adaptation_enabled:
          initialGAConfig.regime_adaptation_enabled ?? false,
        hybrid_mode: initialGAConfig.hybrid_mode ?? false,
        hybrid_model_type: initialGAConfig.hybrid_model_type ?? "lightgbm",
        hybrid_model_types: initialGAConfig.hybrid_model_types,
        hybrid_automl_config: initialGAConfig.hybrid_automl_config,
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

  const drlConfig = config.ga_config.hybrid_automl_config?.drl;
  const drlEnabled = Boolean(drlConfig?.enabled);
  const drlPolicyType = drlConfig?.policy_type ?? "ppo";
  const drlPolicyWeight = drlConfig?.policy_weight ?? 0.5;

  const handleDRLEnabledChange = (enabled: boolean) => {
    setConfig((prev) => {
      const currentAutoml = prev.ga_config.hybrid_automl_config ?? {};
      const currentDrl = currentAutoml.drl ?? {
        policy_type: "ppo",
        policy_weight: 0.5,
      };

      return {
        ...prev,
        ga_config: {
          ...prev.ga_config,
          hybrid_automl_config: {
            ...currentAutoml,
            drl: {
              ...currentDrl,
              enabled,
            },
          },
        },
      };
    });
  };

  const handleDRLPolicyTypeChange = (value: string) => {
    setConfig((prev) => {
      const currentAutoml = prev.ga_config.hybrid_automl_config ?? {};
      const currentDrl = currentAutoml.drl ?? {
        enabled: false,
        policy_weight: 0.5,
      };

      return {
        ...prev,
        ga_config: {
          ...prev.ga_config,
          hybrid_automl_config: {
            ...currentAutoml,
            drl: {
              ...currentDrl,
              policy_type: value,
            },
          },
        },
      };
    });
  };

  const handleDRLPolicyWeightChange = (value: number) => {
    const safeValue = Number.isFinite(value)
      ? Math.min(Math.max(value, 0), 1)
      : 0.5;

    setConfig((prev) => {
      const currentAutoml = prev.ga_config.hybrid_automl_config ?? {};
      const currentDrl = currentAutoml.drl ?? {
        enabled: false,
        policy_type: "ppo",
      };

      return {
        ...prev,
        ga_config: {
          ...prev.ga_config,
          hybrid_automl_config: {
            ...currentAutoml,
            drl: {
              ...currentDrl,
              policy_weight: safeValue,
            },
          },
        },
      };
    });
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
        
        {/* ハイブリッドモード設定 */}
        <div className="p-4 bg-blue-900/20 border border-blue-500/30 rounded-lg space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-blue-200">
              🔬 ハイブリッドGA+MLモード
            </label>
            <input
              type="checkbox"
              checked={config.ga_config.hybrid_mode || false}
              onChange={(e) =>
                handleGAConfigChange({ hybrid_mode: e.target.checked })
              }
              className="w-5 h-5 rounded border-gray-600 text-blue-600 focus:ring-blue-500"
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
                  { value: "catboost", label: "CatBoost" },
                  { value: "randomforest", label: "Random Forest" },
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
              <div className="p-3 bg-slate-900/40 border border-slate-600/30 rounded-md space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium text-slate-200">
                    🧠 DRLポリシーブレンド
                  </label>
                  <input
                    type="checkbox"
                    checked={drlEnabled}
                    onChange={(event) =>
                      handleDRLEnabledChange(event.target.checked)
                    }
                    className="w-5 h-5 rounded border-gray-600 text-blue-600 focus:ring-blue-500"
                    aria-label="DRLポリシーブレンドを有効化"
                  />
                </div>
                {drlEnabled && (
                  <>
                    <SelectField
                      label="DRLポリシー"
                      value={drlPolicyType}
                      onChange={handleDRLPolicyTypeChange}
                      options={DRL_POLICY_OPTIONS}
                    />
                    <InputField
                      label="DRLブレンド重み"
                      type="number"
                      value={drlPolicyWeight}
                      onChange={handleDRLPolicyWeightChange}
                      min={0}
                      max={1}
                      step={0.05}
                      description="DRLとML予測の混合比率（0-1）"
                    />
                  </>
                )}
              </div>
              <p className="text-xs text-blue-300">
                💡 事前にMLモデルを学習しておく必要があります。未学習の場合はデフォルト予測を使用します。
              </p>
            </>
          )}
        </div>
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
            <div className="pt-1">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.ga_config.regime_adaptation_enabled ?? false}
                  onChange={(e) =>
                    handleGAConfigChange({
                      regime_adaptation_enabled: e.target.checked,
                    })
                  }
                  className="rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
                />
                <span className="text-sm text-indigo-200">
                  レジーム適応を有効化
                </span>
              </label>
            </div>
          </div>

          {/* 多目的最適化設定 */}
          <ObjectiveSelection
            gaConfig={config.ga_config}
            onGAConfigChange={handleGAConfigChange}
          />
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
