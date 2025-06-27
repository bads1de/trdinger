/**
 * ストラテジービルダー用カスタムフック
 *
 * 戦略作成フローの状態管理とフォームバリデーションを提供します。
 */

"use client";

import { useState, useCallback, useEffect } from "react";
import {
  getAvailableIndicators,
  validateStrategy,
  saveStrategy,
  getStrategies,
  updateStrategy,
  deleteStrategy,
  formatApiError,
  formatValidationErrors,
  type IndicatorCategories,
  type UserStrategy,
  type ValidationResult,
} from "@/lib/api/strategy-builder";

// 型定義
export interface SelectedIndicator {
  id: string;
  type: string;
  name: string;
  parameters: Record<string, any>;
  enabled: boolean;
}

export interface Condition {
  id: string;
  type: "threshold" | "crossover" | "comparison";
  indicator1?: string;
  indicator2?: string;
  operator: string;
  value?: number;
  logicalOperator?: "AND" | "OR";
}

export type BuilderStep =
  | "indicators"
  | "parameters"
  | "conditions"
  | "preview"
  | "saved";

interface StrategyBuilderState {
  // 現在のステップ
  currentStep: BuilderStep;

  // 指標関連
  selectedIndicators: SelectedIndicator[];
  availableIndicators: IndicatorCategories;
  indicatorsLoading: boolean;
  indicatorsError: string | null;

  // 条件関連
  entryConditions: Condition[];
  exitConditions: Condition[];

  // 戦略情報
  strategyName: string;
  strategyDescription: string;

  // 保存済み戦略
  savedStrategies: UserStrategy[];
  strategiesLoading: boolean;
  strategiesError: string | null;

  // バリデーション
  validationResult: ValidationResult | null;
  validating: boolean;

  // 保存状態
  saving: boolean;
  saveError: string | null;
}

/**
 * ストラテジービルダー用カスタムフック
 */
export function useStrategyBuilder() {
  // 状態管理
  const [state, setState] = useState<StrategyBuilderState>({
    currentStep: "indicators",
    selectedIndicators: [],
    availableIndicators: {},
    indicatorsLoading: true,
    indicatorsError: null,
    entryConditions: [],
    exitConditions: [],
    strategyName: "",
    strategyDescription: "",
    savedStrategies: [],
    strategiesLoading: false,
    strategiesError: null,
    validationResult: null,
    validating: false,
    saving: false,
    saveError: null,
  });

  // 状態更新のヘルパー関数
  const updateState = useCallback((updates: Partial<StrategyBuilderState>) => {
    setState((prev) => ({ ...prev, ...updates }));
  }, []);

  // 利用可能な指標を取得
  const fetchIndicators = useCallback(async () => {
    try {
      updateState({ indicatorsLoading: true, indicatorsError: null });
      const indicators = await getAvailableIndicators();
      updateState({
        availableIndicators: indicators,
        indicatorsLoading: false,
      });
    } catch (error) {
      updateState({
        indicatorsError: formatApiError(error),
        indicatorsLoading: false,
      });
    }
  }, [updateState]);

  // 保存済み戦略を取得
  const fetchSavedStrategies = useCallback(async () => {
    try {
      updateState({ strategiesLoading: true, strategiesError: null });
      const result = await getStrategies();
      updateState({
        savedStrategies: result.strategies,
        strategiesLoading: false,
      });
    } catch (error) {
      updateState({
        strategiesError: formatApiError(error),
        strategiesLoading: false,
      });
    }
  }, [updateState]);

  // 指標の選択/選択解除
  const toggleIndicator = useCallback(
    (indicatorType: string, indicatorName: string) => {
      setState((prev) => {
        const isSelected = prev.selectedIndicators.some(
          (ind) => ind.type === indicatorType
        );

        if (isSelected) {
          // 選択解除
          return {
            ...prev,
            selectedIndicators: prev.selectedIndicators.filter(
              (ind) => ind.type !== indicatorType
            ),
          };
        } else {
          // 選択
          const newIndicator: SelectedIndicator = {
            id: `${indicatorType}_${Date.now()}`,
            type: indicatorType,
            name: indicatorName,
            parameters: {},
            enabled: true,
          };

          return {
            ...prev,
            selectedIndicators: [...prev.selectedIndicators, newIndicator],
          };
        }
      });
    },
    []
  );

  // 選択済み指標の直接更新
  const updateSelectedIndicators = useCallback(
    (indicators: SelectedIndicator[]) => {
      updateState({ selectedIndicators: indicators });
    },
    [updateState]
  );

  // 指標のパラメータ更新
  const updateIndicatorParameters = useCallback(
    (indicatorId: string, parameters: Record<string, any>) => {
      setState((prev) => ({
        ...prev,
        selectedIndicators: prev.selectedIndicators.map((indicator) =>
          indicator.id === indicatorId
            ? { ...indicator, parameters }
            : indicator
        ),
      }));
    },
    []
  );

  // 指標の有効/無効切り替え
  const toggleIndicatorEnabled = useCallback(
    (indicatorId: string, enabled: boolean) => {
      setState((prev) => ({
        ...prev,
        selectedIndicators: prev.selectedIndicators.map((indicator) =>
          indicator.id === indicatorId ? { ...indicator, enabled } : indicator
        ),
      }));
    },
    []
  );

  // エントリー条件の更新
  const updateEntryConditions = useCallback(
    (conditions: Condition[]) => {
      updateState({ entryConditions: conditions });
    },
    [updateState]
  );

  // イグジット条件の更新
  const updateExitConditions = useCallback(
    (conditions: Condition[]) => {
      updateState({ exitConditions: conditions });
    },
    [updateState]
  );

  // 戦略名の更新
  const updateStrategyName = useCallback(
    (name: string) => {
      updateState({ strategyName: name, saveError: null });
    },
    [updateState]
  );

  // 戦略説明の更新
  const updateStrategyDescription = useCallback(
    (description: string) => {
      updateState({ strategyDescription: description });
    },
    [updateState]
  );

  // ステップの変更
  const setCurrentStep = useCallback(
    (step: BuilderStep) => {
      updateState({ currentStep: step });
    },
    [updateState]
  );

  // 戦略の検証
  const validateCurrentStrategy = useCallback(async (): Promise<boolean> => {
    try {
      updateState({ validating: true });

      // StrategyGene形式に変換
      const strategyConfig = {
        indicators: state.selectedIndicators
          .filter((ind) => ind.enabled)
          .map((ind) => ({
            type: ind.type,
            parameters: ind.parameters,
            enabled: true,
          })),
        entry_conditions: state.entryConditions,
        exit_conditions: state.exitConditions,
      };

      const result = await validateStrategy(strategyConfig);
      updateState({
        validationResult: result,
        validating: false,
      });

      return result.is_valid;
    } catch (error) {
      updateState({
        validationResult: {
          is_valid: false,
          errors: [formatApiError(error)],
        },
        validating: false,
      });
      return false;
    }
  }, [
    state.selectedIndicators,
    state.entryConditions,
    state.exitConditions,
    updateState,
  ]);

  // 戦略の保存
  const saveCurrentStrategy =
    useCallback(async (): Promise<UserStrategy | null> => {
      try {
        updateState({ saving: true, saveError: null });

        // バリデーション
        if (!state.strategyName.trim()) {
          throw new Error("戦略名を入力してください");
        }

        const enabledIndicators = state.selectedIndicators.filter(
          (ind) => ind.enabled
        );
        if (enabledIndicators.length === 0) {
          throw new Error("少なくとも1つの指標を有効にしてください");
        }

        if (
          state.entryConditions.length === 0 &&
          state.exitConditions.length === 0
        ) {
          throw new Error("少なくとも1つの条件を設定してください");
        }

        // StrategyGene形式に変換
        const strategyConfig = {
          indicators: enabledIndicators.map((ind) => ({
            type: ind.type,
            parameters: ind.parameters,
            enabled: true,
            json_config: {
              indicator_name: ind.type,
              parameters: ind.parameters,
            },
          })),
          entry_conditions: state.entryConditions,
          exit_conditions: state.exitConditions,
          risk_management: {
            stop_loss_pct: 0.02,
            take_profit_pct: 0.05,
            position_sizing: "fixed",
          },
          metadata: {
            created_by: "strategy_builder",
            version: "1.0",
            created_at: new Date().toISOString(),
          },
        };

        const savedStrategy = await saveStrategy({
          name: state.strategyName,
          description: state.strategyDescription || undefined,
          strategy_config: strategyConfig,
        });

        updateState({ saving: false });

        // 保存済み戦略一覧を更新
        await fetchSavedStrategies();

        return savedStrategy;
      } catch (error) {
        updateState({
          saving: false,
          saveError: formatApiError(error),
        });
        return null;
      }
    }, [
      state.strategyName,
      state.strategyDescription,
      state.selectedIndicators,
      state.entryConditions,
      state.exitConditions,
      updateState,
      fetchSavedStrategies,
    ]);

  // 戦略の読み込み
  const loadStrategy = useCallback(
    (strategy: UserStrategy) => {
      const config = strategy.strategy_config;

      // 指標を復元
      const indicators: SelectedIndicator[] = (config.indicators || []).map(
        (ind: any, index: number) => ({
          id: `${ind.type}_${Date.now()}_${index}`,
          type: ind.type,
          name: ind.json_config?.indicator_name || ind.type,
          parameters: ind.parameters || {},
          enabled: ind.enabled !== false,
        })
      );

      updateState({
        selectedIndicators: indicators,
        entryConditions: config.entry_conditions || [],
        exitConditions: config.exit_conditions || [],
        strategyName: strategy.name,
        strategyDescription: strategy.description || "",
        currentStep: "indicators",
      });
    },
    [updateState]
  );

  // 戦略の削除
  const deleteStrategyById = useCallback(
    async (strategyId: number): Promise<boolean> => {
      try {
        await deleteStrategy(strategyId);
        await fetchSavedStrategies();
        return true;
      } catch (error) {
        updateState({ strategiesError: formatApiError(error) });
        return false;
      }
    },
    [fetchSavedStrategies, updateState]
  );

  // 戦略のリセット
  const resetStrategy = useCallback(() => {
    updateState({
      selectedIndicators: [],
      entryConditions: [],
      exitConditions: [],
      strategyName: "",
      strategyDescription: "",
      validationResult: null,
      saveError: null,
      currentStep: "indicators",
    });
  }, [updateState]);

  // ステップ進行の可否判定
  const canProceedToStep = useCallback(
    (step: BuilderStep): boolean => {
      switch (step) {
        case "indicators":
          return true;
        case "parameters":
          return state.selectedIndicators.length > 0;
        case "conditions":
          return state.selectedIndicators.some((ind) => ind.enabled);
        case "preview":
          return (
            state.selectedIndicators.some((ind) => ind.enabled) &&
            (state.entryConditions.length > 0 ||
              state.exitConditions.length > 0)
          );
        case "saved":
          return true;
        default:
          return false;
      }
    },
    [state.selectedIndicators, state.entryConditions, state.exitConditions]
  );

  // 初期化
  useEffect(() => {
    fetchIndicators();
  }, [fetchIndicators]);

  // 戦略作成の完了状況
  const isStrategyComplete =
    canProceedToStep("preview") && state.strategyName.trim().length > 0;

  return {
    // 状態
    ...state,

    // 計算されたプロパティ
    isStrategyComplete,

    // アクション
    fetchIndicators,
    fetchSavedStrategies,
    toggleIndicator,
    updateSelectedIndicators,
    updateIndicatorParameters,
    toggleIndicatorEnabled,
    updateEntryConditions,
    updateExitConditions,
    updateStrategyName,
    updateStrategyDescription,
    setCurrentStep,
    validateCurrentStrategy,
    saveCurrentStrategy,
    loadStrategy,
    deleteStrategyById,
    resetStrategy,
    canProceedToStep,
  };
}
