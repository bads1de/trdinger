/**
 * ストラテジービルダーページ
 *
 * ユーザーが58種類のテクニカル指標を組み合わせて
 * 独自の投資戦略を作成できるページです。
 */

"use client";

import React from "react";
import TabButton from "@/components/common/TabButton";
import IndicatorSelector from "@/components/strategy-builder/IndicatorSelector";
import ParameterEditor from "@/components/strategy-builder/ParameterEditor";
import ConditionBuilder from "@/components/strategy-builder/ConditionBuilder";
import StrategyPreview from "@/components/strategy-builder/StrategyPreview";
import SavedStrategies from "@/components/strategy-builder/SavedStrategies";
import { useStrategyBuilder } from "@/hooks/useStrategyBuilder";

// ステップの定義
type BuilderStep =
  | "indicators"
  | "parameters"
  | "conditions"
  | "preview"
  | "saved";

interface StepInfo {
  id: BuilderStep;
  label: string;
  description: string;
  icon: React.ReactNode;
}

const BUILDER_STEPS: StepInfo[] = [
  {
    id: "indicators",
    label: "指標選択",
    description: "使用するテクニカル指標を選択",
    icon: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
        />
      </svg>
    ),
  },
  {
    id: "parameters",
    label: "パラメータ設定",
    description: "選択した指標のパラメータを調整",
    icon: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4"
        />
      </svg>
    ),
  },
  {
    id: "conditions",
    label: "条件設定",
    description: "エントリー・イグジット条件を設定",
    icon: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
    ),
  },
  {
    id: "preview",
    label: "プレビュー",
    description: "戦略の確認と保存",
    icon: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
        />
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
        />
      </svg>
    ),
  },
  {
    id: "saved",
    label: "保存済み戦略",
    description: "作成済み戦略の管理",
    icon: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
        />
      </svg>
    ),
  },
];

/**
 * ストラテジービルダーページコンポーネント
 */
const StrategyBuilderPage: React.FC = () => {
  // カスタムフックから状態とアクションを取得
  const {
    currentStep,
    selectedIndicators,
    entryConditions,
    exitConditions,
    strategyName,
    strategyDescription,
    isStrategyComplete,
    setCurrentStep,
    updateSelectedIndicators,
    updateIndicatorParameters,
    toggleIndicatorEnabled,
    updateEntryConditions,
    updateExitConditions,
    updateStrategyName,
    updateStrategyDescription,
    validateCurrentStrategy,
    saveCurrentStrategy,
    loadStrategy,
    canProceedToStep,
  } = useStrategyBuilder();

  // ステップ変更ハンドラー
  const handleStepChange = (step: BuilderStep) => {
    if (canProceedToStep(step)) {
      setCurrentStep(step);
    }
  };

  // 現在のステップの情報を取得
  const currentStepInfo = BUILDER_STEPS.find((step) => step.id === currentStep);

  // ステップコンテンツのレンダリング
  const renderStepContent = () => {
    switch (currentStep) {
      case "indicators":
        return (
          <IndicatorSelector
            selectedIndicators={selectedIndicators}
            onIndicatorsChange={updateSelectedIndicators}
            maxIndicators={5}
          />
        );

      case "parameters":
        return (
          <ParameterEditor
            selectedIndicators={selectedIndicators}
            onParametersChange={updateIndicatorParameters}
            onIndicatorToggle={toggleIndicatorEnabled}
          />
        );

      case "conditions":
        return (
          <ConditionBuilder
            selectedIndicators={selectedIndicators}
            entryConditions={entryConditions}
            exitConditions={exitConditions}
            onEntryConditionsChange={updateEntryConditions}
            onExitConditionsChange={updateExitConditions}
          />
        );

      case "preview":
        return (
          <StrategyPreview
            selectedIndicators={selectedIndicators}
            entryConditions={entryConditions}
            exitConditions={exitConditions}
            strategyName={strategyName}
            strategyDescription={strategyDescription}
            onStrategyNameChange={updateStrategyName}
            onStrategyDescriptionChange={updateStrategyDescription}
            onSaveStrategy={async () => {
              await saveCurrentStrategy();
            }}
            onValidateStrategy={validateCurrentStrategy}
          />
        );

      case "saved":
        return (
          <SavedStrategies
            onLoadStrategy={loadStrategy}
            onEditStrategy={(strategy) => {
              loadStrategy(strategy);
              setCurrentStep("indicators");
            }}
          />
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen animate-fade-in">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        <div className="bg-card text-card-foreground shadow-lg rounded-lg p-6">
          <h1 className="text-3xl font-bold">ストラテジービルダー</h1>
          <p className="mt-2 text-muted-foreground">
            テクニカル指標を組み合わせて独自の投資戦略を作成
          </p>
        </div>

        {/* ステップナビゲーション */}
        <div className="bg-card text-card-foreground shadow-lg rounded-lg p-4">
          <div className="flex flex-wrap gap-2">
            {BUILDER_STEPS.map((step) => (
              <TabButton
                key={step.id}
                label={step.label}
                isActive={currentStep === step.id}
                onClick={() => handleStepChange(step.id)}
                disabled={!canProceedToStep(step.id)}
                icon={step.icon}
                variant="primary"
                size="md"
              />
            ))}
          </div>
        </div>

        {/* メインコンテンツ */}
        <div className="bg-card text-card-foreground shadow-lg rounded-lg p-6">
          {currentStepInfo && (
            <div className="mb-4">
              <div className="flex items-center gap-3 mb-2">
                <div className="text-primary">{currentStepInfo.icon}</div>
                <h2 className="text-xl font-semibold">
                  {currentStepInfo.label}
                </h2>
              </div>
              <p className="text-muted-foreground">
                {currentStepInfo.description}
              </p>
            </div>
          )}
          {/* ステップコンテンツ */}
          <div className="mb-8">{renderStepContent()}</div>

          {/* ナビゲーションボタン */}
          <div className="flex justify-between">
            <button
              onClick={() => {
                const currentIndex = BUILDER_STEPS.findIndex(
                  (step) => step.id === currentStep
                );
                if (currentIndex > 0) {
                  setCurrentStep(BUILDER_STEPS[currentIndex - 1].id);
                }
              }}
              disabled={
                BUILDER_STEPS.findIndex((step) => step.id === currentStep) === 0
              }
              className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-secondary text-secondary-foreground hover:bg-secondary/80 h-10 px-4 py-2"
            >
              前のステップ
            </button>

            <button
              onClick={() => {
                const currentIndex = BUILDER_STEPS.findIndex(
                  (step) => step.id === currentStep
                );
                if (currentIndex < BUILDER_STEPS.length - 1) {
                  const nextStep = BUILDER_STEPS[currentIndex + 1];
                  if (canProceedToStep(nextStep.id)) {
                    setCurrentStep(nextStep.id);
                  }
                }
              }}
              disabled={
                BUILDER_STEPS.findIndex((step) => step.id === currentStep) ===
                  BUILDER_STEPS.length - 1 ||
                !canProceedToStep(
                  BUILDER_STEPS[
                    BUILDER_STEPS.findIndex((step) => step.id === currentStep) +
                      1
                  ]?.id
                )
              }
              className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground hover:bg-primary/90 h-10 px-4 py-2"
            >
              次のステップ
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StrategyBuilderPage;
