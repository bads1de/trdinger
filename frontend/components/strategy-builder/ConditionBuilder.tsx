/**
 * 条件ビルダーコンポーネント
 *
 * エントリー・イグジット条件の設定と論理演算子管理を提供します。
 */

"use client";

import React, { useState } from "react";
import { SelectField } from "@/components/common/SelectField";
import { InputField } from "@/components/common/InputField";

interface SelectedIndicator {
  id: string;
  type: string;
  name: string;
  parameters: Record<string, any>;
  enabled: boolean;
}

interface Condition {
  id: string;
  type: "threshold" | "crossover" | "comparison";
  indicator1?: string;
  indicator2?: string;
  operator: string;
  value?: number;
  logicalOperator?: "AND" | "OR";
}

interface ConditionBuilderProps {
  selectedIndicators: SelectedIndicator[];
  entryConditions: Condition[];
  exitConditions: Condition[];
  onEntryConditionsChange: (conditions: Condition[]) => void;
  onExitConditionsChange: (conditions: Condition[]) => void;
}

// 演算子の定義
const OPERATORS = {
  threshold: [
    { value: ">", label: "より大きい (>)" },
    { value: ">=", label: "以上 (>=)" },
    { value: "<", label: "より小さい (<)" },
    { value: "<=", label: "以下 (<=)" },
    { value: "==", label: "等しい (==)" },
    { value: "!=", label: "等しくない (!=)" },
  ],
  crossover: [
    { value: "above", label: "上抜け" },
    { value: "below", label: "下抜け" },
  ],
  comparison: [
    { value: ">", label: "より大きい (>)" },
    { value: ">=", label: "以上 (>=)" },
    { value: "<", label: "より小さい (<)" },
    { value: "<=", label: "以下 (<=)" },
    { value: "==", label: "等しい (==)" },
  ],
};

const CONDITION_TYPES = [
  { value: "threshold", label: "閾値条件" },
  { value: "crossover", label: "クロスオーバー" },
  { value: "comparison", label: "指標比較" },
];

const LOGICAL_OPERATORS = [
  { value: "AND", label: "AND (すべて満たす)" },
  { value: "OR", label: "OR (いずれか満たす)" },
];

/**
 * 条件ビルダーコンポーネント
 */
const ConditionBuilder: React.FC<ConditionBuilderProps> = ({
  selectedIndicators,
  entryConditions,
  exitConditions,
  onEntryConditionsChange,
  onExitConditionsChange,
}) => {
  // 状態管理
  const [activeTab, setActiveTab] = useState<"entry" | "exit">("entry");

  // 有効な指標のオプションを取得
  const getIndicatorOptions = () => {
    return selectedIndicators
      .filter((indicator) => indicator.enabled)
      .map((indicator) => ({
        value: indicator.id,
        label: `${indicator.name} (${indicator.type})`,
      }));
  };

  // 新しい条件を追加
  const addCondition = (type: "entry" | "exit") => {
    const newCondition: Condition = {
      id: `condition_${Date.now()}`,
      type: "threshold",
      operator: ">",
      value: 0,
      logicalOperator: "AND",
    };

    if (type === "entry") {
      onEntryConditionsChange([...entryConditions, newCondition]);
    } else {
      onExitConditionsChange([...exitConditions, newCondition]);
    }
  };

  // 条件を削除
  const removeCondition = (type: "entry" | "exit", conditionId: string) => {
    if (type === "entry") {
      onEntryConditionsChange(
        entryConditions.filter((c) => c.id !== conditionId)
      );
    } else {
      onExitConditionsChange(
        exitConditions.filter((c) => c.id !== conditionId)
      );
    }
  };

  // 条件を更新
  const updateCondition = (
    type: "entry" | "exit",
    conditionId: string,
    updates: Partial<Condition>
  ) => {
    const updateConditions = (conditions: Condition[]) =>
      conditions.map((condition) =>
        condition.id === conditionId ? { ...condition, ...updates } : condition
      );

    if (type === "entry") {
      onEntryConditionsChange(updateConditions(entryConditions));
    } else {
      onExitConditionsChange(updateConditions(exitConditions));
    }
  };

  // 条件エディターをレンダリング
  const renderConditionEditor = (
    condition: Condition,
    type: "entry" | "exit",
    index: number
  ) => {
    const indicatorOptions = getIndicatorOptions();

    return (
      <div
        key={condition.id}
        className="border border-gray-600 rounded-lg p-4 bg-secondary-950"
      >
        <div className="flex items-center justify-between mb-4">
          <h5 className="font-medium text-white">条件 {index + 1}</h5>
          <button
            onClick={() => removeCondition(type, condition.id)}
            className="text-red-400 hover:text-red-300 p-1"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
              />
            </svg>
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* 条件タイプ */}
          <SelectField
            label="条件タイプ"
            value={condition.type}
            onChange={(value) =>
              updateCondition(type, condition.id, {
                type: value as Condition["type"],
                // タイプ変更時にリセット
                indicator1: undefined,
                indicator2: undefined,
                value: undefined,
                operator:
                  OPERATORS[value as keyof typeof OPERATORS][0]?.value || ">",
              })
            }
            options={CONDITION_TYPES}
          />

          {/* 論理演算子（最初の条件以外） */}
          {index > 0 && (
            <SelectField
              label="論理演算子"
              value={condition.logicalOperator || "AND"}
              onChange={(value) =>
                updateCondition(type, condition.id, {
                  logicalOperator: value as "AND" | "OR",
                })
              }
              options={LOGICAL_OPERATORS}
            />
          )}
        </div>

        <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* 指標1 */}
          <SelectField
            label={condition.type === "crossover" ? "基準指標" : "指標"}
            value={condition.indicator1 || ""}
            onChange={(value) =>
              updateCondition(type, condition.id, { indicator1: value })
            }
            options={[
              { value: "", label: "指標を選択..." },
              ...indicatorOptions,
            ]}
          />

          {/* 演算子 */}
          <SelectField
            label="演算子"
            value={condition.operator}
            onChange={(value) =>
              updateCondition(type, condition.id, { operator: value })
            }
            options={OPERATORS[condition.type] || OPERATORS.threshold}
          />

          {/* 右辺（指標2または値） */}
          {condition.type === "crossover" || condition.type === "comparison" ? (
            <SelectField
              label={condition.type === "crossover" ? "比較指標" : "比較指標"}
              value={condition.indicator2 || ""}
              onChange={(value) =>
                updateCondition(type, condition.id, { indicator2: value })
              }
              options={[
                { value: "", label: "指標を選択..." },
                ...indicatorOptions.filter(
                  (opt) => opt.value !== condition.indicator1
                ),
              ]}
            />
          ) : (
            <InputField
              label="値"
              type="number"
              value={condition.value || 0}
              onChange={(value) =>
                updateCondition(type, condition.id, { value: Number(value) })
              }
              step={0.1}
            />
          )}
        </div>

        {/* 条件の説明 */}
        <div className="mt-4 p-3 bg-secondary-950 rounded border border-gray-600">
          <p className="text-sm text-gray-300">
            {renderConditionDescription(condition, indicatorOptions)}
          </p>
        </div>
      </div>
    );
  };

  // 条件の説明文を生成
  const renderConditionDescription = (
    condition: Condition,
    indicatorOptions: Array<{ value: string; label: string }>
  ) => {
    const getIndicatorName = (id: string) => {
      const option = indicatorOptions.find((opt) => opt.value === id);
      return option ? option.label : "未選択";
    };

    const operatorText =
      OPERATORS[condition.type]?.find((op) => op.value === condition.operator)
        ?.label || condition.operator;

    switch (condition.type) {
      case "threshold":
        return `${getIndicatorName(condition.indicator1 || "")} が ${
          condition.value
        } ${operatorText}`;
      case "crossover":
        return `${getIndicatorName(
          condition.indicator1 || ""
        )} が ${getIndicatorName(
          condition.indicator2 || ""
        )} を ${operatorText}`;
      case "comparison":
        return `${getIndicatorName(
          condition.indicator1 || ""
        )} が ${getIndicatorName(condition.indicator2 || "")} ${operatorText}`;
      default:
        return "条件を設定してください";
    }
  };

  if (selectedIndicators.filter((ind) => ind.enabled).length === 0) {
    return (
      <div className="bg-secondary-950 rounded-lg p-6">
        <div className="text-center py-8">
          <svg
            className="w-12 h-12 text-gray-500 mx-auto mb-4"
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
          <p className="text-gray-400 text-lg">有効な指標がありません</p>
          <p className="text-gray-500 text-sm mt-2">
            まず指標を選択し、パラメータを設定してください
          </p>
        </div>
      </div>
    );
  }

  const currentConditions =
    activeTab === "entry" ? entryConditions : exitConditions;

  return (
    <div className="bg-secondary-950 rounded-lg p-6">
      {/* タブ */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setActiveTab("entry")}
          className={`
            px-4 py-2 rounded-lg font-medium transition-colors
            ${
              activeTab === "entry"
                ? "bg-blue-600 text-white"
                : "bg-gray-700 text-gray-300 hover:bg-gray-600"
            }
          `}
        >
          エントリー条件 ({entryConditions.length})
        </button>
        <button
          onClick={() => setActiveTab("exit")}
          className={`
            px-4 py-2 rounded-lg font-medium transition-colors
            ${
              activeTab === "exit"
                ? "bg-blue-600 text-white"
                : "bg-gray-700 text-gray-300 hover:bg-gray-600"
            }
          `}
        >
          イグジット条件 ({exitConditions.length})
        </button>
      </div>

      {/* 条件一覧 */}
      <div className="space-y-4 mb-6">
        {currentConditions.length > 0 ? (
          currentConditions.map((condition, index) =>
            renderConditionEditor(condition, activeTab, index)
          )
        ) : (
          <div className="text-center py-8 border-2 border-dashed border-gray-600 rounded-lg">
            <p className="text-gray-400">
              {activeTab === "entry" ? "エントリー" : "イグジット"}
              条件が設定されていません
            </p>
            <p className="text-gray-500 text-sm mt-2">
              「条件を追加」ボタンで新しい条件を作成してください
            </p>
          </div>
        )}
      </div>

      {/* 条件追加ボタン */}
      <button
        onClick={() => addCondition(activeTab)}
        className="w-full py-3 border-2 border-dashed border-gray-600 rounded-lg text-gray-400 hover:border-gray-500 hover:text-gray-300 transition-colors"
      >
        <div className="flex items-center justify-center gap-2">
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
              d="M12 6v6m0 0v6m0-6h6m-6 0H6"
            />
          </svg>
          {activeTab === "entry" ? "エントリー" : "イグジット"}条件を追加
        </div>
      </button>

      {/* 条件サマリー */}
      <div className="mt-6 p-4 bg-secondary-950 rounded-lg border border-gray-600">
        <h5 className="text-gray-300 font-medium mb-3">条件サマリー</h5>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-gray-400">エントリー条件</p>
            <p className="text-white font-medium">{entryConditions.length}個</p>
          </div>
          <div>
            <p className="text-gray-400">イグジット条件</p>
            <p className="text-white font-medium">{exitConditions.length}個</p>
          </div>
        </div>

        {entryConditions.length === 0 && exitConditions.length === 0 && (
          <div className="mt-3 p-3 bg-yellow-900/30 border border-yellow-700 rounded">
            <p className="text-yellow-300 text-sm">
              ⚠️ 少なくとも1つの条件を設定してください
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ConditionBuilder;
