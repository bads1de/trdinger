/**
 * 戦略プレビューコンポーネント
 *
 * 戦略のプレビューとStrategyGene構造の表示を提供します。
 */

"use client";

import React, { useState } from "react";
import TabButton from "@/components/common/TabButton";
import { InputField } from "@/components/common/InputField";
import ApiButton from "@/components/button/ApiButton";

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

interface StrategyPreviewProps {
  selectedIndicators: SelectedIndicator[];
  entryConditions: Condition[];
  exitConditions: Condition[];
  strategyName: string;
  strategyDescription: string;
  onStrategyNameChange: (name: string) => void;
  onStrategyDescriptionChange: (description: string) => void;
  onSaveStrategy: () => Promise<void>;
  onValidateStrategy: () => Promise<boolean>;
}

type PreviewTab = "summary" | "indicators" | "conditions" | "json";

/**
 * 戦略プレビューコンポーネント
 */
const StrategyPreview: React.FC<StrategyPreviewProps> = ({
  selectedIndicators,
  entryConditions,
  exitConditions,
  strategyName,
  strategyDescription,
  onStrategyNameChange,
  onStrategyDescriptionChange,
  onSaveStrategy,
  onValidateStrategy,
}) => {
  // 状態管理
  const [activeTab, setActiveTab] = useState<PreviewTab>("summary");
  const [saving, setSaving] = useState<boolean>(false);
  const [validating, setValidating] = useState<boolean>(false);
  const [validationResult, setValidationResult] = useState<{
    isValid: boolean;
    errors: string[];
  } | null>(null);

  // StrategyGene形式のJSONを生成
  const generateStrategyGene = () => {
    const enabledIndicators = selectedIndicators.filter((ind) => ind.enabled);

    // 指標をStrategyGene形式に変換
    const indicators = enabledIndicators.map((indicator, index) => ({
      type: indicator.type,
      parameters: indicator.parameters,
      enabled: true,
      json_config: {
        indicator_name: indicator.type,
        parameters: indicator.parameters,
      },
    }));

    // 条件をStrategyGene形式に変換
    const convertConditions = (conditions: Condition[]) => {
      return conditions.map((condition) => {
        const baseCondition: any = {
          type: condition.type,
          operator: condition.operator,
        };

        if (condition.type === "threshold") {
          baseCondition.indicator = condition.indicator1;
          baseCondition.value = condition.value;
        } else if (
          condition.type === "crossover" ||
          condition.type === "comparison"
        ) {
          baseCondition.indicator1 = condition.indicator1;
          baseCondition.indicator2 = condition.indicator2;
        }

        return baseCondition;
      });
    };

    return {
      id: `user_strategy_${Date.now()}`,
      indicators,
      entry_conditions: convertConditions(entryConditions),
      exit_conditions: convertConditions(exitConditions),
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
  };

  // 戦略の検証
  const handleValidate = async () => {
    try {
      setValidating(true);
      const isValid = await onValidateStrategy();

      if (isValid) {
        setValidationResult({
          isValid: true,
          errors: [],
        });
      } else {
        setValidationResult({
          isValid: false,
          errors: ["戦略の設定に問題があります"],
        });
      }
    } catch (error) {
      setValidationResult({
        isValid: false,
        errors: [
          error instanceof Error ? error.message : "検証エラーが発生しました",
        ],
      });
    } finally {
      setValidating(false);
    }
  };

  // 戦略の保存
  const handleSave = async () => {
    try {
      setSaving(true);
      await onSaveStrategy();
    } finally {
      setSaving(false);
    }
  };

  // 戦略が保存可能かチェック
  const canSave = () => {
    return (
      strategyName.trim().length > 0 &&
      selectedIndicators.some((ind) => ind.enabled) &&
      (entryConditions.length > 0 || exitConditions.length > 0)
    );
  };

  // サマリータブの内容
  const renderSummaryTab = () => (
    <div className="space-y-6">
      {/* 戦略情報 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InputField
          label="戦略名 *"
          value={strategyName}
          onChange={onStrategyNameChange}
          placeholder="戦略名を入力してください"
          required
        />
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            戦略の説明
          </label>
          <textarea
            value={strategyDescription}
            onChange={(e) => onStrategyDescriptionChange(e.target.value)}
            placeholder="戦略の説明を入力してください（オプション）"
            className="w-full p-3 bg-secondary-950 border border-gray-600 text-white rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent h-24"
          />
        </div>
      </div>

      {/* 戦略統計 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-700 p-4 rounded-lg">
          <p className="text-gray-400 text-sm">選択指標</p>
          <p className="text-white text-xl font-bold">
            {selectedIndicators.filter((ind) => ind.enabled).length}
          </p>
        </div>
        <div className="bg-gray-700 p-4 rounded-lg">
          <p className="text-gray-400 text-sm">エントリー条件</p>
          <p className="text-white text-xl font-bold">
            {entryConditions.length}
          </p>
        </div>
        <div className="bg-gray-700 p-4 rounded-lg">
          <p className="text-gray-400 text-sm">イグジット条件</p>
          <p className="text-white text-xl font-bold">
            {exitConditions.length}
          </p>
        </div>
        <div className="bg-gray-700 p-4 rounded-lg">
          <p className="text-gray-400 text-sm">設定完了</p>
          <p
            className={`text-xl font-bold ${
              canSave() ? "text-green-400" : "text-yellow-400"
            }`}
          >
            {canSave() ? "完了" : "未完了"}
          </p>
        </div>
      </div>

      {/* 検証結果 */}
      {validationResult && (
        <div
          className={`
          p-4 rounded-lg border
          ${
            validationResult.isValid
              ? "bg-green-900/30 border-green-700"
              : "bg-red-900/30 border-red-700"
          }
        `}
        >
          <div className="flex items-center gap-2 mb-2">
            {validationResult.isValid ? (
              <svg
                className="w-5 h-5 text-green-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
            ) : (
              <svg
                className="w-5 h-5 text-red-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            )}
            <h5
              className={`font-medium ${
                validationResult.isValid ? "text-green-300" : "text-red-300"
              }`}
            >
              {validationResult.isValid
                ? "戦略は有効です"
                : "戦略に問題があります"}
            </h5>
          </div>
          {validationResult.errors.length > 0 && (
            <ul className="text-red-400 text-sm space-y-1">
              {validationResult.errors.map((error, index) => (
                <li key={index}>• {error}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* アクションボタン */}
      <div className="flex gap-4">
        <ApiButton
          onClick={handleValidate}
          loading={validating}
          variant="secondary"
          className="flex-1"
        >
          戦略を検証
        </ApiButton>
        <ApiButton
          onClick={handleSave}
          loading={saving}
          disabled={!canSave()}
          variant="primary"
          className="flex-1"
        >
          戦略を保存
        </ApiButton>
      </div>
    </div>
  );

  // 指標タブの内容
  const renderIndicatorsTab = () => (
    <div className="space-y-4">
      {selectedIndicators
        .filter((ind) => ind.enabled)
        .map((indicator) => (
          <div key={indicator.id} className="bg-gray-700 p-4 rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h5 className="font-medium text-white">{indicator.name}</h5>
                <p className="text-sm text-gray-400">{indicator.type}</p>
              </div>
              <span className="px-2 py-1 bg-green-600 text-white text-xs rounded">
                有効
              </span>
            </div>

            {Object.keys(indicator.parameters).length > 0 && (
              <div>
                <p className="text-sm text-gray-300 mb-2">パラメータ:</p>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  {Object.entries(indicator.parameters).map(([key, value]) => (
                    <div key={key} className="bg-gray-800 p-2 rounded">
                      <p className="text-xs text-gray-400">{key}</p>
                      <p className="text-sm text-white">{String(value)}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}

      {selectedIndicators.filter((ind) => ind.enabled).length === 0 && (
        <div className="text-center py-8">
          <p className="text-gray-400">有効な指標がありません</p>
        </div>
      )}
    </div>
  );

  // 条件タブの内容
  const renderConditionsTab = () => (
    <div className="space-y-6">
      {/* エントリー条件 */}
      <div>
        <h5 className="font-medium text-white mb-3">
          エントリー条件 ({entryConditions.length})
        </h5>
        {entryConditions.length > 0 ? (
          <div className="space-y-2">
            {entryConditions.map((condition, index) => (
              <div key={condition.id} className="bg-gray-700 p-3 rounded">
                <p className="text-sm text-gray-300">
                  {index > 0 && (
                    <span className="text-blue-400 mr-2">
                      {condition.logicalOperator}
                    </span>
                  )}
                  条件 {index + 1}: {condition.type} - {condition.operator}
                  {condition.value !== undefined && ` ${condition.value}`}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-400 text-sm">
            エントリー条件が設定されていません
          </p>
        )}
      </div>

      {/* イグジット条件 */}
      <div>
        <h5 className="font-medium text-white mb-3">
          イグジット条件 ({exitConditions.length})
        </h5>
        {exitConditions.length > 0 ? (
          <div className="space-y-2">
            {exitConditions.map((condition, index) => (
              <div key={condition.id} className="bg-gray-700 p-3 rounded">
                <p className="text-sm text-gray-300">
                  {index > 0 && (
                    <span className="text-blue-400 mr-2">
                      {condition.logicalOperator}
                    </span>
                  )}
                  条件 {index + 1}: {condition.type} - {condition.operator}
                  {condition.value !== undefined && ` ${condition.value}`}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-400 text-sm">
            イグジット条件が設定されていません
          </p>
        )}
      </div>
    </div>
  );

  // JSONタブの内容
  const renderJsonTab = () => {
    const strategyGene = generateStrategyGene();

    return (
      <div>
        <div className="flex items-center justify-between mb-4">
          <h5 className="font-medium text-white">StrategyGene JSON</h5>
          <button
            onClick={() =>
              navigator.clipboard.writeText(
                JSON.stringify(strategyGene, null, 2)
              )
            }
            className="px-3 py-1 bg-gray-600 text-white text-sm rounded hover:bg-gray-500"
          >
            コピー
          </button>
        </div>
        <pre className="bg-secondary-950 p-4 rounded-lg text-sm text-gray-300 overflow-auto max-h-96">
          {JSON.stringify(strategyGene, null, 2)}
        </pre>
      </div>
    );
  };

  return (
    <div className="bg-secondary-950 rounded-lg p-6">
      {/* タブナビゲーション */}
      <div className="flex flex-wrap gap-2 mb-6">
        <TabButton
          label="サマリー"
          isActive={activeTab === "summary"}
          onClick={() => setActiveTab("summary")}
          icon={
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
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
          }
        />
        <TabButton
          label="指標"
          isActive={activeTab === "indicators"}
          onClick={() => setActiveTab("indicators")}
          badge={selectedIndicators.filter((ind) => ind.enabled).length}
          icon={
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
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
              />
            </svg>
          }
        />
        <TabButton
          label="条件"
          isActive={activeTab === "conditions"}
          onClick={() => setActiveTab("conditions")}
          badge={entryConditions.length + exitConditions.length}
          icon={
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
                d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          }
        />
        <TabButton
          label="JSON"
          isActive={activeTab === "json"}
          onClick={() => setActiveTab("json")}
          icon={
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
                d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
              />
            </svg>
          }
        />
      </div>

      {/* タブコンテンツ */}
      <div>
        {activeTab === "summary" && renderSummaryTab()}
        {activeTab === "indicators" && renderIndicatorsTab()}
        {activeTab === "conditions" && renderConditionsTab()}
        {activeTab === "json" && renderJsonTab()}
      </div>
    </div>
  );
};

export default StrategyPreview;
