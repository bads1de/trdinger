/**
 * パラメータエディターコンポーネント
 * 
 * 選択した指標のパラメータ設定とリアルタイムバリデーションを提供します。
 */

"use client";

import React, { useState, useEffect } from "react";
import { InputField } from "@/components/common/InputField";
import { SelectField } from "@/components/common/SelectField";

interface SelectedIndicator {
  id: string;
  type: string;
  name: string;
  parameters: Record<string, any>;
  enabled: boolean;
}

interface ParameterDefinition {
  name: string;
  type: string;
  default: any;
  min?: number;
  max?: number;
  description: string;
  options?: Array<{ value: any; label: string }>;
}

interface IndicatorParameterInfo {
  type: string;
  name: string;
  parameters: ParameterDefinition[];
  data_sources: string[];
}

interface ParameterEditorProps {
  selectedIndicators: SelectedIndicator[];
  onParametersChange: (indicatorId: string, parameters: Record<string, any>) => void;
  onIndicatorToggle: (indicatorId: string, enabled: boolean) => void;
}

interface ValidationError {
  indicatorId: string;
  parameterName: string;
  message: string;
}

/**
 * パラメータエディターコンポーネント
 */
const ParameterEditor: React.FC<ParameterEditorProps> = ({
  selectedIndicators,
  onParametersChange,
  onIndicatorToggle,
}) => {
  // 状態管理
  const [indicatorDefinitions, setIndicatorDefinitions] = useState<Record<string, IndicatorParameterInfo>>({});
  const [validationErrors, setValidationErrors] = useState<ValidationError[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  // 指標定義を取得
  useEffect(() => {
    const fetchIndicatorDefinitions = async () => {
      try {
        setLoading(true);
        const response = await fetch("/api/strategy-builder/indicators");
        
        if (!response.ok) {
          throw new Error("指標定義の取得に失敗しました");
        }

        const data = await response.json();
        
        if (data.success) {
          const definitions: Record<string, IndicatorParameterInfo> = {};
          
          // カテゴリ別の指標を統合
          Object.values(data.data.categories).forEach((categoryIndicators: any[]) => {
            categoryIndicators.forEach((indicator) => {
              definitions[indicator.type] = indicator;
            });
          });
          
          setIndicatorDefinitions(definitions);
        }
      } catch (err) {
        console.error("指標定義取得エラー:", err);
        
        // フォールバック用のダミーデータ
        setIndicatorDefinitions({
          SMA: {
            type: "SMA",
            name: "Simple Moving Average",
            parameters: [
              {
                name: "period",
                type: "integer",
                default: 20,
                min: 2,
                max: 200,
                description: "移動平均期間"
              }
            ],
            data_sources: ["close"]
          },
          RSI: {
            type: "RSI",
            name: "Relative Strength Index",
            parameters: [
              {
                name: "period",
                type: "integer",
                default: 14,
                min: 2,
                max: 100,
                description: "RSI計算期間"
              }
            ],
            data_sources: ["close"]
          }
        });
      } finally {
        setLoading(false);
      }
    };

    fetchIndicatorDefinitions();
  }, []);

  // パラメータ値の変更ハンドラー
  const handleParameterChange = (indicatorId: string, parameterName: string, value: any) => {
    const indicator = selectedIndicators.find(ind => ind.id === indicatorId);
    if (!indicator) return;

    const newParameters = {
      ...indicator.parameters,
      [parameterName]: value
    };

    // バリデーション
    validateParameter(indicatorId, parameterName, value);

    onParametersChange(indicatorId, newParameters);
  };

  // パラメータのバリデーション
  const validateParameter = (indicatorId: string, parameterName: string, value: any) => {
    const indicator = selectedIndicators.find(ind => ind.id === indicatorId);
    if (!indicator) return;

    const definition = indicatorDefinitions[indicator.type];
    if (!definition) return;

    const paramDef = definition.parameters.find(p => p.name === parameterName);
    if (!paramDef) return;

    const errors = validationErrors.filter(
      err => !(err.indicatorId === indicatorId && err.parameterName === parameterName)
    );

    // 型チェック
    if (paramDef.type === "integer" && (!Number.isInteger(Number(value)) || isNaN(Number(value)))) {
      errors.push({
        indicatorId,
        parameterName,
        message: "整数値を入力してください"
      });
    } else if (paramDef.type === "float" && isNaN(Number(value))) {
      errors.push({
        indicatorId,
        parameterName,
        message: "数値を入力してください"
      });
    }

    // 範囲チェック
    const numValue = Number(value);
    if (!isNaN(numValue)) {
      if (paramDef.min !== undefined && numValue < paramDef.min) {
        errors.push({
          indicatorId,
          parameterName,
          message: `${paramDef.min}以上の値を入力してください`
        });
      }
      if (paramDef.max !== undefined && numValue > paramDef.max) {
        errors.push({
          indicatorId,
          parameterName,
          message: `${paramDef.max}以下の値を入力してください`
        });
      }
    }

    setValidationErrors(errors);
  };

  // 指標のエラーを取得
  const getIndicatorErrors = (indicatorId: string): ValidationError[] => {
    return validationErrors.filter(err => err.indicatorId === indicatorId);
  };

  // パラメータのエラーを取得
  const getParameterError = (indicatorId: string, parameterName: string): string | null => {
    const error = validationErrors.find(
      err => err.indicatorId === indicatorId && err.parameterName === parameterName
    );
    return error ? error.message : null;
  };

  // パラメータ入力フィールドをレンダリング
  const renderParameterField = (indicator: SelectedIndicator, paramDef: ParameterDefinition) => {
    const currentValue = indicator.parameters[paramDef.name] ?? paramDef.default;
    const error = getParameterError(indicator.id, paramDef.name);

    if (paramDef.options) {
      // セレクトフィールド
      return (
        <SelectField
          key={paramDef.name}
          label={paramDef.description}
          value={String(currentValue)}
          onChange={(value) => handleParameterChange(indicator.id, paramDef.name, value)}
          options={paramDef.options.map(opt => ({
            value: String(opt.value),
            label: opt.label
          }))}
          className={error ? "border-red-500" : ""}
        />
      );
    } else {
      // 入力フィールド
      return (
        <InputField
          key={paramDef.name}
          label={paramDef.description}
          value={currentValue}
          onChange={(value) => handleParameterChange(indicator.id, paramDef.name, value)}
          type={paramDef.type === "integer" || paramDef.type === "float" ? "number" : "text"}
          min={paramDef.min}
          max={paramDef.max}
          step={paramDef.type === "float" ? 0.1 : 1}
          className={error ? "border-red-500" : ""}
        />
      );
    }
  };

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
          <span className="ml-3 text-gray-300">パラメータ定義を読み込み中...</span>
        </div>
      </div>
    );
  }

  if (selectedIndicators.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="text-center py-8">
          <svg className="w-12 h-12 text-gray-500 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
          </svg>
          <p className="text-gray-400 text-lg">指標が選択されていません</p>
          <p className="text-gray-500 text-sm mt-2">
            まず「指標選択」ステップで使用する指標を選択してください
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="space-y-6">
        {selectedIndicators.map((indicator) => {
          const definition = indicatorDefinitions[indicator.type];
          const indicatorErrors = getIndicatorErrors(indicator.id);
          
          if (!definition) {
            return (
              <div key={indicator.id} className="border border-gray-600 rounded-lg p-4">
                <p className="text-red-400">
                  指標 {indicator.type} の定義が見つかりません
                </p>
              </div>
            );
          }

          return (
            <div
              key={indicator.id}
              className={`
                border rounded-lg p-4 transition-all
                ${indicator.enabled ? "border-gray-600 bg-gray-700" : "border-gray-700 bg-gray-800 opacity-60"}
              `}
            >
              {/* 指標ヘッダー */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={indicator.enabled}
                      onChange={(e) => onIndicatorToggle(indicator.id, e.target.checked)}
                      className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                    />
                    <div>
                      <h4 className="font-medium text-white">{indicator.name}</h4>
                      <p className="text-sm text-gray-400">{indicator.type}</p>
                    </div>
                  </label>
                </div>
                
                {indicatorErrors.length > 0 && (
                  <div className="text-red-400">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                    </svg>
                  </div>
                )}
              </div>

              {/* パラメータフィールド */}
              {indicator.enabled && (
                <div className="space-y-4">
                  {definition.parameters.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {definition.parameters.map((paramDef) => (
                        <div key={paramDef.name}>
                          {renderParameterField(indicator, paramDef)}
                          {getParameterError(indicator.id, paramDef.name) && (
                            <p className="text-red-400 text-sm mt-1">
                              {getParameterError(indicator.id, paramDef.name)}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-gray-400 text-sm">
                      この指標にはパラメータがありません
                    </p>
                  )}

                  {/* データソース情報 */}
                  <div className="mt-4 p-3 bg-gray-900 rounded border border-gray-600">
                    <p className="text-xs text-gray-400 mb-1">データソース:</p>
                    <div className="flex flex-wrap gap-1">
                      {definition.data_sources.map((source) => (
                        <span
                          key={source}
                          className="px-2 py-1 bg-gray-700 text-gray-300 text-xs rounded"
                        >
                          {source}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* バリデーションサマリー */}
      {validationErrors.length > 0 && (
        <div className="mt-6 p-4 bg-red-900/30 border border-red-700 rounded-lg">
          <h5 className="text-red-300 font-medium mb-2">パラメータエラー</h5>
          <ul className="text-red-400 text-sm space-y-1">
            {validationErrors.map((error, index) => {
              const indicator = selectedIndicators.find(ind => ind.id === error.indicatorId);
              return (
                <li key={index}>
                  {indicator?.name} - {error.parameterName}: {error.message}
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {/* 設定サマリー */}
      <div className="mt-6 p-4 bg-gray-900 rounded-lg border border-gray-600">
        <h5 className="text-gray-300 font-medium mb-2">設定サマリー</h5>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-gray-400">有効な指標</p>
            <p className="text-white font-medium">
              {selectedIndicators.filter(ind => ind.enabled).length}個
            </p>
          </div>
          <div>
            <p className="text-gray-400">無効な指標</p>
            <p className="text-white font-medium">
              {selectedIndicators.filter(ind => !ind.enabled).length}個
            </p>
          </div>
          <div>
            <p className="text-gray-400">エラー</p>
            <p className={`font-medium ${validationErrors.length > 0 ? "text-red-400" : "text-green-400"}`}>
              {validationErrors.length}個
            </p>
          </div>
          <div>
            <p className="text-gray-400">設定完了</p>
            <p className={`font-medium ${validationErrors.length === 0 && selectedIndicators.some(ind => ind.enabled) ? "text-green-400" : "text-yellow-400"}`}>
              {validationErrors.length === 0 && selectedIndicators.some(ind => ind.enabled) ? "完了" : "未完了"}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ParameterEditor;
