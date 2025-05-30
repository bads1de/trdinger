/**
 * 指標タイプセレクタコンポーネント
 *
 * テクニカル指標のタイプを選択するためのセレクタコンポーネントです。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React from "react";

/**
 * 指標タイプセレクタのプロパティ
 */
interface IndicatorTypeSelectorProps {
  /** 選択された指標タイプ */
  selectedIndicatorType: string;
  /** 指標タイプ変更時のコールバック */
  onIndicatorTypeChange: (indicatorType: string) => void;
  /** セレクタの無効化フラグ */
  disabled?: boolean;
  /** カスタムクラス名 */
  className?: string;
  /** ラベルを表示するかどうか */
  showLabel?: boolean;
}

/**
 * サポートされている指標タイプの定義
 */
const SUPPORTED_INDICATOR_TYPES = [
  {
    value: "SMA",
    label: "SMA",
    description: "単純移動平均",
    color: "text-blue-400",
  },
  {
    value: "EMA",
    label: "EMA",
    description: "指数移動平均",
    color: "text-green-400",
  },
  {
    value: "RSI",
    label: "RSI",
    description: "相対力指数",
    color: "text-purple-400",
  },
];

/**
 * 指標タイプセレクタコンポーネント
 */
const IndicatorTypeSelector: React.FC<IndicatorTypeSelectorProps> = ({
  selectedIndicatorType,
  onIndicatorTypeChange,
  disabled = false,
  className = "",
  showLabel = true,
}) => {
  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    onIndicatorTypeChange(event.target.value);
  };

  const selectedIndicator = SUPPORTED_INDICATOR_TYPES.find(
    (indicator) => indicator.value === selectedIndicatorType
  );

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      {showLabel && (
        <label className="text-sm font-medium text-secondary-700 dark:text-secondary-300">
          指標タイプ
        </label>
      )}
      <div className="relative">
        <select
          value={selectedIndicatorType}
          onChange={handleChange}
          disabled={disabled}
          className={`
            w-full px-3 py-2 text-sm
            bg-white dark:bg-gray-800
            border border-gray-300 dark:border-gray-600
            rounded-lg shadow-sm
            text-gray-900 dark:text-gray-100
            focus:ring-2 focus:ring-primary-500 focus:border-primary-500
            disabled:bg-gray-100 dark:disabled:bg-gray-700
            disabled:text-gray-400 dark:disabled:text-gray-500
            disabled:cursor-not-allowed
            transition-colors duration-200
            appearance-none cursor-pointer
          `}
        >
          {SUPPORTED_INDICATOR_TYPES.map((indicator) => (
            <option key={indicator.value} value={indicator.value}>
              {indicator.label} - {indicator.description}
            </option>
          ))}
        </select>
        
        {/* カスタム矢印アイコン */}
        <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
          <svg
            className="w-4 h-4 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </div>
      </div>
      
      {/* 選択された指標の詳細情報 */}
      {selectedIndicator && (
        <div className="flex items-center gap-2 mt-1">
          <span className={`text-xs font-semibold ${selectedIndicator.color}`}>
            {selectedIndicator.label}
          </span>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            {selectedIndicator.description}
          </span>
        </div>
      )}
    </div>
  );
};

export default IndicatorTypeSelector;
