/**
 * 指標期間セレクタコンポーネント
 *
 * テクニカル指標の期間を選択するためのセレクタコンポーネントです。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React from "react";

/**
 * 指標期間セレクタのプロパティ
 */
interface IndicatorPeriodSelectorProps {
  /** 選択された期間 */
  selectedPeriod: number;
  /** 期間変更時のコールバック */
  onPeriodChange: (period: number) => void;
  /** 指標タイプ（期間の選択肢を決定するため） */
  indicatorType: string;
  /** セレクタの無効化フラグ */
  disabled?: boolean;
  /** カスタムクラス名 */
  className?: string;
  /** ラベルを表示するかどうか */
  showLabel?: boolean;
}

/**
 * 指標タイプ別のサポートされている期間の定義
 */
const SUPPORTED_PERIODS: Record<string, number[]> = {
  SMA: [5, 10, 20, 50, 100, 200],
  EMA: [5, 10, 20, 50, 100, 200],
  RSI: [14, 21, 30],
  MACD: [12, 26], // 将来の拡張用
};

/**
 * 期間の説明を取得
 */
const getPeriodDescription = (period: number, indicatorType: string): string => {
  switch (indicatorType) {
    case "SMA":
    case "EMA":
      return `${period}期間移動平均`;
    case "RSI":
      return `${period}期間RSI`;
    case "MACD":
      return `${period}期間MACD`;
    default:
      return `${period}期間`;
  }
};

/**
 * 期間の推奨度を取得（よく使われる期間を強調）
 */
const isRecommendedPeriod = (period: number, indicatorType: string): boolean => {
  const recommendedPeriods: Record<string, number[]> = {
    SMA: [20, 50],
    EMA: [20, 50],
    RSI: [14],
    MACD: [12, 26],
  };
  
  return recommendedPeriods[indicatorType]?.includes(period) || false;
};

/**
 * 指標期間セレクタコンポーネント
 */
const IndicatorPeriodSelector: React.FC<IndicatorPeriodSelectorProps> = ({
  selectedPeriod,
  onPeriodChange,
  indicatorType,
  disabled = false,
  className = "",
  showLabel = true,
}) => {
  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    onPeriodChange(parseInt(event.target.value, 10));
  };

  // 指標タイプに応じた期間の選択肢を取得
  const availablePeriods = SUPPORTED_PERIODS[indicatorType] || [14, 20, 50];

  // 選択された期間が利用可能な期間に含まれていない場合、最初の期間を選択
  React.useEffect(() => {
    if (!availablePeriods.includes(selectedPeriod)) {
      onPeriodChange(availablePeriods[0]);
    }
  }, [indicatorType, selectedPeriod, availablePeriods, onPeriodChange]);

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      {showLabel && (
        <label className="text-sm font-medium text-secondary-700 dark:text-secondary-300">
          期間
        </label>
      )}
      <div className="relative">
        <select
          value={selectedPeriod}
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
          {availablePeriods.map((period) => (
            <option key={period} value={period}>
              {period}
              {isRecommendedPeriod(period, indicatorType) && " (推奨)"}
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
      
      {/* 選択された期間の詳細情報 */}
      <div className="flex items-center gap-2 mt-1">
        <span className="text-xs text-gray-600 dark:text-gray-400">
          {getPeriodDescription(selectedPeriod, indicatorType)}
        </span>
        {isRecommendedPeriod(selectedPeriod, indicatorType) && (
          <span className="text-xs font-semibold text-green-500">
            推奨
          </span>
        )}
      </div>
    </div>
  );
};

export default IndicatorPeriodSelector;
