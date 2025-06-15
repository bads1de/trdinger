/**
 * 時間軸選択コンポーネント（共通化版）
 *
 * 簡略化された時間軸選択コンポーネントです。
 *
 */

"use client";

import React from "react";
import { TimeFrame } from "@/types/strategy";
import { SUPPORTED_TIMEFRAMES } from "@/constants";

/**
 * 時間軸選択コンポーネントのプロパティ
 */
interface TimeFrameSelectorProps {
  /** 現在選択されている時間軸 */
  selectedTimeFrame: TimeFrame;
  /** 時間軸変更時のコールバック */
  onTimeFrameChange: (timeFrame: TimeFrame) => void;
  /** 無効化フラグ */
  disabled?: boolean;
  /** 表示モード */
  mode?: "dropdown" | "buttons" | "compact";
  /** カスタムクラス名 */
  className?: string;
}

/**
 * 時間軸選択コンポーネント（共通化版）
 */
const TimeFrameSelector: React.FC<TimeFrameSelectorProps> = ({
  selectedTimeFrame,
  onTimeFrameChange,
  disabled = false,
  mode = "compact",
  className = "",
}) => {
  if (mode === "compact") {
    return (
      <div className={className}>
        <label className="block text-sm font-medium text-secondary-600 dark:text-secondary-400 mb-2">
          時間軸
        </label>
        <select
          value={selectedTimeFrame}
          onChange={(e) => onTimeFrameChange(e.target.value as TimeFrame)}
          disabled={disabled}
          className="appearance-none bg-gray-800 border border-gray-600 rounded-lg text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200 cursor-pointer px-3 py-2 text-sm min-w-[120px] w-full"
        >
          {SUPPORTED_TIMEFRAMES.map((timeFrame) => (
            <option key={timeFrame.value} value={timeFrame.value}>
              {timeFrame.label}
            </option>
          ))}
        </select>
      </div>
    );
  }

  if (mode === "buttons") {
    return (
      <div className={className}>
        <label className="block text-sm font-medium text-gray-100 mb-3">
          時間軸選択
        </label>
        <div className="flex flex-wrap gap-2">
          {SUPPORTED_TIMEFRAMES.map((timeFrame) => {
            const isSelected = selectedTimeFrame === timeFrame.value;
            return (
              <button
                key={timeFrame.value}
                onClick={() => onTimeFrameChange(timeFrame.value)}
                disabled={disabled}
                title={timeFrame.description}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 min-w-[60px] ${
                  isSelected
                    ? "bg-primary-600 text-white shadow-md border border-primary-600 scale-105"
                    : "bg-gray-800 text-gray-300 border border-gray-600 hover:border-primary-400 hover:bg-gray-700"
                } ${
                  disabled
                    ? "opacity-50 cursor-not-allowed"
                    : "cursor-pointer hover:scale-105"
                }`}
              >
                {timeFrame.label}
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  // その他のモードは後で実装
  return (
    <div className={className}>
      <p>他のモードは実装中です</p>
    </div>
  );
};

export default TimeFrameSelector;
