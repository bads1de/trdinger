/**
 * コンパクト時間軸選択コンポーネント
 * 
 * データ設定セクション用のコンパクトな時間軸選択UI
 */

"use client";

import React from "react";
import { TimeFrame } from "@/types/strategy";
import { SUPPORTED_TIMEFRAMES } from "@/constants";

interface CompactTimeFrameSelectorProps {
  selectedTimeFrame: TimeFrame;
  onTimeFrameChange: (timeFrame: TimeFrame) => void;
  disabled?: boolean;
}

const CompactTimeFrameSelector: React.FC<CompactTimeFrameSelectorProps> = ({
  selectedTimeFrame,
  onTimeFrameChange,
  disabled = false,
}) => {
  return (
    <div className="flex gap-1">
      {SUPPORTED_TIMEFRAMES.map((timeFrame) => {
        const isSelected = selectedTimeFrame === timeFrame.value;
        
        return (
          <button
            key={timeFrame.value}
            onClick={() => onTimeFrameChange(timeFrame.value)}
            disabled={disabled}
            title={timeFrame.description}
            className={`
              px-3 py-2 text-xs font-medium rounded-md transition-all duration-200
              min-w-[40px] h-[36px] flex items-center justify-center
              ${isSelected
                ? "bg-primary-600 text-white border border-primary-600 shadow-md"
                : "bg-gray-800 text-gray-300 border border-gray-600 hover:border-primary-400 hover:bg-gray-700"
              }
              ${disabled
                ? "opacity-50 cursor-not-allowed"
                : "cursor-pointer"
              }
              focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-1
            `}
          >
            {timeFrame.label}
          </button>
        );
      })}
    </div>
  );
};

export default CompactTimeFrameSelector;
