/**
 * 時間軸選択コンポーネント
 *
 * チャートの時間軸を選択するためのUIコンポーネントです。
 * ボタン形式で複数の時間軸から選択できます。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

'use client';

import React from 'react';
import { TimeFrame, TimeFrameInfo } from '@/types/strategy';

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
}

/**
 * 利用可能な時間軸の定義
 */
const TIME_FRAMES: TimeFrameInfo[] = [
  {
    value: '1m',
    label: '1分',
    description: '1分足チャート',
  },
  {
    value: '5m',
    label: '5分',
    description: '5分足チャート',
  },
  {
    value: '15m',
    label: '15分',
    description: '15分足チャート',
  },
  {
    value: '30m',
    label: '30分',
    description: '30分足チャート',
  },
  {
    value: '1h',
    label: '1時間',
    description: '1時間足チャート',
  },
  {
    value: '4h',
    label: '4時間',
    description: '4時間足チャート',
  },
  {
    value: '1d',
    label: '1日',
    description: '日足チャート',
  },
];

/**
 * 時間軸選択コンポーネント
 */
const TimeFrameSelector: React.FC<TimeFrameSelectorProps> = ({
  selectedTimeFrame,
  onTimeFrameChange,
  disabled = false,
}) => {
  return (
    <div className="flex flex-wrap gap-2">
      <span className="text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center mr-3">
        時間軸:
      </span>
      
      {TIME_FRAMES.map((timeFrame) => {
        const isSelected = selectedTimeFrame === timeFrame.value;
        
        return (
          <button
            key={timeFrame.value}
            onClick={() => onTimeFrameChange(timeFrame.value)}
            disabled={disabled}
            title={timeFrame.description}
            className={`
              px-3 py-1.5 text-sm font-medium rounded-md transition-colors duration-200
              ${
                isSelected
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600'
              }
              ${
                disabled
                  ? 'opacity-50 cursor-not-allowed'
                  : 'cursor-pointer'
              }
              focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
            `}
          >
            {timeFrame.label}
          </button>
        );
      })}
    </div>
  );
};

export default TimeFrameSelector;
