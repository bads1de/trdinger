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
import { TimeFrame } from '@/types/strategy';
import { SUPPORTED_TIMEFRAMES } from '@/constants';

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
 * 時間軸選択コンポーネント
 */
const TimeFrameSelector: React.FC<TimeFrameSelectorProps> = ({
  selectedTimeFrame,
  onTimeFrameChange,
  disabled = false,
}) => {
  return (
    <div className="space-y-3">
      <label className="label-enterprise flex items-center gap-2">
        <svg className="w-4 h-4 text-primary-600 dark:text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
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
              className={`
                group relative px-4 py-2.5 text-sm font-medium rounded-enterprise transition-all duration-200
                ${
                  isSelected
                    ? 'bg-primary-600 text-white shadow-enterprise border border-primary-600 scale-105'
                    : 'bg-white dark:bg-secondary-800 text-secondary-700 dark:text-secondary-300 border border-secondary-300 dark:border-secondary-600 hover:border-primary-400 dark:hover:border-primary-500 hover:bg-primary-50 dark:hover:bg-primary-900/20'
                }
                ${
                  disabled
                    ? 'opacity-50 cursor-not-allowed'
                    : 'cursor-pointer hover:shadow-enterprise-md hover:scale-105'
                }
                focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2
                min-w-[60px] flex items-center justify-center
              `}
            >
              {/* アクティブインジケーター */}
              {isSelected && (
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-accent-500 rounded-full border-2 border-white dark:border-secondary-800 animate-pulse"></div>
              )}

              {/* ホバーエフェクト */}
              <div className={`
                absolute inset-0 rounded-enterprise transition-opacity duration-200
                ${
                  isSelected
                    ? 'bg-gradient-to-r from-primary-600 to-primary-700 opacity-100'
                    : 'bg-gradient-to-r from-primary-500 to-accent-500 opacity-0 group-hover:opacity-10'
                }
              `}></div>

              <span className="relative z-10">{timeFrame.label}</span>
            </button>
          );
        })}
      </div>

      {/* 選択中の時間軸情報 */}
      <div className="mt-3 p-3 bg-secondary-50 dark:bg-secondary-800/50 rounded-enterprise border border-secondary-200 dark:border-secondary-700">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-primary-500 rounded-full animate-pulse"></div>
          <span className="text-sm font-medium text-secondary-700 dark:text-secondary-300">
            選択中: {SUPPORTED_TIMEFRAMES.find(tf => tf.value === selectedTimeFrame)?.description}
          </span>
        </div>
        <p className="text-xs text-secondary-500 dark:text-secondary-400 mt-1 ml-4">
          この時間軸でチャートデータを表示します
        </p>
      </div>

      {/* ヘルプテキスト */}
      <p className="text-xs text-secondary-500 dark:text-secondary-400 flex items-center gap-1">
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        短期分析には小さい時間軸、長期分析には大きい時間軸をおすすめします
      </p>
    </div>
  );
};

export default TimeFrameSelector;
