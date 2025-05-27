/**
 * OHLCVデータ一括取得・保存ボタンコンポーネント
 *
 * 全ての取引ペアと全ての時間軸でOHLCVデータを一括取得し、データベースに保存するためのUIコンポーネントです。
 * ダークモード対応、状態管理、エラーハンドリング、確認ダイアログを含みます。
 *
 * @author Trdinger Development Team
 * @version 2.0.0
 */

'use client';

import React, { useState } from 'react';
import { BulkOHLCVCollectionResult } from '@/types/strategy';
import { SUPPORTED_TRADING_PAIRS, SUPPORTED_TIMEFRAMES } from '@/constants';

/**
 * OHLCVDataCollectionButtonコンポーネントのプロパティ
 */
interface OHLCVDataCollectionButtonProps {
  /** データ収集開始時のコールバック */
  onBulkCollectionStart?: (result: BulkOHLCVCollectionResult) => void;
  /** データ収集エラー時のコールバック */
  onCollectionError?: (error: string) => void;
  /** ボタンの無効化フラグ */
  disabled?: boolean;
  /** カスタムクラス名 */
  className?: string;
}

/**
 * ボタンの状態を表す列挙型
 */
type ButtonState = 'idle' | 'loading' | 'success' | 'error';

/**
 * OHLCVデータ一括取得・保存ボタンコンポーネント
 */
const OHLCVDataCollectionButton: React.FC<OHLCVDataCollectionButtonProps> = ({
  onBulkCollectionStart,
  onCollectionError,
  disabled = false,
  className = '',
}) => {
  const [buttonState, setButtonState] = useState<ButtonState>('idle');
  const [lastResult, setLastResult] = useState<BulkOHLCVCollectionResult | null>(null);

  // 総タスク数を計算
  const totalTasks = SUPPORTED_TRADING_PAIRS.length * SUPPORTED_TIMEFRAMES.length;

  /**
   * OHLCVデータ一括収集を実行
   */
  const handleCollectData = async () => {
    // 確認ダイアログ
    const confirmed = window.confirm(
      `全ての取引ペア（${SUPPORTED_TRADING_PAIRS.length}個）と全ての時間軸（${SUPPORTED_TIMEFRAMES.length}個）でOHLCVデータを取得します。\n` +
      `合計${totalTasks}個のタスクが実行されます。\n\n` +
      'この処理には時間がかかる場合があります。続行しますか？'
    );

    if (!confirmed) {
      return;
    }

    try {
      setButtonState('loading');
      setLastResult(null);

      // API リクエスト
      const response = await fetch('/api/data/ohlcv/bulk', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const result: BulkOHLCVCollectionResult = await response.json();

      if (response.ok && result.success) {
        setButtonState('success');
        setLastResult(result);
        onBulkCollectionStart?.(result);

        // 10秒後に通常状態に戻す
        setTimeout(() => {
          setButtonState('idle');
        }, 10000);
      } else {
        setButtonState('error');
        const errorMessage = result.message || '一括データ収集に失敗しました';
        onCollectionError?.(errorMessage);

        // 10秒後に通常状態に戻す
        setTimeout(() => {
          setButtonState('idle');
        }, 10000);
      }
    } catch (error) {
      setButtonState('error');
      const errorMessage = '一括データ収集中にエラーが発生しました';
      onCollectionError?.(errorMessage);

      // 10秒後に通常状態に戻す
      setTimeout(() => {
        setButtonState('idle');
      }, 10000);
    }
  };



  /**
   * ボタンのテキストを取得
   */
  const getButtonText = (): string => {
    switch (buttonState) {
      case 'loading':
        return '一括取得・保存中...';
      case 'success':
        return '一括取得・保存開始';
      case 'error':
        return 'エラーが発生しました';
      default:
        return '全データ一括取得・保存';
    }
  };

  /**
   * ボタンのアイコンを取得
   */
  const getButtonIcon = () => {
    switch (buttonState) {
      case 'loading':
        return (
          <div
            className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"
            data-testid="loading-spinner"
          />
        );
      case 'success':
        return (
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            data-testid="success-icon"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M5 13l4 4L19 7"
            />
          </svg>
        );
      case 'error':
        return (
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            data-testid="error-icon"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        );
      default:
        return (
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
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
            />
          </svg>
        );
    }
  };

  /**
   * ボタンのスタイルクラスを取得
   */
  const getButtonClasses = (): string => {
    const baseClasses = `
      inline-flex items-center gap-3 px-6 py-3 rounded-enterprise-lg
      font-semibold text-base transition-all duration-200
      focus:outline-none focus:ring-2 focus:ring-offset-2
      disabled:opacity-50 disabled:cursor-not-allowed
      shadow-lg hover:shadow-xl
    `;

    switch (buttonState) {
      case 'loading':
        return `${baseClasses}
          bg-primary-600 dark:bg-primary-600 text-white
          cursor-not-allowed
        `;
      case 'success':
        return `${baseClasses}
          bg-success-600 dark:bg-success-600 text-white
          focus:ring-success-500
        `;
      case 'error':
        return `${baseClasses}
          bg-error-600 dark:bg-error-600 text-white
          focus:ring-error-500
        `;
      default:
        return `${baseClasses}
          bg-gradient-to-r from-primary-600 to-primary-700
          dark:from-primary-600 dark:to-primary-700 text-white
          hover:from-primary-700 hover:to-primary-800
          dark:hover:from-primary-700 dark:hover:to-primary-800
          focus:ring-primary-500
        `;
    }
  };

  const isButtonDisabled = disabled || buttonState === 'loading';

  return (
    <div className={`flex flex-col gap-3 ${className}`}>
      {/* メインボタン */}
      <button
        onClick={handleCollectData}
        disabled={isButtonDisabled}
        className={getButtonClasses()}
        title={`全ての取引ペア（${SUPPORTED_TRADING_PAIRS.length}個）と全ての時間軸（${SUPPORTED_TIMEFRAMES.length}個）のOHLCVデータを一括取得・保存`}
      >
        {getButtonIcon()}
        <span>{getButtonText()}</span>
      </button>

      {/* 説明テキスト */}
      <div className="text-sm text-gray-400 dark:text-gray-400">
        <div className="flex items-center gap-2 mb-1">
          <span className="inline-block w-2 h-2 bg-primary-500 rounded-full"></span>
          <span>全ての取引ペア: {SUPPORTED_TRADING_PAIRS.length}個</span>
        </div>
        <div className="flex items-center gap-2 mb-1">
          <span className="inline-block w-2 h-2 bg-primary-500 rounded-full"></span>
          <span>全ての時間軸: {SUPPORTED_TIMEFRAMES.length}個</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block w-2 h-2 bg-warning-500 rounded-full"></span>
          <span className="font-medium">合計タスク数: {totalTasks}個</span>
        </div>
      </div>

      {/* 結果メッセージ表示 */}
      {lastResult && buttonState === 'success' && (
        <div className="text-sm text-success-400 dark:text-success-400 bg-gray-900 dark:bg-gray-900 p-3 rounded-lg border border-gray-700 dark:border-gray-700">
          <div className="font-medium mb-1">{lastResult.message}</div>
          <div className="text-xs">
            {lastResult.total_tasks}個のタスクを開始しました
          </div>
        </div>
      )}
    </div>
  );
};

export default OHLCVDataCollectionButton;
