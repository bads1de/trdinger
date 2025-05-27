/**
 * OHLCVデータ取得・保存ボタンコンポーネント
 *
 * Bybit取引所からOHLCVデータを取得し、データベースに保存するためのUIコンポーネントです。
 * ダークモード対応、状態管理、エラーハンドリングを含みます。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

'use client';

import React, { useState } from 'react';
import { OHLCVCollectionResult, OHLCVCollectionRequest } from '@/types/strategy';

/**
 * OHLCVDataCollectionButtonコンポーネントのプロパティ
 */
interface OHLCVDataCollectionButtonProps {
  /** 選択された取引ペアシンボル */
  selectedSymbol: string;
  /** 選択された時間軸 */
  timeframe: string;
  /** データ収集開始時のコールバック */
  onCollectionStart?: () => void;
  /** データ収集完了時のコールバック */
  onCollectionComplete?: (result: OHLCVCollectionResult) => void;
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
 * OHLCVデータ取得・保存ボタンコンポーネント
 */
const OHLCVDataCollectionButton: React.FC<OHLCVDataCollectionButtonProps> = ({
  selectedSymbol,
  timeframe,
  onCollectionStart,
  onCollectionComplete,
  onCollectionError,
  disabled = false,
  className = '',
}) => {
  const [buttonState, setButtonState] = useState<ButtonState>('idle');
  const [lastResult, setLastResult] = useState<OHLCVCollectionResult | null>(null);

  /**
   * OHLCVデータ収集を実行
   */
  const handleCollectData = async () => {
    try {
      setButtonState('loading');
      setLastResult(null);
      
      // コールバック実行
      onCollectionStart?.();

      // API リクエスト
      const requestData: OHLCVCollectionRequest = {
        symbol: selectedSymbol,
        timeframe: timeframe,
      };

      const response = await fetch('/api/data/ohlcv', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      const result: OHLCVCollectionResult = await response.json();

      if (response.ok && result.success) {
        setButtonState('success');
        setLastResult(result);
        onCollectionComplete?.(result);
        
        // 3秒後に通常状態に戻す
        setTimeout(() => {
          setButtonState('idle');
        }, 3000);
      } else {
        setButtonState('error');
        const errorMessage = result.message || 'データ収集に失敗しました';
        onCollectionError?.(errorMessage);
        
        // 5秒後に通常状態に戻す
        setTimeout(() => {
          setButtonState('idle');
        }, 5000);
      }
    } catch (error) {
      setButtonState('error');
      const errorMessage = 'データ収集中にエラーが発生しました';
      onCollectionError?.(errorMessage);
      
      // 5秒後に通常状態に戻す
      setTimeout(() => {
        setButtonState('idle');
      }, 5000);
    }
  };

  /**
   * ボタンのテキストを取得
   */
  const getButtonText = (): string => {
    switch (buttonState) {
      case 'loading':
        return '取得・保存中...';
      case 'success':
        return '取得・保存完了';
      case 'error':
        return 'エラーが発生しました';
      default:
        return 'OHLCVデータ取得・保存';
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
            className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"
            data-testid="loading-spinner"
          />
        );
      case 'success':
        return (
          <svg 
            className="w-4 h-4" 
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
            className="w-4 h-4" 
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
            className="w-4 h-4" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" 
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
      inline-flex items-center gap-2 px-4 py-2 rounded-enterprise-md
      font-medium text-sm transition-all duration-200
      focus:outline-none focus:ring-2 focus:ring-offset-2
      disabled:opacity-50 disabled:cursor-not-allowed
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
          bg-primary-600 dark:bg-primary-600 text-white
          hover:bg-primary-700 dark:hover:bg-primary-700
          focus:ring-primary-500
        `;
    }
  };

  const isButtonDisabled = disabled || buttonState === 'loading';

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <button
        onClick={handleCollectData}
        disabled={isButtonDisabled}
        className={getButtonClasses()}
        title={`${selectedSymbol} ${timeframe} のOHLCVデータを取得・保存`}
      >
        {getButtonIcon()}
        <span>{getButtonText()}</span>
      </button>
      
      {/* 結果メッセージ表示 */}
      {lastResult && buttonState === 'success' && (
        <div className="text-xs text-success-600 dark:text-success-400">
          {lastResult.saved_count && lastResult.saved_count > 0 
            ? `${lastResult.saved_count}件のデータを保存しました`
            : lastResult.message
          }
        </div>
      )}
    </div>
  );
};

export default OHLCVDataCollectionButton;
