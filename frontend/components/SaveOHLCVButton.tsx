/**
 * OHLCVデータ保存ボタンコンポーネント
 * 
 * 現在表示中の銘柄のOHLCVデータをデータベースに保存するためのボタンです。
 * ローディング状態、成功/失敗メッセージの表示機能を含みます。
 * 
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React, { useState } from 'react';

/**
 * 保存ボタンのプロパティ
 */
interface SaveOHLCVButtonProps {
  /** 取引ペアシンボル */
  symbol: string;
  /** 時間軸 */
  timeframe: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d';
  /** 取得件数制限 */
  limit?: number;
  /** ボタンの無効化状態 */
  disabled?: boolean;
  /** カスタムクラス名 */
  className?: string;
  /** 保存完了時のコールバック */
  onSaveComplete?: (result: SaveResult) => void;
}

/**
 * 保存結果の型定義
 */
interface SaveResult {
  success: boolean;
  records_saved?: number;
  message: string;
}

/**
 * メッセージの種類
 */
type MessageType = 'success' | 'error' | 'info';

/**
 * メッセージ状態の型定義
 */
interface MessageState {
  type: MessageType;
  text: string;
  visible: boolean;
}

/**
 * OHLCVデータ保存ボタンコンポーネント
 */
const SaveOHLCVButton: React.FC<SaveOHLCVButtonProps> = ({
  symbol,
  timeframe,
  limit = 100,
  disabled = false,
  className = '',
  onSaveComplete,
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState<MessageState>({
    type: 'success',
    text: '',
    visible: false,
  });

  /**
   * メッセージを表示する
   */
  const showMessage = (type: MessageType, text: string) => {
    setMessage({ type, text, visible: true });
    
    // 5秒後にメッセージを自動で隠す
    setTimeout(() => {
      setMessage(prev => ({ ...prev, visible: false }));
    }, 5000);
  };

  /**
   * OHLCVデータ保存処理
   */
  const handleSave = async () => {
    if (disabled || isLoading) return;

    setIsLoading(true);
    setMessage(prev => ({ ...prev, visible: false }));

    try {
      const response = await fetch('/api/v1/market-data/save-ohlcv', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol,
          timeframe,
          limit,
        }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        const result: SaveResult = {
          success: true,
          records_saved: data.records_saved,
          message: data.message,
        };

        // 保存件数に応じてメッセージタイプを決定
        if (data.records_saved === 0) {
          showMessage('info', data.message);
        } else {
          showMessage('success', data.message);
        }

        onSaveComplete?.(result);
      } else {
        const errorMessage = data.detail?.message || data.message || 'データの保存に失敗しました';
        showMessage('error', errorMessage);
        
        onSaveComplete?.({
          success: false,
          message: errorMessage,
        });
      }
    } catch (error) {
      const errorMessage = 'ネットワークエラーが発生しました。しばらく後に再試行してください。';
      showMessage('error', errorMessage);
      
      onSaveComplete?.({
        success: false,
        message: errorMessage,
      });
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * メッセージアイコンを取得
   */
  const getMessageIcon = (type: MessageType) => {
    switch (type) {
      case 'success':
        return (
          <svg
            data-testid="success-icon"
            className="w-5 h-5 text-success-500"
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
        );
      case 'error':
        return (
          <svg
            data-testid="error-icon"
            className="w-5 h-5 text-error-500"
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
        );
      case 'info':
        return (
          <svg
            data-testid="info-icon"
            className="w-5 h-5 text-warning-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        );
    }
  };

  /**
   * メッセージの背景色クラスを取得
   */
  const getMessageBgClass = (type: MessageType) => {
    switch (type) {
      case 'success':
        return 'bg-success-50 dark:bg-success-900/20 border-success-200 dark:border-success-800';
      case 'error':
        return 'bg-error-50 dark:bg-error-900/20 border-error-200 dark:border-error-800';
      case 'info':
        return 'bg-warning-50 dark:bg-warning-900/20 border-warning-200 dark:border-warning-800';
    }
  };

  return (
    <div className="space-y-4">
      {/* 保存ボタン */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
        <button
          onClick={handleSave}
          disabled={disabled || isLoading}
          className={`
            btn-primary group relative overflow-hidden
            ${disabled || isLoading ? 'opacity-50 cursor-not-allowed' : 'hover:shadow-lg'}
            ${className}
          `}
          aria-label="OHLCVデータ保存"
        >
          <div className="flex items-center gap-2">
            {isLoading ? (
              <div
                data-testid="loading-spinner"
                className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"
              />
            ) : (
              <svg
                className="w-4 h-4 transition-transform duration-200 group-hover:scale-110"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
            )}
            <span className="font-medium">
              {isLoading ? '保存中...' : 'OHLCVデータ保存'}
            </span>
          </div>
        </button>

        {/* 保存対象情報 */}
        <div className="flex items-center gap-2 text-sm text-secondary-600 dark:text-secondary-400">
          <span className="badge-primary">{symbol}</span>
          <span className="badge-secondary">{timeframe}足</span>
          <span className="text-xs">({limit}件)</span>
        </div>
      </div>

      {/* メッセージ表示エリア */}
      {message.visible && (
        <div
          className={`
            enterprise-card p-4 border animate-slide-down
            ${getMessageBgClass(message.type)}
          `}
        >
          <div className="flex items-start gap-3">
            {getMessageIcon(message.type)}
            <div className="flex-1">
              <p className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
                {message.text}
              </p>
            </div>
            <button
              onClick={() => setMessage(prev => ({ ...prev, visible: false }))}
              className="text-secondary-400 hover:text-secondary-600 dark:hover:text-secondary-300 transition-colors"
              aria-label="メッセージを閉じる"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default SaveOHLCVButton;
