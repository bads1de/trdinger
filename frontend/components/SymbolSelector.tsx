/**
 * 通貨ペア選択コンポーネント
 *
 * チャートで表示する通貨ペアを選択するためのUIコンポーネントです。
 * ドロップダウン形式で利用可能な通貨ペアから選択できます。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

'use client';

import React from 'react';
import { TradingPair } from '@/types/strategy';

/**
 * 通貨ペア選択コンポーネントのプロパティ
 */
interface SymbolSelectorProps {
  /** 利用可能な通貨ペアのリスト */
  symbols: TradingPair[];
  /** 現在選択されている通貨ペア */
  selectedSymbol: string;
  /** 通貨ペア変更時のコールバック */
  onSymbolChange: (symbol: string) => void;
  /** ローディング状態 */
  loading?: boolean;
  /** 無効化フラグ */
  disabled?: boolean;
}

/**
 * 通貨ペア選択コンポーネント
 */
const SymbolSelector: React.FC<SymbolSelectorProps> = ({
  symbols,
  selectedSymbol,
  onSymbolChange,
  loading = false,
  disabled = false,
}) => {
  return (
    <div className="space-y-3">
      <label
        htmlFor="symbol-select"
        className="label-enterprise flex items-center gap-2"
      >
        <svg className="w-4 h-4 text-primary-600 dark:text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
        </svg>
        通貨ペア選択
        {loading && (
          <div className="animate-spin rounded-full h-3 w-3 border-b border-primary-600"></div>
        )}
      </label>

      <div className="relative group">
        <select
          id="symbol-select"
          value={selectedSymbol}
          onChange={(e) => onSymbolChange(e.target.value)}
          disabled={disabled || loading}
          className={`
            select-enterprise min-w-[280px]
            ${
              disabled || loading
                ? 'opacity-50 cursor-not-allowed'
                : 'hover:border-primary-400 dark:hover:border-primary-500 group-hover:shadow-enterprise-md'
            }
            transition-all duration-200
          `}
        >
          {loading ? (
            <option value="">🔄 データを読み込み中...</option>
          ) : (
            <>
              {symbols.length === 0 && (
                <option value="">⚠️ 利用可能な通貨ペアがありません</option>
              )}
              {symbols.map((symbol) => (
                <option key={symbol.symbol} value={symbol.symbol}>
                  💰 {symbol.symbol} - {symbol.name}
                </option>
              ))}
            </>
          )}
        </select>

        {/* ローディングスピナー */}
        {loading && (
          <div className="absolute right-10 top-1/2 transform -translate-y-1/2">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-600"></div>
          </div>
        )}

        {/* エンタープライズドロップダウンアイコン */}
        {!loading && (
          <div className="absolute right-3 top-1/2 transform -translate-y-1/2 pointer-events-none transition-transform duration-200 group-hover:scale-110">
            <svg
              className="h-5 w-5 text-secondary-400 dark:text-secondary-500"
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
        )}

        {/* フォーカスインジケーター */}
        <div className="absolute inset-0 rounded-enterprise border-2 border-transparent group-focus-within:border-primary-500 pointer-events-none transition-colors duration-200"></div>
      </div>

      {/* ヘルプテキスト */}
      <p className="text-xs text-secondary-500 dark:text-secondary-400 flex items-center gap-1">
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        分析したい仮想通貨ペアを選択してください
      </p>
    </div>
  );
};

export default SymbolSelector;
