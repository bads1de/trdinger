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
    <div className="flex flex-col sm:flex-row sm:items-center gap-2">
      <label 
        htmlFor="symbol-select"
        className="text-sm font-medium text-gray-700 dark:text-gray-300"
      >
        通貨ペア:
      </label>
      
      <div className="relative">
        <select
          id="symbol-select"
          value={selectedSymbol}
          onChange={(e) => onSymbolChange(e.target.value)}
          disabled={disabled || loading}
          className={`
            block w-full px-3 py-2 text-sm
            bg-white dark:bg-gray-800
            border border-gray-300 dark:border-gray-600
            rounded-md shadow-sm
            text-gray-900 dark:text-gray-100
            focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500
            ${
              disabled || loading
                ? 'opacity-50 cursor-not-allowed'
                : 'cursor-pointer hover:border-gray-400 dark:hover:border-gray-500'
            }
          `}
        >
          {loading ? (
            <option value="">読み込み中...</option>
          ) : (
            <>
              {symbols.length === 0 && (
                <option value="">利用可能な通貨ペアがありません</option>
              )}
              {symbols.map((symbol) => (
                <option key={symbol.symbol} value={symbol.symbol}>
                  {symbol.symbol} - {symbol.name}
                </option>
              ))}
            </>
          )}
        </select>
        
        {/* ローディングスピナー */}
        {loading && (
          <div className="absolute right-8 top-1/2 transform -translate-y-1/2">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
          </div>
        )}
        
        {/* ドロップダウンアイコン */}
        {!loading && (
          <div className="absolute right-3 top-1/2 transform -translate-y-1/2 pointer-events-none">
            <svg
              className="h-4 w-4 text-gray-400"
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
      </div>
    </div>
  );
};

export default SymbolSelector;
