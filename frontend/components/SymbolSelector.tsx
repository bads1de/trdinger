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

import React, { useState } from 'react';
import { TradingPair } from '@/types/strategy';
import {
  SUPPORTED_TRADING_PAIRS,
  categorizeTradingPairs,
  getTradingPairIcon,
  getMarketType
} from '@/constants';

/**
 * 通貨ペア選択コンポーネントのプロパティ
 */
interface SymbolSelectorProps {
  /** 利用可能な通貨ペアのリスト（オプション、デフォルトで定数を使用） */
  symbols?: TradingPair[];
  /** 現在選択されている通貨ペア */
  selectedSymbol: string;
  /** 通貨ペア変更時のコールバック */
  onSymbolChange: (symbol: string) => void;
  /** ローディング状態 */
  loading?: boolean;
  /** 無効化フラグ */
  disabled?: boolean;
  /** カテゴリ別表示を有効にするか */
  showCategories?: boolean;
}

/**
 * 通貨ペア選択コンポーネント
 */
const SymbolSelector: React.FC<SymbolSelectorProps> = ({
  symbols = SUPPORTED_TRADING_PAIRS,
  selectedSymbol,
  onSymbolChange,
  loading = false,
  disabled = false,
  showCategories = true,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  // カテゴリ別に分類
  const categorizedPairs = categorizeTradingPairs(symbols);

  // 選択されたペアの情報を取得
  const selectedPair = symbols.find(pair => pair.symbol === selectedSymbol);

  const handleSymbolSelect = (symbol: string) => {
    onSymbolChange(symbol);
    setIsOpen(false);
  };
  if (showCategories) {
    // カテゴリ別表示版
    return (
      <div className="space-y-3">
        <label className="label-enterprise flex items-center gap-2">
          <svg className="w-4 h-4 text-primary-600 dark:text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
          </svg>
          通貨ペア選択
          {loading && (
            <div className="animate-spin rounded-full h-3 w-3 border-b border-primary-600"></div>
          )}
        </label>

        <div className="relative">
          {/* 選択されたペアの表示 */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            disabled={disabled || loading}
            className={`
              w-full min-w-[320px] px-4 py-3 text-left bg-white dark:bg-gray-800
              border border-gray-300 dark:border-gray-600 rounded-enterprise
              flex items-center justify-between gap-3
              ${
                disabled || loading
                  ? 'opacity-50 cursor-not-allowed'
                  : 'hover:border-primary-400 dark:hover:border-primary-500 hover:shadow-enterprise-md cursor-pointer'
              }
              transition-all duration-200
            `}
          >
            <div className="flex items-center gap-3">
              <span className="text-xl">{getTradingPairIcon(selectedSymbol)}</span>
              <div>
                <div className="font-medium text-gray-900 dark:text-gray-100">
                  {selectedSymbol}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {selectedPair?.name || getMarketType(selectedSymbol)}
                </div>
              </div>
            </div>

            {loading ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-600"></div>
            ) : (
              <svg
                className={`h-5 w-5 text-gray-400 transition-transform duration-200 ${
                  isOpen ? 'rotate-180' : ''
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            )}
          </button>

          {/* ドロップダウンメニュー */}
          {isOpen && !loading && (
            <div className="absolute z-50 w-full mt-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-enterprise shadow-enterprise-lg max-h-80 overflow-y-auto">
              {/* スポットペア */}
              {categorizedPairs.spot.length > 0 && (
                <div>
                  <div className="px-4 py-2 text-xs font-semibold text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600">
                    💰 スポット取引
                  </div>
                  {categorizedPairs.spot.map((pair) => (
                    <button
                      key={pair.symbol}
                      onClick={() => handleSymbolSelect(pair.symbol)}
                      className={`
                        w-full px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-gray-700
                        flex items-center gap-3 transition-colors duration-150
                        ${
                          selectedSymbol === pair.symbol
                            ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
                            : 'text-gray-900 dark:text-gray-100'
                        }
                      `}
                    >
                      <span className="text-lg">{getTradingPairIcon(pair.symbol)}</span>
                      <div className="flex-1">
                        <div className="font-medium">{pair.symbol}</div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {pair.name}
                        </div>
                      </div>
                      <div className="text-xs px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded">
                        スポット
                      </div>
                    </button>
                  ))}
                </div>
              )}

              {/* 永続契約 */}
              {categorizedPairs.perpetual.length > 0 && (
                <div>
                  <div className="px-4 py-2 text-xs font-semibold text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600">
                    ⚡ 永続契約
                  </div>
                  {categorizedPairs.perpetual.map((pair) => (
                    <button
                      key={pair.symbol}
                      onClick={() => handleSymbolSelect(pair.symbol)}
                      className={`
                        w-full px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-gray-700
                        flex items-center gap-3 transition-colors duration-150
                        ${
                          selectedSymbol === pair.symbol
                            ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
                            : 'text-gray-900 dark:text-gray-100'
                        }
                      `}
                    >
                      <span className="text-lg">{getTradingPairIcon(pair.symbol)}</span>
                      <div className="flex-1">
                        <div className="font-medium">{pair.symbol}</div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {pair.name}
                        </div>
                      </div>
                      <div className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded">
                        {getMarketType(pair.symbol)}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* ヘルプテキスト */}
        <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
          <div className="flex items-center gap-1">
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            分析したい仮想通貨ペアを選択してください
          </div>
          <div className="text-primary-600 dark:text-primary-400">
            {symbols.length}ペア利用可能
          </div>
        </div>
      </div>
    );
  }

  // シンプル表示版（従来のセレクトボックス）
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
                  {getTradingPairIcon(symbol.symbol)} {symbol.symbol} - {symbol.name}
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

// クリック外でドロップダウンを閉じるためのフック
// このコンポーネントを使用する場合は、親コンポーネントでuseEffectを使用してクリックイベントをリスンしてください
