/**
 * コンパクト通貨ペア選択コンポーネント
 *
 * データ設定セクション用のコンパクトな通貨ペア選択UI
 */

"use client";

import React from "react";
import { TradingPair } from "@/types/strategy";
import { SUPPORTED_TRADING_PAIRS, getTradingPairIcon } from "@/constants";

interface CompactSymbolSelectorProps {
  symbols?: TradingPair[];
  selectedSymbol: string;
  onSymbolChange: (symbol: string) => void;
  loading?: boolean;
  disabled?: boolean;
}

const CompactSymbolSelector: React.FC<CompactSymbolSelectorProps> = ({
  symbols = SUPPORTED_TRADING_PAIRS,
  selectedSymbol,
  onSymbolChange,
  loading = false,
  disabled = false,
}) => {
  return (
    <div className="relative">
      <select
        value={selectedSymbol}
        onChange={(e) => onSymbolChange(e.target.value)}
        disabled={disabled || loading}
        className={`
          appearance-none bg-gray-800 border border-gray-600 rounded-lg
          pl-8 pr-8 py-2 text-sm font-medium text-white
          focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent
          transition-all duration-200 min-w-[140px]
          ${disabled || loading
            ? "opacity-50 cursor-not-allowed"
            : "hover:border-primary-400 cursor-pointer"
          }
        `}
      >
        {loading ? (
          <option value="">読み込み中...</option>
        ) : (
          symbols.map((symbol) => (
            <option key={symbol.symbol} value={symbol.symbol}>
              {symbol.symbol}
            </option>
          ))
        )}
      </select>

      {/* アイコンとドロップダウン矢印 */}
      <div className="absolute inset-y-0 left-2 flex items-center pointer-events-none">
        <span className="text-lg">{getTradingPairIcon(selectedSymbol)}</span>
      </div>

      <div className="absolute inset-y-0 right-2 flex items-center pointer-events-none">
        {loading ? (
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-500"></div>
        ) : (
          <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        )}
      </div>
    </div>
  );
};

export default CompactSymbolSelector;
