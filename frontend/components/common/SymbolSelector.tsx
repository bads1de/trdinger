/**
 * 通貨ペア選択コンポーネント（共通化版）
 *
 * 簡略化された通貨ペア選択コンポーネントです。
 *
 * @author Trdinger Development Team
 * @version 2.0.0
 */

"use client";

import React from "react";
import { TradingPair } from "@/types/strategy";

/**
 * 通貨ペア選択コンポーネントのプロパティ
 */
interface SymbolSelectorProps {
  /** 利用可能な通貨ペアのリスト */
  symbols?: TradingPair[];
  /** 現在選択されている通貨ペア */
  selectedSymbol: string;
  /** 通貨ペア変更時のコールバック */
  onSymbolChange: (symbol: string) => void;
  /** ローディング状態 */
  loading?: boolean;
  /** 無効化フラグ */
  disabled?: boolean;
  /** 表示モード */
  mode?: "dropdown" | "buttons" | "compact";
  /** カテゴリ別表示を有効にするか */
  showCategories?: boolean;
  /** 検索機能を有効にするか */
  enableSearch?: boolean;
  /** カスタムクラス名 */
  className?: string;
}

/**
 * 通貨ペア選択コンポーネント（共通化版）
 */
const SymbolSelector: React.FC<SymbolSelectorProps> = ({
  symbols = [],
  selectedSymbol,
  onSymbolChange,
  loading = false,
  disabled = false,
  mode = "compact",
  className = "",
}) => {
  if (mode === "compact") {
    return (
      <div className={className}>
        <label className="block text-sm font-medium text-secondary-600 dark:text-secondary-400 mb-2">
          通貨ペア
        </label>
        <select
          value={selectedSymbol}
          onChange={(e) => onSymbolChange(e.target.value)}
          disabled={disabled || loading}
          className="appearance-none bg-gray-800 border border-gray-600 rounded-lg text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200 cursor-pointer px-3 py-2 text-sm min-w-[120px] w-full"
        >
          {loading ? (
            <option value="">読み込み中...</option>
          ) : symbols.length === 0 ? (
            <option value="">利用可能な通貨ペアがありません</option>
          ) : (
            symbols.map((symbol) => (
              <option key={symbol.symbol} value={symbol.symbol}>
                {symbol.symbol}
              </option>
            ))
          )}
        </select>
      </div>
    );
  }

  // その他のモードは後で実装
  return (
    <div className={className}>
      <p>他のモードは実装中です</p>
    </div>
  );
};

export default SymbolSelector;
