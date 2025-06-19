/**
 * 指標選択コンポーネント
 * 
 * 全58指標から選択可能な指標選択UI
 */

"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp, Search, X } from "lucide-react";

// 全58指標をカテゴリ別に分類
const INDICATOR_CATEGORIES = {
  trend: {
    name: "トレンド系指標",
    indicators: [
      "SMA", "EMA", "WMA", "HMA", "KAMA", "TEMA", "DEMA", "T3", "MAMA", 
      "ZLEMA", "MACD", "MIDPOINT", "MIDPRICE", "TRIMA", "VWMA"
    ]
  },
  momentum: {
    name: "モメンタム系指標",
    indicators: [
      "RSI", "STOCH", "STOCHRSI", "STOCHF", "CCI", "WILLR", "MOMENTUM", 
      "MOM", "ROC", "ROCP", "ROCR", "ADX", "AROON", "AROONOSC", "MFI", 
      "CMO", "TRIX", "ULTOSC", "BOP", "APO", "PPO", "DX", "ADXR", 
      "PLUS_DI", "MINUS_DI"
    ]
  },
  volatility: {
    name: "ボラティリティ系指標",
    indicators: [
      "BB", "ATR", "NATR", "TRANGE", "KELTNER", "STDDEV", "DONCHIAN"
    ]
  },
  volume: {
    name: "出来高系指標",
    indicators: [
      "OBV", "AD", "ADOSC", "VWAP", "PVT", "EMV"
    ]
  },
  price: {
    name: "価格変換系指標",
    indicators: [
      "AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"
    ]
  },
  other: {
    name: "その他の指標",
    indicators: [
      "PSAR"
    ]
  }
};

// 人気の指標（デフォルト選択用）
const POPULAR_INDICATORS = [
  "SMA", "EMA", "RSI", "MACD", "BB", "STOCH", "CCI", "ADX", 
  "AROON", "MFI", "ATR", "VWAP", "OBV"
];

interface IndicatorSelectorProps {
  selectedIndicators: string[];
  onSelectionChange: (indicators: string[]) => void;
  maxSelection?: number;
}

const IndicatorSelector: React.FC<IndicatorSelectorProps> = ({
  selectedIndicators,
  onSelectionChange,
  maxSelection = 20
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(["trend", "momentum"]) // デフォルトで主要カテゴリを展開
  );

  // 検索フィルタリング
  const filteredCategories = Object.entries(INDICATOR_CATEGORIES).map(([key, category]) => ({
    key,
    ...category,
    indicators: category.indicators.filter(indicator =>
      indicator.toLowerCase().includes(searchTerm.toLowerCase())
    )
  })).filter(category => category.indicators.length > 0);

  const toggleIndicator = (indicator: string) => {
    const newSelection = selectedIndicators.includes(indicator)
      ? selectedIndicators.filter(i => i !== indicator)
      : selectedIndicators.length < maxSelection
        ? [...selectedIndicators, indicator]
        : selectedIndicators;
    
    onSelectionChange(newSelection);
  };

  const toggleCategory = (categoryKey: string) => {
    const newExpanded = new Set(expandedCategories);
    if (newExpanded.has(categoryKey)) {
      newExpanded.delete(categoryKey);
    } else {
      newExpanded.add(categoryKey);
    }
    setExpandedCategories(newExpanded);
  };

  const selectPopularIndicators = () => {
    const newSelection = POPULAR_INDICATORS.slice(0, maxSelection);
    onSelectionChange(newSelection);
  };

  const clearSelection = () => {
    onSelectionChange([]);
  };

  const selectAllInCategory = (categoryIndicators: string[]) => {
    const availableSlots = maxSelection - selectedIndicators.length;
    const newIndicators = categoryIndicators
      .filter(indicator => !selectedIndicators.includes(indicator))
      .slice(0, availableSlots);
    
    onSelectionChange([...selectedIndicators, ...newIndicators]);
  };

  return (
    <div className="space-y-4">
      {/* ヘッダー */}
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-white">
          利用指標選択 ({selectedIndicators.length}/{maxSelection})
        </label>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={selectPopularIndicators}
            className="text-xs px-2 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            人気指標を選択
          </button>
          <button
            type="button"
            onClick={clearSelection}
            className="text-xs px-2 py-1 bg-gray-600 text-white rounded hover:bg-gray-700"
          >
            クリア
          </button>
        </div>
      </div>

      {/* 選択済み指標の表示 */}
      {selectedIndicators.length > 0 && (
        <div className="bg-secondary-800 rounded-lg p-3">
          <div className="flex flex-wrap gap-2">
            {selectedIndicators.map(indicator => (
              <span
                key={indicator}
                className="inline-flex items-center gap-1 px-2 py-1 bg-blue-600 text-white text-xs rounded"
              >
                {indicator}
                <button
                  type="button"
                  onClick={() => toggleIndicator(indicator)}
                  className="hover:bg-blue-700 rounded"
                >
                  <X size={12} />
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 展開/折りたたみボタン */}
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 bg-secondary-800 rounded-lg hover:bg-secondary-700 transition-colors"
      >
        <span className="text-white">指標を選択 (全58指標対応)</span>
        {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
      </button>

      {/* 指標選択パネル */}
      {isExpanded && (
        <div className="bg-secondary-800 rounded-lg p-4 space-y-4">
          {/* 検索バー */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-secondary-400" size={16} />
            <input
              type="text"
              placeholder="指標を検索..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-secondary-700 border border-secondary-600 rounded-lg text-white placeholder-secondary-400 focus:outline-none focus:border-blue-500"
            />
          </div>

          {/* カテゴリ別指標リスト */}
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {filteredCategories.map(category => (
              <div key={category.key} className="border border-secondary-600 rounded-lg">
                <button
                  type="button"
                  onClick={() => toggleCategory(category.key)}
                  className="w-full flex items-center justify-between p-3 hover:bg-secondary-700 transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-white font-medium">{category.name}</span>
                    <span className="text-xs text-secondary-400">
                      ({category.indicators.length}個)
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        selectAllInCategory(category.indicators);
                      }}
                      className="text-xs px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700"
                    >
                      全選択
                    </button>
                    {expandedCategories.has(category.key) ? 
                      <ChevronUp size={16} /> : <ChevronDown size={16} />
                    }
                  </div>
                </button>

                {expandedCategories.has(category.key) && (
                  <div className="p-3 pt-0">
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                      {category.indicators.map(indicator => (
                        <label
                          key={indicator}
                          className="flex items-center gap-2 p-2 hover:bg-secondary-600 rounded cursor-pointer"
                        >
                          <input
                            type="checkbox"
                            checked={selectedIndicators.includes(indicator)}
                            onChange={() => toggleIndicator(indicator)}
                            disabled={
                              !selectedIndicators.includes(indicator) && 
                              selectedIndicators.length >= maxSelection
                            }
                            className="rounded border-secondary-500 text-blue-600 focus:ring-blue-500"
                          />
                          <span className="text-sm text-white">{indicator}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* 統計情報 */}
          <div className="text-xs text-secondary-400 pt-2 border-t border-secondary-600">
            選択済み: {selectedIndicators.length}個 | 
            利用可能: {Object.values(INDICATOR_CATEGORIES).reduce((sum, cat) => sum + cat.indicators.length, 0)}個 |
            残り: {maxSelection - selectedIndicators.length}個選択可能
          </div>
        </div>
      )}
    </div>
  );
};

export default IndicatorSelector;
