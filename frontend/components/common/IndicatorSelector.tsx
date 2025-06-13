/**
 * 指標選択コンポーネント
 * 
 * 動的に取得した指標リストから選択可能な再利用可能コンポーネント
 */

import React from 'react';
import { useIndicators, useIndicatorCategories, useIndicatorInfo } from '@/hooks/useIndicators';

interface IndicatorSelectorProps {
  selectedIndicators: string[];
  onIndicatorToggle: (indicator: string) => void;
  maxIndicators?: number;
  showCategories?: boolean;
  disabled?: boolean;
  className?: string;
}

const IndicatorSelector: React.FC<IndicatorSelectorProps> = ({
  selectedIndicators,
  onIndicatorToggle,
  maxIndicators,
  showCategories = false,
  disabled = false,
  className = "",
}) => {
  const { indicators, loading: indicatorsLoading, error: indicatorsError } = useIndicators();
  const { categories, loading: categoriesLoading } = useIndicatorCategories();
  const { indicatorInfo, loading: infoLoading } = useIndicatorInfo();

  const isLoading = indicatorsLoading || (showCategories && categoriesLoading) || infoLoading;

  // カテゴリ別表示の場合
  if (showCategories && categories) {
    return (
      <div className={`space-y-4 ${className}`}>
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-2 text-gray-600">指標リストを読み込み中...</span>
          </div>
        ) : indicatorsError ? (
          <div className="bg-red-50 border border-red-200 rounded-md p-4">
            <p className="text-red-800 text-sm">
              指標リストの読み込みに失敗しました。
            </p>
          </div>
        ) : (
          Object.entries(categories).map(([category, categoryIndicators]) => (
            <div key={category} className="space-y-2">
              <h4 className="font-medium text-gray-700 capitalize">
                {getCategoryDisplayName(category)} ({categoryIndicators.length}個)
              </h4>
              <div className="grid grid-cols-3 gap-2 pl-4">
                {categoryIndicators.map((indicator) => (
                  <IndicatorCheckbox
                    key={indicator}
                    indicator={indicator}
                    isSelected={selectedIndicators.includes(indicator)}
                    onToggle={onIndicatorToggle}
                    disabled={disabled}
                    displayName={indicatorInfo[indicator]?.name || indicator}
                  />
                ))}
              </div>
            </div>
          ))
        )}
        
        <div className="mt-4 text-xs text-gray-500">
          選択済み: {selectedIndicators.length}個
          {maxIndicators && ` / 最大: ${maxIndicators}個`}
          {` / 利用可能: ${indicators.length}個`}
        </div>
      </div>
    );
  }

  // 通常の一覧表示
  return (
    <div className={className}>
      {isLoading ? (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600">指標リストを読み込み中...</span>
        </div>
      ) : indicatorsError ? (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <p className="text-red-800 text-sm">
            指標リストの読み込みに失敗しました。フォールバック指標を使用しています。
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-2">
          {indicators.map((indicator) => (
            <IndicatorCheckbox
              key={indicator}
              indicator={indicator}
              isSelected={selectedIndicators.includes(indicator)}
              onToggle={onIndicatorToggle}
              disabled={disabled}
              displayName={indicatorInfo[indicator]?.name || indicator}
            />
          ))}
        </div>
      )}
      
      <div className="mt-2 text-xs text-gray-500">
        選択済み: {selectedIndicators.length}個
        {maxIndicators && ` / 最大: ${maxIndicators}個`}
        {` / 利用可能: ${indicators.length}個`}
      </div>
    </div>
  );
};

// 個別の指標チェックボックスコンポーネント
interface IndicatorCheckboxProps {
  indicator: string;
  isSelected: boolean;
  onToggle: (indicator: string) => void;
  disabled: boolean;
  displayName: string;
}

const IndicatorCheckbox: React.FC<IndicatorCheckboxProps> = ({
  indicator,
  isSelected,
  onToggle,
  disabled,
  displayName,
}) => {
  return (
    <label 
      className={`flex items-center space-x-2 p-2 rounded-md hover:bg-gray-50 cursor-pointer ${
        disabled ? 'opacity-50 cursor-not-allowed' : ''
      }`}
      title={displayName}
    >
      <input
        type="checkbox"
        checked={isSelected}
        onChange={() => !disabled && onToggle(indicator)}
        disabled={disabled}
        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
      />
      <span className="text-sm font-mono">{indicator}</span>
    </label>
  );
};

// カテゴリ表示名の取得
const getCategoryDisplayName = (category: string): string => {
  const categoryNames: Record<string, string> = {
    trend: 'トレンド系',
    momentum: 'モメンタム系',
    volatility: 'ボラティリティ系',
    volume: '出来高系',
    price_transform: '価格変換系',
    other: 'その他',
  };
  
  return categoryNames[category] || category;
};

export default IndicatorSelector;

// 指標選択用のユーティリティフック
export const useIndicatorSelection = (initialIndicators: string[] = []) => {
  const [selectedIndicators, setSelectedIndicators] = React.useState<string[]>(initialIndicators);

  const toggleIndicator = React.useCallback((indicator: string) => {
    setSelectedIndicators(prev => 
      prev.includes(indicator)
        ? prev.filter(ind => ind !== indicator)
        : [...prev, indicator]
    );
  }, []);

  const selectAll = React.useCallback((indicators: string[]) => {
    setSelectedIndicators(indicators);
  }, []);

  const clearAll = React.useCallback(() => {
    setSelectedIndicators([]);
  }, []);

  const selectByCategory = React.useCallback((categoryIndicators: string[]) => {
    setSelectedIndicators(prev => {
      const newSelection = new Set(prev);
      categoryIndicators.forEach(indicator => newSelection.add(indicator));
      return Array.from(newSelection);
    });
  }, []);

  return {
    selectedIndicators,
    setSelectedIndicators,
    toggleIndicator,
    selectAll,
    clearAll,
    selectByCategory,
  };
};
