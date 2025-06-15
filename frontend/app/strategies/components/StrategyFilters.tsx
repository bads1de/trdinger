/**
 * 戦略フィルターコンポーネント
 *
 * 戦略一覧のフィルタリング・ソート・検索機能を提供
 *
 */

import React, { useState } from "react";
import {
  UnifiedStrategyFilter,
  UnifiedPagination,
} from "@/types/auto-strategy";
import { StrategyCategory, RiskLevel } from "@/types/strategy-showcase";

interface StrategyFiltersProps {
  filter: UnifiedStrategyFilter;
  pagination: UnifiedPagination;
  onFilterChange: (filter: UnifiedStrategyFilter) => void;
  onPaginationChange: (pagination: UnifiedPagination) => void;
  categories: Record<StrategyCategory, string>;
  riskLevels: Record<RiskLevel, string>;
}

/**
 * 戦略フィルターコンポーネント
 */
const StrategyFilters: React.FC<StrategyFiltersProps> = ({
  filter,
  pagination,
  onFilterChange,
  onPaginationChange,
  categories,
  riskLevels,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  /**
   * フィルター変更ハンドラ
   */
  const handleFilterChange = (key: string, value: any) => {
    const newFilter = { ...filter, [key]: value };
    onFilterChange(newFilter);
  };

  /**
   * ソート変更ハンドラ
   */
  const handleSortChange = (
    sortBy: UnifiedPagination["sort_by"],
    sortOrder: UnifiedPagination["sort_order"]
  ) => {
    const newPagination = {
      ...pagination,
      sort_by: sortBy,
      sort_order: sortOrder,
      offset: 0, // ソート変更時はページをリセット
    };
    onPaginationChange(newPagination);
  };

  /**
   * フィルターリセット
   */
  const resetFilters = () => {
    onFilterChange({});
    onPaginationChange({
      ...pagination,
      offset: 0,
      sort_by: "expected_return",
      sort_order: "desc",
    });
  };

  /**
   * ソートオプション
   */
  const sortOptions = [
    { value: "expected_return", label: "期待リターン" },
    { value: "sharpe_ratio", label: "シャープレシオ" },
    { value: "max_drawdown", label: "最大ドローダウン" },
    { value: "win_rate", label: "勝率" },
    { value: "created_at", label: "作成日時" },
    { value: "fitness_score", label: "フィットネススコア" }, // 新しいソートオプションを追加
  ];

  return (
    <div className="enterprise-card mb-6">
      <div className="p-6">
        {/* フィルターヘッダー */}
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100">
            🔍 フィルター・ソート
          </h2>
          <div className="flex items-center gap-3">
            <button
              onClick={resetFilters}
              className="text-sm text-secondary-600 dark:text-secondary-400 hover:text-primary-500 transition-colors"
            >
              リセット
            </button>
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-500 transition-colors"
            >
              {isExpanded ? "折りたたむ" : "詳細フィルター"}
            </button>
          </div>
        </div>

        {/* 基本フィルター */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          {/* カテゴリフィルター */}
          <div>
            <label className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2">
              戦略カテゴリ
            </label>
            <select
              value={filter.category || ""}
              onChange={(e) =>
                handleFilterChange("category", e.target.value || undefined)
              }
              className="w-full px-3 py-2 border border-secondary-300 dark:border-secondary-600 rounded-md bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="">すべてのカテゴリ</option>
              {Object.entries(categories).map(([key, label]) => (
                <option key={key} value={key}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          {/* リスクレベルフィルター */}
          <div>
            <label className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2">
              リスクレベル
            </label>
            <select
              value={filter.risk_level || ""}
              onChange={(e) =>
                handleFilterChange("risk_level", e.target.value || undefined)
              }
              className="w-full px-3 py-2 border border-secondary-300 dark:border-secondary-600 rounded-md bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="">すべてのリスクレベル</option>
              {Object.entries(riskLevels).map(([key, label]) => (
                <option key={key} value={key}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          {/* ソート項目 */}
          <div>
            <label className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2">
              ソート項目
            </label>
            <select
              value={pagination.sort_by}
              onChange={(e) =>
                handleSortChange(
                  e.target.value as UnifiedPagination["sort_by"],
                  pagination.sort_order
                )
              }
              className="w-full px-3 py-2 border border-secondary-300 dark:border-secondary-600 rounded-md bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              {sortOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* ソート順序 */}
          <div>
            <label className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2">
              ソート順序
            </label>
            <select
              value={pagination.sort_order}
              onChange={(e) =>
                handleSortChange(
                  pagination.sort_by,
                  e.target.value as UnifiedPagination["sort_order"]
                )
              }
              className="w-full px-3 py-2 border border-secondary-300 dark:border-secondary-600 rounded-md bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="desc">降順（高い順）</option>
              <option value="asc">昇順（低い順）</option>
            </select>
          </div>
        </div>

        {/* 詳細フィルター（展開時のみ表示） */}
        {isExpanded && (
          <div className="border-t border-secondary-200 dark:border-secondary-700 pt-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* 期待リターン範囲 */}
              <div>
                <label className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2">
                  期待リターン範囲 (%)
                </label>
                <div className="flex gap-2">
                  <input
                    type="number"
                    placeholder="最小"
                    value={filter.min_return || ""}
                    onChange={(e) =>
                      handleFilterChange(
                        "min_return",
                        e.target.value ? parseFloat(e.target.value) : undefined
                      )
                    }
                    className="w-full px-3 py-2 border border-secondary-300 dark:border-secondary-600 rounded-md bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  />
                  <input
                    type="number"
                    placeholder="最大"
                    value={filter.max_return || ""}
                    onChange={(e) =>
                      handleFilterChange(
                        "max_return",
                        e.target.value ? parseFloat(e.target.value) : undefined
                      )
                    }
                    className="w-full px-3 py-2 border border-secondary-300 dark:border-secondary-600 rounded-md bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  />
                </div>
              </div>

              {/* シャープレシオ範囲 */}
              <div>
                <label className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2">
                  シャープレシオ範囲
                </label>
                <div className="flex gap-2">
                  <input
                    type="number"
                    step="0.1"
                    placeholder="最小"
                    value={filter.min_sharpe || ""}
                    onChange={(e) =>
                      handleFilterChange(
                        "min_sharpe",
                        e.target.value ? parseFloat(e.target.value) : undefined
                      )
                    }
                    className="w-full px-3 py-2 border border-secondary-300 dark:border-secondary-600 rounded-md bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  />
                  <input
                    type="number"
                    step="0.1"
                    placeholder="最大"
                    value={filter.max_sharpe || ""}
                    onChange={(e) =>
                      handleFilterChange(
                        "max_sharpe",
                        e.target.value ? parseFloat(e.target.value) : undefined
                      )
                    }
                    className="w-full px-3 py-2 border border-secondary-300 dark:border-secondary-600 rounded-md bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  />
                </div>
              </div>

              {/* 検索 */}
              <div>
                <label className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2">
                  戦略名検索
                </label>
                <input
                  type="text"
                  placeholder="戦略名で検索..."
                  value={filter.search_query || ""}
                  onChange={(e) =>
                    handleFilterChange(
                      "search_query",
                      e.target.value || undefined
                    )
                  }
                  className="w-full px-3 py-2 border border-secondary-300 dark:border-secondary-600 rounded-md bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
              </div>
            </div>
          </div>
        )}

        {/* アクティブフィルター表示 */}
        {(filter.category || filter.risk_level || filter.search_query) && (
          <div className="mt-4 pt-4 border-t border-secondary-200 dark:border-secondary-700">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm text-secondary-600 dark:text-secondary-400">
                アクティブフィルター:
              </span>

              {filter.category && (
                <span className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200">
                  カテゴリ: {categories[filter.category as StrategyCategory]}
                  <button
                    onClick={() => handleFilterChange("category", undefined)}
                    className="ml-1 text-primary-600 hover:text-primary-800"
                  >
                    ×
                  </button>
                </span>
              )}

              {filter.risk_level && (
                <span className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200">
                  リスク: {riskLevels[filter.risk_level as RiskLevel]}
                  <button
                    onClick={() => handleFilterChange("risk_level", undefined)}
                    className="ml-1 text-primary-600 hover:text-primary-800"
                  >
                    ×
                  </button>
                </span>
              )}

              {filter.search_query && (
                <span className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200">
                  検索: {filter.search_query}
                  <button
                    onClick={() =>
                      handleFilterChange("search_query", undefined)
                    }
                    className="ml-1 text-primary-600 hover:text-primary-800"
                  >
                    ×
                  </button>
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default StrategyFilters;
