/**
 * 指標選択コンポーネント
 *
 * カテゴリ別タブ表示と指標選択機能を提供します。
 */

"use client";

import React, { useState, useEffect } from "react";
import TabButton from "@/components/common/TabButton";
import { InputField } from "@/components/common/InputField";

// 指標カテゴリの定義
type IndicatorCategory =
  | "trend"
  | "momentum"
  | "volatility"
  | "volume"
  | "price_transform";

interface IndicatorInfo {
  type: string;
  name: string;
  description: string;
  parameters: Array<{
    name: string;
    type: string;
    default: any;
    min?: number;
    max?: number;
    description: string;
  }>;
  data_sources: string[];
}

interface IndicatorCategories {
  [key: string]: IndicatorInfo[];
}

interface SelectedIndicator {
  id: string;
  type: string;
  name: string;
  parameters: Record<string, any>;
  enabled: boolean;
}

interface IndicatorSelectorProps {
  selectedIndicators: SelectedIndicator[];
  onIndicatorsChange: (indicators: SelectedIndicator[]) => void;
  maxIndicators?: number;
}

const CATEGORY_INFO = {
  trend: {
    label: "トレンド系",
    description: "価格の方向性を分析する指標",
    icon: (
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
          d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
        />
      </svg>
    ),
  },
  momentum: {
    label: "モメンタム系",
    description: "価格変動の勢いを分析する指標",
    icon: (
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
          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
        />
      </svg>
    ),
  },
  volatility: {
    label: "ボラティリティ系",
    description: "価格変動の大きさを分析する指標",
    icon: (
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
          d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"
        />
      </svg>
    ),
  },
  volume: {
    label: "ボリューム系",
    description: "出来高を分析する指標",
    icon: (
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
          d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"
        />
      </svg>
    ),
  },
  price_transform: {
    label: "価格変換系",
    description: "価格データを変換・加工する指標",
    icon: (
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
          d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"
        />
      </svg>
    ),
  },
};

/**
 * 指標選択コンポーネント
 */
const IndicatorSelector: React.FC<IndicatorSelectorProps> = ({
  selectedIndicators,
  onIndicatorsChange,
  maxIndicators = 5,
}) => {
  // 状態管理
  const [activeCategory, setActiveCategory] =
    useState<IndicatorCategory>("trend");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [availableIndicators, setAvailableIndicators] =
    useState<IndicatorCategories>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // 利用可能な指標を取得
  useEffect(() => {
    const fetchIndicators = async () => {
      try {
        setLoading(true);
        const response = await fetch("/api/strategy-builder/indicators");

        if (!response.ok) {
          throw new Error("指標一覧の取得に失敗しました");
        }

        const data = await response.json();

        if (data.success) {
          setAvailableIndicators(data.data.categories);
        } else {
          throw new Error(data.message || "指標一覧の取得に失敗しました");
        }
      } catch (err) {
        console.error("指標取得エラー:", err);
        setError(
          err instanceof Error ? err.message : "不明なエラーが発生しました"
        );

        // フォールバック用のダミーデータ
        setAvailableIndicators({
          trend: [
            {
              type: "SMA",
              name: "Simple Moving Average",
              description: "単純移動平均",
              parameters: [
                {
                  name: "period",
                  type: "integer",
                  default: 20,
                  min: 2,
                  max: 200,
                  description: "移動平均期間",
                },
              ],
              data_sources: ["close"],
            },
            {
              type: "EMA",
              name: "Exponential Moving Average",
              description: "指数移動平均",
              parameters: [
                {
                  name: "period",
                  type: "integer",
                  default: 20,
                  min: 2,
                  max: 200,
                  description: "移動平均期間",
                },
              ],
              data_sources: ["close"],
            },
          ],
          momentum: [
            {
              type: "RSI",
              name: "Relative Strength Index",
              description: "相対力指数",
              parameters: [
                {
                  name: "period",
                  type: "integer",
                  default: 14,
                  min: 2,
                  max: 100,
                  description: "RSI計算期間",
                },
              ],
              data_sources: ["close"],
            },
          ],
          volatility: [
            {
              type: "ATR",
              name: "Average True Range",
              description: "平均真の値幅",
              parameters: [
                {
                  name: "period",
                  type: "integer",
                  default: 14,
                  min: 2,
                  max: 100,
                  description: "ATR計算期間",
                },
              ],
              data_sources: ["high", "low", "close"],
            },
          ],
          volume: [
            {
              type: "OBV",
              name: "On Balance Volume",
              description: "オンバランスボリューム",
              parameters: [],
              data_sources: ["close", "volume"],
            },
          ],
          price_transform: [
            {
              type: "AVGPRICE",
              name: "Average Price",
              description: "平均価格",
              parameters: [],
              data_sources: ["open", "high", "low", "close"],
            },
          ],
        });
      } finally {
        setLoading(false);
      }
    };

    fetchIndicators();
  }, []);

  // 現在のカテゴリの指標を取得
  const getCurrentCategoryIndicators = (): IndicatorInfo[] => {
    const indicators = availableIndicators[activeCategory] || [];

    if (!searchQuery) {
      return indicators;
    }

    return indicators.filter(
      (indicator) =>
        indicator.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        indicator.type.toLowerCase().includes(searchQuery.toLowerCase()) ||
        indicator.description.toLowerCase().includes(searchQuery.toLowerCase())
    );
  };

  // 指標が選択済みかチェック
  const isIndicatorSelected = (indicatorType: string): boolean => {
    return selectedIndicators.some(
      (selected) => selected.type === indicatorType
    );
  };

  // 指標の選択/選択解除
  const toggleIndicator = (indicator: IndicatorInfo) => {
    const isSelected = isIndicatorSelected(indicator.type);

    if (isSelected) {
      // 選択解除
      const newSelected = selectedIndicators.filter(
        (selected) => selected.type !== indicator.type
      );
      onIndicatorsChange(newSelected);
    } else {
      // 選択
      if (selectedIndicators.length >= maxIndicators) {
        alert(`最大${maxIndicators}個まで選択できます`);
        return;
      }

      const defaultParameters: Record<string, any> = {};
      indicator.parameters.forEach((param) => {
        defaultParameters[param.name] = param.default;
      });

      const newIndicator: SelectedIndicator = {
        id: `${indicator.type}_${Date.now()}`,
        type: indicator.type,
        name: indicator.name,
        parameters: defaultParameters,
        enabled: true,
      };

      onIndicatorsChange([...selectedIndicators, newIndicator]);
    }
  };

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-3 text-gray-300">指標を読み込み中...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-secondary-950 rounded-lg p-6">
      {/* エラー表示 */}
      {error && (
        <div className="mb-4 p-4 bg-red-900/50 border border-red-700 rounded-lg">
          <p className="text-red-300 text-sm">⚠️ {error}</p>
          <p className="text-red-400 text-xs mt-1">
            ダミーデータを表示しています
          </p>
        </div>
      )}

      {/* 選択済み指標の表示 */}
      {selectedIndicators.length > 0 && (
        <div className="mb-6">
          <h4 className="text-sm font-medium text-gray-300 mb-3">
            選択済み指標 ({selectedIndicators.length}/{maxIndicators})
          </h4>
          <div className="flex flex-wrap gap-2">
            {selectedIndicators.map((indicator) => (
              <div
                key={indicator.id}
                className="flex items-center gap-2 px-3 py-1 bg-blue-600 text-white rounded-full text-sm"
              >
                <span>{indicator.name}</span>
                <button
                  onClick={() =>
                    toggleIndicator({ type: indicator.type } as IndicatorInfo)
                  }
                  className="hover:bg-blue-700 rounded-full p-1"
                >
                  <svg
                    className="w-3 h-3"
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
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 検索フィールド */}
      <div className="mb-4">
        <InputField
          label="指標を検索"
          value={searchQuery}
          onChange={setSearchQuery}
          placeholder="指標名、タイプ、説明で検索..."
          className="bg-gray-700"
        />
      </div>

      {/* カテゴリタブ */}
      <div className="mb-6">
        <div className="flex flex-wrap gap-2">
          {Object.entries(CATEGORY_INFO).map(([category, info]) => (
            <TabButton
              key={category}
              label={info.label}
              isActive={activeCategory === category}
              onClick={() => setActiveCategory(category as IndicatorCategory)}
              icon={info.icon}
              variant="secondary"
              size="sm"
              badge={availableIndicators[category]?.length || 0}
            />
          ))}
        </div>
        <p className="text-sm text-gray-400 mt-2">
          {CATEGORY_INFO[activeCategory]?.description}
        </p>
      </div>

      {/* 指標一覧 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {getCurrentCategoryIndicators().map((indicator) => {
          const isSelected = isIndicatorSelected(indicator.type);

          return (
            <div
              key={indicator.type}
              className={`
                border rounded-lg p-4 cursor-pointer transition-all
                ${
                  isSelected
                    ? "border-blue-500 bg-blue-900/30"
                    : "border-gray-800 bg-gray-900 hover:border-gray-700"
                }
              `}
              onClick={() => toggleIndicator(indicator)}
            >
              <div className="flex items-start justify-between mb-2">
                <div>
                  <h5 className="font-medium text-white">{indicator.name}</h5>
                  <p className="text-xs text-gray-400">{indicator.type}</p>
                </div>
                <div
                  className={`
                  w-5 h-5 rounded border-2 flex items-center justify-center
                  ${
                    isSelected
                      ? "border-blue-500 bg-blue-500"
                      : "border-gray-400"
                  }
                `}
                >
                  {isSelected && (
                    <svg
                      className="w-3 h-3 text-white"
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
                  )}
                </div>
              </div>

              <p className="text-sm text-gray-300 mb-3">
                {indicator.description}
              </p>

              <div className="text-xs text-gray-400">
                <p>パラメータ: {indicator.parameters.length}個</p>
                <p>データソース: {indicator.data_sources.join(", ")}</p>
              </div>
            </div>
          );
        })}
      </div>

      {getCurrentCategoryIndicators().length === 0 && (
        <div className="text-center py-8">
          <p className="text-gray-400">
            {searchQuery
              ? "検索条件に一致する指標が見つかりません"
              : "このカテゴリには指標がありません"}
          </p>
        </div>
      )}
    </div>
  );
};

export default IndicatorSelector;
