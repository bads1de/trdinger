/**
 * GA設定フォームコンポーネント
 *
 * 遺伝的アルゴリズムによる自動戦略生成の設定を行います。
 */

"use client";

import React, { useState, useEffect } from "react";
import { useApiCall } from "@/hooks/useApiCall";
import IndicatorSelector from "@/components/common/IndicatorSelector";

interface GAConfig {
  experiment_name: string;
  base_config: {
    symbol: string;
    timeframe: string;
    start_date: string;
    end_date: string;
    initial_capital: number;
    commission_rate: number;
  };
  ga_config: {
    population_size: number;
    generations: number;
    crossover_rate: number;
    mutation_rate: number;
    elite_size: number;
    max_indicators: number;
    allowed_indicators: string[];
    fitness_weights: {
      total_return: number;
      sharpe_ratio: number;
      max_drawdown: number;
      win_rate: number;
    };
    fitness_constraints: {
      min_trades: number;
      max_drawdown_limit: number;
      min_sharpe_ratio: number;
    };
  };
}

interface GAConfigFormProps {
  onSubmit: (config: GAConfig) => void;
  isLoading?: boolean;
  initialConfig?: any;
  currentBacktestConfig?: any;
}

const GAConfigForm: React.FC<GAConfigFormProps> = ({
  onSubmit,
  isLoading = false,
  initialConfig = null,
  currentBacktestConfig = null,
}) => {
  // 基本設定（既存のバックテスト設定から引き継ぎ）
  const [baseConfig, setBaseConfig] = useState({
    symbol: currentBacktestConfig?.symbol || "BTC/USDT",
    timeframe: currentBacktestConfig?.timeframe || "1h",
    start_date: currentBacktestConfig?.start_date || "2024-01-01",
    end_date: currentBacktestConfig?.end_date || "2024-12-19",
    initial_capital: currentBacktestConfig?.initial_capital || 100000,
    commission_rate: currentBacktestConfig?.commission_rate || 0.00055,
  });

  // GA設定
  const [gaConfig, setGaConfig] = useState<GAConfig["ga_config"]>({
    population_size: 50,
    generations: 30,
    crossover_rate: 0.8,
    mutation_rate: 0.1,
    elite_size: 5,
    max_indicators: 5,
    allowed_indicators: [], // 型を明示的に指定
    fitness_weights: {
      total_return: 0.3,
      sharpe_ratio: 0.4,
      max_drawdown: 0.2,
      win_rate: 0.1,
    },
    fitness_constraints: {
      min_trades: 10,
      max_drawdown_limit: 0.3,
      min_sharpe_ratio: 0.5,
    },
  });

  // 実験名
  const [experimentName, setExperimentName] = useState(
    `${baseConfig.symbol.replace("/", "_")}_GA_${new Date()
      .toISOString()
      .slice(0, 10)}`
  );

  // プリセット設定
  const [selectedPreset, setSelectedPreset] = useState("default");

  const { execute: fetchPresets } = useApiCall();

  // プリセット設定の読み込み
  useEffect(() => {
    const loadPresets = async () => {
      try {
        const response = await fetchPresets(
          "/api/auto-strategy/config/presets"
        );
        if (response?.success && response.presets) {
          // デフォルトプリセットを適用
          if (response.presets.default) {
            setGaConfig((prev) => ({
              ...prev,
              ...response.presets.default,
            }));
          }
        }
      } catch (error) {
        console.error("Failed to load GA presets:", error);
      }
    };

    loadPresets();
  }, []);

  // プリセット変更ハンドラー
  const handlePresetChange = async (preset: string) => {
    setSelectedPreset(preset);

    try {
      const response = await fetchPresets("/api/auto-strategy/config/presets");
      if (response?.success && response.presets && response.presets[preset]) {
        setGaConfig((prev) => ({
          ...prev,
          ...response.presets[preset],
        }));
      }
    } catch (error) {
      console.error("Failed to apply preset:", error);
    }
  };

  // フォーム送信ハンドラー
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const config: GAConfig = {
      experiment_name: experimentName,
      base_config: baseConfig,
      ga_config: gaConfig,
    };

    onSubmit(config);
  };

  // 指標選択ハンドラー
  const handleIndicatorToggle = (indicator: string) => {
    setGaConfig((prev) => ({
      ...prev,
      allowed_indicators: prev.allowed_indicators.includes(indicator)
        ? prev.allowed_indicators.filter((ind) => ind !== indicator)
        : [...prev.allowed_indicators, indicator],
    }));
  };

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* 実験名 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            実験名
          </label>
          <input
            type="text"
            value={experimentName}
            onChange={(e) => setExperimentName(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
        </div>

        {/* プリセット選択 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            設定プリセット
          </label>
          <select
            value={selectedPreset}
            onChange={(e) => handlePresetChange(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="fast">高速（50個体×30世代）</option>
            <option value="default">標準（100個体×50世代）</option>
            <option value="thorough">徹底（200個体×100世代）</option>
          </select>
        </div>

        {/* GA基本設定 */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              個体数
            </label>
            <input
              type="number"
              value={gaConfig.population_size}
              onChange={(e) =>
                setGaConfig((prev) => ({
                  ...prev,
                  population_size: parseInt(e.target.value),
                }))
              }
              min="10"
              max="500"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              世代数
            </label>
            <input
              type="number"
              value={gaConfig.generations}
              onChange={(e) =>
                setGaConfig((prev) => ({
                  ...prev,
                  generations: parseInt(e.target.value),
                }))
              }
              min="5"
              max="200"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* 使用指標選択 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            使用可能指標（最大{gaConfig.max_indicators}個）
          </label>

          <IndicatorSelector
            selectedIndicators={gaConfig.allowed_indicators}
            onIndicatorToggle={handleIndicatorToggle}
            maxIndicators={gaConfig.max_indicators}
            showCategories={true}
            disabled={isLoading}
          />
        </div>

        {/* フィットネス重み */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            フィットネス重み（合計1.0）
          </label>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-gray-600">総リターン</label>
              <input
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={gaConfig.fitness_weights.total_return}
                onChange={(e) =>
                  setGaConfig((prev) => ({
                    ...prev,
                    fitness_weights: {
                      ...prev.fitness_weights,
                      total_return: parseFloat(e.target.value),
                    },
                  }))
                }
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-600">
                シャープレシオ
              </label>
              <input
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={gaConfig.fitness_weights.sharpe_ratio}
                onChange={(e) =>
                  setGaConfig((prev) => ({
                    ...prev,
                    fitness_weights: {
                      ...prev.fitness_weights,
                      sharpe_ratio: parseFloat(e.target.value),
                    },
                  }))
                }
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-600">
                最大ドローダウン
              </label>
              <input
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={gaConfig.fitness_weights.max_drawdown}
                onChange={(e) =>
                  setGaConfig((prev) => ({
                    ...prev,
                    fitness_weights: {
                      ...prev.fitness_weights,
                      max_drawdown: parseFloat(e.target.value),
                    },
                  }))
                }
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-600">勝率</label>
              <input
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={gaConfig.fitness_weights.win_rate}
                onChange={(e) =>
                  setGaConfig((prev) => ({
                    ...prev,
                    fitness_weights: {
                      ...prev.fitness_weights,
                      win_rate: parseFloat(e.target.value),
                    },
                  }))
                }
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
              />
            </div>
          </div>
        </div>

        {/* 送信ボタン */}
        <div className="flex justify-end space-x-3">
          <button
            type="submit"
            disabled={isLoading}
            className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? "実行中..." : "GA戦略生成開始"}
          </button>
        </div>
      </form>
    </div>
  );
};

export default GAConfigForm;
