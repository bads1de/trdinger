/**
 * GA設定フォームコンポーネント
 *
 * 遺伝的アルゴリズムによる自動戦略生成の設定を行います。
 */

"use client";

import React, { useState } from "react";
import { GAConfig } from "@/types/optimization";

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

  // GA設定（シンプル版）
  const [gaConfig, setGaConfig] = useState<GAConfig["ga_config"]>({
    population_size: 50,
    generations: 30,
    crossover_rate: 0.8,
    mutation_rate: 0.1,
    elite_size: 5,
    max_indicators: 5,
    allowed_indicators: [], // バックエンドで自動設定
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

  // プリセット設定（シンプル版）
  const [selectedPreset, setSelectedPreset] = useState("default");

  const presets = {
    default: { population_size: 50, generations: 30 },
    fast: { population_size: 30, generations: 20 },
    thorough: { population_size: 100, generations: 50 },
  };

  // プリセット変更ハンドラー
  const handlePresetChange = (preset: string) => {
    setSelectedPreset(preset);
    if (presets[preset as keyof typeof presets]) {
      setGaConfig((prev) => ({
        ...prev,
        ...presets[preset as keyof typeof presets],
      }));
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const config: GAConfig = {
      experiment_name: experimentName,
      base_config: baseConfig,
      ga_config: gaConfig,
    };

    onSubmit(config);
  };

  // フォーム送信ハンドラー

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
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600"
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
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600"
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
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600"
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
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600"
            />
          </div>
        </div>

        {/* 指標は自動選択される旨の説明 */}
        <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg
                className="h-5 w-5 text-blue-400"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-blue-800">
                自動指標選択
              </h3>
              <div className="mt-2 text-sm text-blue-700">
                <p>
                  使用する指標は遺伝的アルゴリズムによって自動的に選択・最適化されます。
                  手動で指標を選択する必要はありません。
                </p>
              </div>
            </div>
          </div>
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
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600"
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
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600"
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
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600"
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
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600"
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
