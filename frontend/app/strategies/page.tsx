/**
 * 生成戦略一覧ページ
 *
 * オートストラテジー（GA）で生成された戦略を一覧・詳細表示します。
 */

"use client";

import React, { useCallback, useEffect, useState } from "react";
import GeneratedStrategiesTable from "@/components/backtest/GeneratedStrategiesTable";
import StrategyGeneDisplay from "@/components/backtest/StrategyGeneDisplay";
import LoadingSpinner from "@/components/common/LoadingSpinner";
import { GeneratedStrategy } from "@/types/generatedStrategy";
import { StrategyGene } from "@/types/backtest";
import { BACKEND_API_URL } from "@/constants";
import { formatNumber, formatPercentage } from "@/utils/formatters";

interface StrategiesApiResponse {
  success: boolean;
  strategies: GeneratedStrategy[];
  total_count: number;
  has_more: boolean;
  message: string;
  timestamp: string;
}

export default function StrategiesPage() {
  const [strategies, setStrategies] = useState<GeneratedStrategy[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedStrategy, setSelectedStrategy] =
    useState<GeneratedStrategy | null>(null);

  const loadStrategies = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `${BACKEND_API_URL}/api/strategies?limit=50&sort_by=fitness_score&sort_order=desc`,
      );
      if (!response.ok) {
        throw new Error(`戦略一覧の取得に失敗しました (HTTP ${response.status})`);
      }
      const data: StrategiesApiResponse = await response.json();
      setStrategies(data.strategies);
      setTotalCount(data.total_count);
    } catch (e) {
      console.error("戦略一覧取得エラー:", e);
      setError(e instanceof Error ? e.message : "戦略一覧の取得に失敗しました");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStrategies();
  }, [loadStrategies]);

  const handleStrategySelect = (strategy: GeneratedStrategy) => {
    setSelectedStrategy(strategy);
  };

  return (
    <div className="min-h-screen from-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold mb-2">生成戦略</h1>
              <p className="text-secondary-400">
                オートストラテジー（GA）が生成した戦略の一覧と成績
              </p>
            </div>
            <button
              onClick={loadStrategies}
              className="px-4 py-2 rounded-lg bg-secondary-700 hover:bg-secondary-600 text-sm text-white transition-colors"
            >
              更新
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-6 rounded-lg bg-red-900/20 border border-red-500/30 p-4 text-sm text-red-300">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-secondary-950 rounded-lg p-6 border border-secondary-700 lg:col-span-1">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">
                戦略一覧
                {totalCount > 0 && (
                  <span className="text-sm text-secondary-400 ml-2">
                    ({totalCount}件)
                  </span>
                )}
              </h2>
            </div>
            <GeneratedStrategiesTable
              strategies={strategies}
              loading={loading}
              onStrategySelect={handleStrategySelect}
            />
          </div>

          <div className="space-y-6 lg:col-span-1">
            {selectedStrategy ? (
              <StrategyDetail strategy={selectedStrategy} />
            ) : (
              <div className="flex items-center justify-center h-full bg-secondary-950 rounded-lg p-6 border border-secondary-700 border-dashed">
                <p className="text-secondary-400">
                  戦略を一覧から選択して詳細を表示
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function StrategyDetail({ strategy }: { strategy: GeneratedStrategy }) {
  // parameters から StrategyGene 形式へ変換
  const strategyGene: StrategyGene = {
    id: strategy.id,
    indicators: strategy.parameters?.indicators,
    long_entry_conditions: strategy.parameters?.long_entry_conditions,
    short_entry_conditions: strategy.parameters?.short_entry_conditions,
    tpsl_gene: strategy.parameters?.tpsl_gene,
    position_sizing_gene: strategy.parameters?.position_sizing_gene,
    metadata: {
      evaluation_summary: strategy.evaluation_summary,
      validation: strategy.validation_summary,
    },
  };

  return (
    <div className="space-y-4">
      <div className="bg-secondary-950 rounded-lg p-6 border border-secondary-700">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h2 className="text-xl font-semibold">{strategy.name}</h2>
            <p className="text-sm text-secondary-400 mt-1">
              {strategy.description}
            </p>
          </div>
          <div className="text-right">
            <div className="text-sm text-secondary-400">
              世代 {strategy.generation}
            </div>
            <div className="text-sm text-secondary-400">
              実験ID: {strategy.experiment_id}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          <MetricItem label="Fitness" value={formatNumber(strategy.fitness_score ?? 0)} />
          <MetricItem
            label="総リターン"
            value={formatPercentage(strategy.expected_return)}
            valueClass={
              strategy.expected_return >= 0 ? "text-green-400" : "text-red-400"
            }
          />
          <MetricItem
            label="シャープレシオ"
            value={formatNumber(strategy.sharpe_ratio)}
          />
          <MetricItem
            label="最大DD"
            value={formatPercentage(strategy.max_drawdown)}
            valueClass="text-red-400"
          />
          <MetricItem
            label="勝率"
            value={formatPercentage(strategy.win_rate)}
          />
          <MetricItem
            label="リスク"
            value={strategy.risk_level}
          />
        </div>

        {strategy.validation_summary && (
          <div className="mt-4 rounded-lg border border-secondary-700 p-4">
            <h3 className="text-sm font-semibold text-secondary-300 mb-2">
              自動検証（WFA）
            </h3>
            <div className="text-sm">
              {strategy.validation_passed ? (
                <span className="text-green-400 font-medium">合格</span>
              ) : (
                <span className="text-red-400 font-medium">不合格</span>
              )}
              {typeof strategy.validation_summary.pass_rate === "number" && (
                <span className="text-secondary-400 ml-2">
                  pass_rate:{" "}
                  {formatPercentage(strategy.validation_summary.pass_rate)}
                </span>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="bg-secondary-950 rounded-lg p-6 border border-secondary-700">
        <h2 className="text-xl font-semibold mb-4">戦略構成</h2>
        <StrategyGeneDisplay strategyGene={strategyGene} />
      </div>
    </div>
  );
}

function MetricItem({
  label,
  value,
  valueClass = "text-white",
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div className="bg-secondary-800/50 rounded-lg p-3">
      <div className="text-xs text-secondary-400">{label}</div>
      <div className={`text-lg font-semibold mt-1 ${valueClass}`}>{value}</div>
    </div>
  );
}
