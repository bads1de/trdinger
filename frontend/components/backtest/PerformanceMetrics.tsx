/**
 * パフォーマンス指標表示コンポーネント
 *
 * バックテスト結果のパフォーマンス指標を視覚的に表示します。
 */

"use client";

import React, { useState } from "react";
import ChartModal from "./charts/ChartModal";
import { BacktestResult } from "@/types/backtest";
import TabButton from "../common/TabButton";
import OverviewTab from "./tabs/OverviewTab";
import ParametersTab from "./tabs/ParametersTab";
import TradesTab from "./tabs/TradesTab";

interface PerformanceMetricsProps {
  result: BacktestResult;
  onOptimizationClick?: () => void;
}

export default function PerformanceMetrics({
  result,
  onOptimizationClick,
}: PerformanceMetricsProps) {
  const [activeTab, setActiveTab] = useState<
    "overview" | "parameters" | "trades"
  >("overview");
  const [isChartModalOpen, setIsChartModalOpen] = useState(false);

  return (
    <div className="space-y-6">
      {/* タブナビゲーション */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex space-x-1">
          <TabButton
            label="概要"
            isActive={activeTab === "overview"}
            onClick={() => setActiveTab("overview")}
          />
          <TabButton
            label="パラメータ"
            isActive={activeTab === "parameters"}
            onClick={() => setActiveTab("parameters")}
          />
          <TabButton
            label="取引履歴"
            isActive={activeTab === "trades"}
            onClick={() => setActiveTab("trades")}
          />
        </div>

        {/* ボタングループ */}
        <div className="flex items-center space-x-3">
          {onOptimizationClick && (
            <button
              onClick={onOptimizationClick}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors font-medium text-sm"
            >
              🔧 最適化
            </button>
          )}
          <button
            onClick={() => setIsChartModalOpen(true)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors flex items-center space-x-2"
            title="チャート分析を表示"
          >
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
            <span>分析</span>
          </button>
        </div>
      </div>

      {/* タブコンテンツ */}
      {activeTab === "overview" && <OverviewTab result={result} />}
      {activeTab === "parameters" && <ParametersTab result={result} />}
      {activeTab === "trades" && <TradesTab result={result} />}

      {/* チャートモーダル */}
      <ChartModal
        isOpen={isChartModalOpen}
        onClose={() => setIsChartModalOpen(false)}
        result={result}
      />
    </div>
  );
}
