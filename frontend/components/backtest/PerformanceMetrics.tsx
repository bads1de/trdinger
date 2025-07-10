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
import ActionButton from "../common/ActionButton";
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
        <div className="flex space-x-2">
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
            <ActionButton
              onClick={onOptimizationClick}
              variant="secondary"
            >
              🔧 最適化
            </ActionButton>
          )}
          <ActionButton
            onClick={() => setIsChartModalOpen(true)}
            title="チャート分析を表示"
          >
            📊 分析
          </ActionButton>
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
