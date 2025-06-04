/**
 * パフォーマンス指標表示コンポーネント
 *
 * バックテスト結果のパフォーマンス指標を視覚的に表示します。
 */

"use client";

import React, { useState } from "react";
import TradeHistoryTable from "./TradeHistoryTable";
import ChartModal from "./charts/ChartModal";

interface BacktestResult {
  id?: string;
  strategy_name: string;
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  commission_rate: number;
  performance_metrics: {
    total_return: number | null;
    sharpe_ratio: number | null;
    max_drawdown: number | null;
    win_rate: number | null;
    profit_factor: number | null;
    total_trades: number | null;
    winning_trades: number | null;
    losing_trades: number | null;
    avg_win: number | null;
    avg_loss: number | null;
  };
  equity_curve?: Array<{
    timestamp: string;
    equity: number;
  }>;
  trade_history?: Array<{
    size: number;
    entry_price: number;
    exit_price: number;
    pnl: number;
    return_pct: number;
    entry_time: string;
    exit_time: string;
  }>;
  created_at?: string;
}

interface PerformanceMetricsProps {
  result: BacktestResult;
}

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  color?: "green" | "red" | "blue" | "gray" | "yellow";
  icon?: React.ReactNode;
}

function MetricCard({
  title,
  value,
  subtitle,
  color = "gray",
  icon,
}: MetricCardProps) {
  const colorClasses = {
    green: "bg-green-900/20 border-green-500/30 text-green-400",
    red: "bg-red-900/20 border-red-500/30 text-red-400",
    blue: "bg-blue-900/20 border-blue-500/30 text-blue-400",
    yellow: "bg-yellow-900/20 border-yellow-500/30 text-yellow-400",
    gray: "bg-gray-800/50 border-gray-700/30 text-gray-300",
  };

  return (
    <div className={`p-4 rounded-lg border ${colorClasses[color]}`}>
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-400 mb-1">{title}</p>
          <p className="text-2xl font-bold">{value}</p>
          {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
        </div>
        {icon && <div className="ml-3 opacity-60">{icon}</div>}
      </div>
    </div>
  );
}

// タブボタンコンポーネント
interface TabButtonProps {
  id: string;
  label: string;
  isActive: boolean;
  onClick: () => void;
}

function TabButton({ id, label, isActive, onClick }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
        isActive
          ? "bg-blue-600 text-white"
          : "bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white"
      }`}
    >
      {label}
    </button>
  );
}

export default function PerformanceMetrics({
  result,
}: PerformanceMetricsProps) {
  const { performance_metrics: metrics } = result;
  const [activeTab, setActiveTab] = useState<"overview" | "trades">("overview");
  const [isChartModalOpen, setIsChartModalOpen] = useState(false);

  const formatPercentage = (value: number | undefined | null) => {
    if (value === undefined || value === null || isNaN(value)) {
      // 勝率がnullまたはundefinedの場合、"0.00%"として表示する
      // もしくは、より明確な表示「データなし」などを検討することも可能
      return "0.00%";
    }
    return `${(value * 100).toFixed(2)}%`;
  };

  const formatNumber = (
    value: number | undefined | null,
    decimals: number = 2
  ) => {
    if (value === undefined || value === null || isNaN(value)) {
      return "N/A";
    }
    return value.toFixed(decimals);
  };

  const formatCurrency = (value: number | undefined | null) => {
    if (value === undefined || value === null || isNaN(value)) {
      return "N/A";
    }
    return `$${value.toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;
  };

  const getReturnColor = (
    value: number | undefined | null
  ): "green" | "red" | "gray" => {
    if (value === undefined || value === null || isNaN(value)) return "gray";
    if (value > 0) return "green";
    if (value < 0) return "red";
    return "gray";
  };

  const getSharpeColor = (
    value: number | undefined | null
  ): "green" | "yellow" | "red" | "gray" => {
    if (value === undefined || value === null || isNaN(value)) return "gray";
    if (value > 1.5) return "green";
    if (value > 1.0) return "yellow";
    if (value > 0) return "gray";
    return "red";
  };

  const finalEquity =
    result.initial_capital &&
    metrics.total_return !== undefined &&
    metrics.total_return !== null
      ? result.initial_capital * (1 + metrics.total_return)
      : result.initial_capital;

  return (
    <div className="space-y-6">
      {/* タブナビゲーション */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex space-x-1">
          <TabButton
            id="overview"
            label="概要"
            isActive={activeTab === "overview"}
            onClick={() => setActiveTab("overview")}
          />
          <TabButton
            id="trades"
            label="取引履歴"
            isActive={activeTab === "trades"}
            onClick={() => setActiveTab("trades")}
          />
        </div>

        {/* チャート表示ボタン */}
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

      {/* タブコンテンツ */}
      {activeTab === "overview" && (
        <div className="space-y-6">
          {/* 基本情報 */}
          <div className="bg-gray-800/30 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 text-white">基本情報</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-400">戦略:</span>
                <span className="ml-2 text-white font-medium">
                  {result.strategy_name}
                </span>
              </div>
              <div>
                <span className="text-gray-400">シンボル:</span>
                <span className="ml-2 text-white font-medium">
                  {result.symbol}
                </span>
              </div>
              <div>
                <span className="text-gray-400">時間軸:</span>
                <span className="ml-2 text-white font-medium">
                  {result.timeframe}
                </span>
              </div>
              <div>
                <span className="text-gray-400">初期資金:</span>
                <span className="ml-2 text-white font-medium">
                  {formatCurrency(result.initial_capital)}
                </span>
              </div>
            </div>
          </div>

          {/* 主要パフォーマンス指標 */}
          <div>
            <h3 className="text-lg font-semibold mb-3 text-white">
              パフォーマンス指標
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard
                title="総リターン"
                value={formatPercentage(metrics.total_return)}
                subtitle={`最終資産: ${formatCurrency(finalEquity)}`}
                color={getReturnColor(metrics.total_return)}
                icon={
                  <svg
                    className="w-6 h-6"
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
                }
              />

              <MetricCard
                title="シャープレシオ"
                value={formatNumber(metrics.sharpe_ratio)}
                subtitle="リスク調整後リターン"
                color={getSharpeColor(metrics.sharpe_ratio)}
                icon={
                  <svg
                    className="w-6 h-6"
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
                }
              />

              <MetricCard
                title="最大ドローダウン"
                value={formatPercentage(metrics.max_drawdown)}
                subtitle="最大下落率"
                color="red"
                icon={
                  <svg
                    className="w-6 h-6"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6"
                    />
                  </svg>
                }
              />

              <MetricCard
                title="勝率"
                value={formatPercentage(metrics.win_rate)}
                subtitle={`${metrics.winning_trades || 0}勝 / ${
                  metrics.losing_trades || 0
                }敗`}
                color={
                  metrics.win_rate && metrics.win_rate > 0.5
                    ? "green"
                    : "yellow"
                }
                icon={
                  <svg
                    className="w-6 h-6"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                }
              />
            </div>
          </div>

          {/* 詳細指標 */}
          <div>
            <h3 className="text-lg font-semibold mb-3 text-white">詳細指標</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard
                title="プロフィットファクター"
                value={formatNumber(metrics.profit_factor)}
                subtitle="総利益 / 総損失"
                color={
                  metrics.profit_factor && metrics.profit_factor > 1
                    ? "green"
                    : "red"
                }
              />

              <MetricCard
                title="総取引数"
                value={metrics.total_trades || 0}
                subtitle="実行された取引の総数"
                color="blue"
              />

              <MetricCard
                title="平均利益"
                value={formatCurrency(metrics.avg_win)}
                subtitle="勝ちトレードあたり"
                color="green"
              />

              <MetricCard
                title="平均損失"
                value={formatCurrency(
                  metrics.avg_loss !== undefined && metrics.avg_loss !== null
                    ? Math.abs(metrics.avg_loss)
                    : 0
                )}
                subtitle="負けトレードあたり"
                color="red"
              />
            </div>
          </div>

          {/* 期間情報 */}
          <div className="bg-gray-800/30 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 text-white">
              バックテスト期間
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-gray-400">開始日:</span>
                <span className="ml-2 text-white font-medium">
                  {new Date(result.start_date).toLocaleDateString("ja-JP")}
                </span>
              </div>
              <div>
                <span className="text-gray-400">終了日:</span>
                <span className="ml-2 text-white font-medium">
                  {new Date(result.end_date).toLocaleDateString("ja-JP")}
                </span>
              </div>
              <div>
                <span className="text-gray-400">手数料率:</span>
                <span className="ml-2 text-white font-medium">
                  {formatPercentage(result.commission_rate)}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 取引履歴タブ */}
      {activeTab === "trades" && (
        <div className="space-y-6">
          <div className="bg-gray-800/30 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-4 text-white">
              取引履歴詳細
            </h3>
            {result.trade_history && result.trade_history.length > 0 ? (
              <TradeHistoryTable tradeHistory={result.trade_history} />
            ) : (
              <div className="text-center py-8 text-gray-400">
                取引履歴がありません
              </div>
            )}
          </div>
        </div>
      )}

      {/* チャートモーダル */}
      <ChartModal
        isOpen={isChartModalOpen}
        onClose={() => setIsChartModalOpen(false)}
        result={result}
      />
    </div>
  );
}
