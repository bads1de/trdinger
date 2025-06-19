/**
 * パフォーマンス指標表示コンポーネント
 *
 * バックテスト結果のパフォーマンス指標を視覚的に表示します。
 */

"use client";

import React, { useState } from "react";
import TradeHistoryTable from "./TradeHistoryTable";
import ChartModal from "./charts/ChartModal";
import { BacktestResult } from "@/types/backtest";
import TabButton from "../common/TabButton";

interface PerformanceMetricsProps {
  result: BacktestResult;
  onOptimizationClick?: () => void;
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
    gray: "bg-secondary-900/50 border-secondary-600 text-gray-300",
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

export default function PerformanceMetrics({
  result,
  onOptimizationClick,
}: PerformanceMetricsProps) {
  const { performance_metrics: metrics } = result;
  const [activeTab, setActiveTab] = useState<
    "overview" | "parameters" | "trades"
  >("overview");
  const [isChartModalOpen, setIsChartModalOpen] = useState(false);

  const formatPercentage = (value: number | undefined | null) => {
    if (value === undefined || value === null || isNaN(value)) {
      // 勝率がnullまたはundefinedの場合、"0.00%"として表示する
      // もしくは、より明確な表示「データなし」などを検討することも可能
      return "0.00%";
    }
    // バックエンドから既にパーセンテージ値として渡されるため、100を掛けずにそのまま使用する
    return `${value.toFixed(2)}%`;
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
      {activeTab === "overview" && (
        <div className="space-y-6">
          {/* 基本情報 */}
          <div className="bg-secondary-900/30 rounded-lg p-4 border border-secondary-700">
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
          <div className="bg-gray-800/30 rounded-lg p-4 border border-secondary-700">
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

      {/* パラメータタブ */}
      {activeTab === "parameters" && (
        <div className="space-y-6">
          <div className="bg-secondary-900/30 rounded-lg p-4 border border-secondary-700">
            <h3 className="text-lg font-semibold mb-4 text-white">
              戦略パラメータ
            </h3>

            {/* デバッグ情報 */}
            <div className="mb-4 p-3 bg-gray-800/50 rounded text-xs text-gray-400">
              <p>
                Debug: config_json ={" "}
                {JSON.stringify(result.config_json, null, 2)}
              </p>
            </div>

            {result.config_json && result.config_json.strategy_config ? (
              <div className="space-y-4">
                {/* 戦略タイプ */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <span className="text-gray-400 text-sm">戦略タイプ:</span>
                    <span className="ml-2 text-white font-medium">
                      {result.config_json.strategy_config.strategy_type ||
                        "N/A"}
                    </span>
                  </div>
                </div>

                {/* パラメータ一覧 */}
                {result.config_json.strategy_config.parameters &&
                Object.keys(result.config_json.strategy_config.parameters)
                  .length > 0 ? (
                  <div>
                    <h4 className="text-md font-medium mb-3 text-white">
                      パラメータ設定
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {Object.entries(
                        result.config_json.strategy_config.parameters
                      ).map(([key, value]) => (
                        <div
                          key={key}
                          className="bg-gray-800/50 rounded-lg p-3 border border-gray-600/30"
                        >
                          <div className="flex flex-col space-y-1">
                            <span className="text-gray-400 text-sm capitalize">
                              {key.replace(/_/g, " ")}
                            </span>
                            <span className="text-white font-medium text-lg">
                              {typeof value === "number"
                                ? value.toFixed(4)
                                : String(value)}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-400">
                    パラメータ情報がありません
                  </div>
                )}

                {/* バックテスト設定 */}
                <div>
                  <h4 className="text-md font-medium mb-3 text-white">
                    バックテスト設定
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-600/30">
                      <div className="flex flex-col space-y-1">
                        <span className="text-gray-400 text-sm">初期資金</span>
                        <span className="text-green-400 font-medium text-lg">
                          {formatCurrency(result.initial_capital)}
                        </span>
                      </div>
                    </div>
                    <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-600/30">
                      <div className="flex flex-col space-y-1">
                        <span className="text-gray-400 text-sm">手数料率</span>
                        <span className="text-yellow-400 font-medium text-lg">
                          {formatPercentage(result.commission_rate)}
                        </span>
                      </div>
                    </div>
                    <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-600/30">
                      <div className="flex flex-col space-y-1">
                        <span className="text-gray-400 text-sm">取引ペア</span>
                        <span className="text-blue-400 font-medium text-lg">
                          {result.symbol}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {/* 基本情報のフォールバック表示 */}
                <div>
                  <h4 className="text-md font-medium mb-3 text-white">
                    基本設定
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-600/30">
                      <div className="flex flex-col space-y-1">
                        <span className="text-gray-400 text-sm">戦略名</span>
                        <span className="text-white font-medium text-lg">
                          {result.strategy_name}
                        </span>
                      </div>
                    </div>
                    <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-600/30">
                      <div className="flex flex-col space-y-1">
                        <span className="text-gray-400 text-sm">初期資金</span>
                        <span className="text-green-400 font-medium text-lg">
                          {formatCurrency(result.initial_capital)}
                        </span>
                      </div>
                    </div>
                    <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-600/30">
                      <div className="flex flex-col space-y-1">
                        <span className="text-gray-400 text-sm">手数料率</span>
                        <span className="text-yellow-400 font-medium text-lg">
                          {formatPercentage(result.commission_rate)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="text-center py-8 text-gray-400">
                  <div className="text-6xl mb-4">⚙️</div>
                  <p className="text-lg">詳細なパラメータ情報がありません</p>
                  <p className="text-sm mt-2">
                    この結果は古いバージョンで作成されたため、戦略パラメータの詳細情報が含まれていません
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 取引履歴タブ */}
      {activeTab === "trades" && (
        <div className="space-y-6">
          <div className="bg-secondary-900/30 rounded-lg p-4 border border-secondary-700">
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
