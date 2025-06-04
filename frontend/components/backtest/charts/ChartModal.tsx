/**
 * チャートモーダルコンポーネント
 *
 * バックテスト結果のチャートを表示するモーダル
 */

"use client";

import React, { useState, useEffect } from "react";
import { BacktestResult } from "@/types/backtest";
import {
  transformEquityCurve,
  transformTradeHistory,
} from "@/utils/chartDataTransformers";
import EquityCurveChart from "./EquityCurveChart";
import DrawdownChart from "./DrawdownChart";
import TradeScatterChart from "./TradeScatterChart";
import ReturnsDistributionChart from "./ReturnsDistributionChart";
import MonthlyReturnsHeatmap from "./MonthlyReturnsHeatmap";

interface ChartModalProps {
  /** モーダルの表示状態 */
  isOpen: boolean;
  /** モーダルを閉じるコールバック */
  onClose: () => void;
  /** バックテスト結果データ */
  result: BacktestResult;
}

type TabType = "equity" | "drawdown" | "trades" | "distribution" | "heatmap";

interface TabConfig {
  id: TabType;
  label: string;
  description: string;
}

const tabs: TabConfig[] = [
  { id: "equity", label: "資産曲線", description: "時系列での資産推移" },
  { id: "drawdown", label: "ドローダウン", description: "最大下落期間の分析" },
  { id: "trades", label: "取引分析", description: "利益/損失の分布" },
  {
    id: "distribution",
    label: "リターン分布",
    description: "取引リターンの統計分布",
  },
  {
    id: "heatmap",
    label: "月次ヒートマップ",
    description: "月別パフォーマンスの季節性",
  },
];

/**
 * タブボタンコンポーネント
 */
const TabButton: React.FC<{
  tab: TabConfig;
  isActive: boolean;
  onClick: () => void;
}> = ({ tab, isActive, onClick }) => (
  <button
    onClick={onClick}
    className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
      isActive
        ? "bg-blue-600 text-white"
        : "bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white"
    }`}
    title={tab.description}
  >
    {tab.label}
  </button>
);

/**
 * チャートモーダルメインコンポーネント
 */
const ChartModal: React.FC<ChartModalProps> = ({ isOpen, onClose, result }) => {
  const [activeTab, setActiveTab] = useState<TabType>("equity");

  // Escapeキーでモーダルを閉じる
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener("keydown", handleKeyDown);
      // スクロールを無効化
      document.body.style.overflow = "hidden";
    }

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "unset";
    };
  }, [isOpen, onClose]);

  // データの変換
  const chartData = React.useMemo(() => {
    if (!result.equity_curve || !result.trade_history) {
      return { equity: [], trades: [] };
    }

    return {
      equity: transformEquityCurve(result.equity_curve),
      trades: transformTradeHistory(result.trade_history),
    };
  }, [result]);

  // モーダルが閉じている場合は何も表示しない
  if (!isOpen) {
    return null;
  }

  // オーバーレイクリックでモーダルを閉じる
  const handleOverlayClick = (event: React.MouseEvent) => {
    if (event.target === event.currentTarget) {
      onClose();
    }
  };

  // レスポンシブ対応のクラス
  const modalClasses = `
    fixed inset-0 z-50 flex items-center justify-center p-4
    sm:p-6 md:p-8
  `;

  const contentClasses = `
    bg-gray-900 rounded-lg shadow-2xl border border-gray-700
    w-full h-full max-w-7xl max-h-[90vh]
    sm:w-full sm:h-auto sm:max-h-[85vh]
    md:w-[95vw] md:h-[85vh]
    lg:w-[90vw] lg:h-[80vh]
    flex flex-col
  `;

  return (
    <div
      data-testid="modal-overlay"
      className={modalClasses}
      style={{ backgroundColor: "rgba(0, 0, 0, 0.75)" }}
      onClick={handleOverlayClick}
    >
      <div
        data-testid="modal-content"
        className={contentClasses}
        onClick={(e) => e.stopPropagation()}
      >
        {/* ヘッダー */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div>
            <h2 className="text-xl font-semibold text-white">チャート分析</h2>
            <p className="text-sm text-gray-400 mt-1">
              {result.strategy_name} - {result.symbol} ({result.timeframe})
            </p>
          </div>
          <button
            data-testid="close-button"
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
            title="閉じる"
          >
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
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* タブナビゲーション */}
        <div className="flex space-x-1 p-6 pb-0">
          {tabs.map((tab) => (
            <TabButton
              key={tab.id}
              tab={tab}
              isActive={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
            />
          ))}
        </div>

        {/* チャートコンテンツ */}
        <div className="flex-1 p-6 overflow-auto">
          {activeTab === "equity" && (
            <EquityCurveChart
              data={chartData.equity}
              initialCapital={result.initial_capital}
              buyHoldReturn={
                result.performance_metrics.buy_hold_return || undefined
              }
              showBuyHold={!!result.performance_metrics.buy_hold_return}
              title="資産曲線"
              subtitle="時系列での資産推移とBuy & Hold比較"
              height={500}
            />
          )}

          {activeTab === "drawdown" && (
            <DrawdownChart
              data={chartData.equity}
              maxDrawdown={
                Math.abs(result.performance_metrics.max_drawdown || 0) * 100
              }
              title="ドローダウン分析"
              subtitle="最大下落期間と回復パターンの可視化"
              height={500}
            />
          )}

          {activeTab === "trades" && (
            <TradeScatterChart
              data={chartData.trades}
              title="取引分析"
              subtitle="利益/損失の分布と取引パフォーマンス"
              height={500}
            />
          )}

          {activeTab === "distribution" && (
            <ReturnsDistributionChart
              data={result.trade_history || []}
              title="リターン分布"
              subtitle="取引リターンの統計分布と正規性の検証"
              height={500}
            />
          )}

          {activeTab === "heatmap" && (
            <MonthlyReturnsHeatmap
              data={result.equity_curve || []}
              title="月次リターンヒートマップ"
              subtitle="月別パフォーマンスの季節性分析"
              height={500}
            />
          )}
        </div>

        {/* フッター（統計情報） */}
        <div className="p-6 pt-0 border-t border-gray-700">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="text-center">
              <div className="text-gray-400">総リターン</div>
              <div className="text-white font-semibold">
                {result.performance_metrics.total_return
                  ? `${(result.performance_metrics.total_return * 100).toFixed(
                      2
                    )}%`
                  : "N/A"}
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-400">最大ドローダウン</div>
              <div className="text-red-400 font-semibold">
                {result.performance_metrics.max_drawdown
                  ? `${(
                      Math.abs(result.performance_metrics.max_drawdown) * 100
                    ).toFixed(2)}%`
                  : "N/A"}
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-400">勝率</div>
              <div className="text-green-400 font-semibold">
                {result.performance_metrics.win_rate
                  ? `${(result.performance_metrics.win_rate * 100).toFixed(1)}%`
                  : "N/A"}
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-400">総取引数</div>
              <div className="text-blue-400 font-semibold">
                {result.performance_metrics.total_trades || "N/A"}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChartModal;
