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
import TabButton from "../../common/TabButton";
import { X } from "lucide-react";

interface ChartModalProps {
  /** モーダルの表示状態 */
  isOpen: boolean;
  /** モーダルを閉じるコールバック */
  onClose: () => void;
  /** バックテスト結果データ */
  result: BacktestResult;
}

type TabType = "equity" | "drawdown" | "trades" | "distribution";

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
];

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
    fixed inset-0 z-50 flex items-center justify-center
  `;

  const contentClasses = `
    bg-black rounded-lg shadow-2xl border border-gray-800
    w-screen h-screen
    sm:w-screen sm:h-screen
    md:w-screen md:h-screen
    lg:w-screen lg:h-screen
    flex flex-col
  `;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="fixed inset-0 bg-black opacity-50"
        onClick={onClose}
      ></div>
      <div className="bg-secondary-900 rounded-lg shadow-xl z-10 w-full max-w-5xl h-[90vh] flex flex-col p-6 space-y-4 border border-secondary-700">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-bold text-white">
            バックテスト結果チャート
          </h2>
          <button
            onClick={onClose}
            className="text-secondary-400 hover:text-white transition-colors"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        {/* タブナビゲーション */}
        <div className="flex space-x-2 border-b border-secondary-700 pb-2 overflow-x-auto">
          {tabs.map((tab) => (
            <TabButton
              key={tab.id}
              label={tab.label}
              isActive={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
            />
          ))}
        </div>

        {/* チャートコンテンツ */}
        <div className="flex-1 h-full">
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
            />
          )}

          {activeTab === "trades" && (
            <TradeScatterChart
              data={chartData.trades}
              title="取引分析"
              subtitle="利益/損失の分布と取引パフォーマンス"
            />
          )}

          {activeTab === "distribution" && (
            <ReturnsDistributionChart
              data={result.trade_history || []}
              title="リターン分布"
              subtitle="取引リターンの統計分布と正規性の検証"
            />
          )}
        </div>

        {/* フッター（統計情報） */}
        <div className="p-4 pt-0 border-t border-gray-800">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="text-center">
              <div className="text-gray-400">総リターン</div>
              <div className="text-white font-semibold">
                {result.performance_metrics.total_return
                  ? `${result.performance_metrics.total_return.toFixed(2)}%`
                  : "N/A"}
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-400">最大ドローダウン</div>
              <div className="text-red-400 font-semibold">
                {result.performance_metrics.max_drawdown
                  ? `${Math.abs(
                      result.performance_metrics.max_drawdown
                    ).toFixed(2)}%`
                  : "N/A"}
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-400">勝率</div>
              <div className="text-green-400 font-semibold">
                {result.performance_metrics.win_rate
                  ? `${result.performance_metrics.win_rate.toFixed(1)}%`
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
