"use client";

import React from "react";
import { Trade } from "@/types/backtest";

interface TradeHistoryTableProps {
  tradeHistory: Trade[];
}

const TradeHistoryTable: React.FC<TradeHistoryTableProps> = ({
  tradeHistory,
}) => {
  // フォーマット関数
  const formatDateTime = (dateString: string) => {
    try {
      const date = new Date(dateString);
      return {
        date: date.toLocaleDateString("ja-JP", {
          year: "numeric",
          month: "2-digit",
          day: "2-digit",
        }),
        time: date.toLocaleTimeString("ja-JP", {
          hour: "2-digit",
          minute: "2-digit",
        }),
      };
    } catch {
      return { date: dateString, time: "" };
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("ja-JP", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 1,
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    const formatted = (value * 100).toFixed(2);
    return `${formatted}%`;
  };

  const formatNumber = (value: number, decimals: number = 4) => {
    return value.toFixed(decimals);
  };

  const getPnlColor = (pnl: number) => {
    if (pnl > 0) return "text-green-400";
    if (pnl < 0) return "text-red-400";
    return "text-secondary-400";
  };

  if (!tradeHistory || tradeHistory.length === 0) {
    return (
      <div className="text-center py-8 text-secondary-400">
        取引履歴がありません
      </div>
    );
  }

  return (
    <>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-700/50">
          <thead className="bg-black/30">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-mono font-medium text-cyan-400 uppercase tracking-wider">
                Entry Time
              </th>
              <th className="px-4 py-3 text-left text-xs font-mono font-medium text-cyan-400 uppercase tracking-wider">
                Exit Time
              </th>
              <th className="px-4 py-3 text-right text-xs font-mono font-medium text-cyan-400 uppercase tracking-wider">
                Entry
              </th>
              <th className="px-4 py-3 text-right text-xs font-mono font-medium text-cyan-400 uppercase tracking-wider">
                Exit
              </th>
              <th className="px-4 py-3 text-right text-xs font-mono font-medium text-cyan-400 uppercase tracking-wider">
                Size
              </th>
              <th className="px-4 py-3 text-right text-xs font-mono font-medium text-cyan-400 uppercase tracking-wider">
                P/L
              </th>
              <th className="px-4 py-3 text-right text-xs font-mono font-medium text-cyan-400 uppercase tracking-wider">
                Return %
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/50">
            {tradeHistory.map((trade, index) => {
              return (
                <tr
                  key={index}
                  className="hover:bg-gray-800/70 transition-colors duration-200"
                >
                  <td className="px-4 py-3 text-sm font-mono text-gray-300 whitespace-nowrap">
                    {formatDateTime(trade.entry_time).date}
                  </td>
                  <td className="px-4 py-3 text-sm font-mono text-gray-300 whitespace-nowrap">
                    {formatDateTime(trade.exit_time).date}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-300 text-right font-mono">
                    {formatCurrency(trade.entry_price)}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-300 text-right font-mono">
                    {formatCurrency(trade.exit_price)}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-300 text-right font-mono">
                    {formatNumber(trade.size, 4)}
                  </td>
                  <td
                    className={`px-4 py-3 text-sm text-right font-mono font-semibold ${getPnlColor(
                      trade.pnl
                    )}`}
                  >
                    {formatCurrency(trade.pnl)}
                  </td>
                  <td
                    className={`px-4 py-3 text-sm text-right font-mono font-semibold ${getPnlColor(
                      trade.return_pct
                    )}`}
                  >
                    {formatPercentage(trade.return_pct)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* 統計情報 */}
      <div className="mt-6 pt-4 border-t border-gray-700/50">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div className="bg-black/20 p-3 rounded-lg text-center">
            <span className="text-sm text-gray-400 font-mono uppercase tracking-wider">
              総取引数
            </span>
            <span className="block text-lg text-white font-semibold font-mono mt-1">
              {tradeHistory.length}
            </span>
          </div>
          <div className="bg-black/20 p-3 rounded-lg text-center">
            <span className="text-sm text-gray-400 font-mono uppercase tracking-wider">
              勝ちトレード
            </span>
            <span className="block text-lg text-green-400 font-semibold font-mono mt-1">
              {tradeHistory.filter((t) => t.pnl > 0).length}
            </span>
          </div>
          <div className="bg-black/20 p-3 rounded-lg text-center">
            <span className="text-sm text-gray-400 font-mono uppercase tracking-wider">
              負けトレード
            </span>
            <span className="block text-lg text-red-400 font-semibold font-mono mt-1">
              {tradeHistory.filter((t) => t.pnl < 0).length}
            </span>
          </div>
          <div className="bg-black/20 p-3 rounded-lg text-center">
            <span className="text-sm text-gray-400 font-mono uppercase tracking-wider">
              総損益
            </span>
            <span
              className={`block text-lg font-semibold font-mono mt-1 ${getPnlColor(
                tradeHistory.reduce((sum, t) => sum + t.pnl, 0)
              )}`}
            >
              {formatCurrency(tradeHistory.reduce((sum, t) => sum + t.pnl, 0))}
            </span>
          </div>
        </div>
      </div>
    </>
  );
};

export default TradeHistoryTable;
