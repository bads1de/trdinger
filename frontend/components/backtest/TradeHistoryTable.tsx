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
      return date.toLocaleString("ja-JP", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return dateString;
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("ja-JP", {
      style: "currency",
      currency: "JPY",
      minimumFractionDigits: 0,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const formatNumber = (value: number, decimals: number = 2) => {
    return value.toFixed(decimals);
  };

  const getPnlColor = (pnl: number) => {
    if (pnl > 0) return "text-green-400";
    if (pnl < 0) return "text-red-400";
    return "text-gray-400";
  };

  if (!tradeHistory || tradeHistory.length === 0) {
    return (
      <div className="text-center py-8 text-gray-400">取引履歴がありません</div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-700">
        <thead className="bg-gray-800">
          <tr>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
              エントリー日時
            </th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
              イグジット日時
            </th>
            <th className="px-4 py-3 text-right text-xs font-medium text-gray-300 uppercase tracking-wider">
              エントリー価格
            </th>
            <th className="px-4 py-3 text-right text-xs font-medium text-gray-300 uppercase tracking-wider">
              イグジット価格
            </th>
            <th className="px-4 py-3 text-right text-xs font-medium text-gray-300 uppercase tracking-wider">
              数量
            </th>
            <th className="px-4 py-3 text-right text-xs font-medium text-gray-300 uppercase tracking-wider">
              損益
            </th>
            <th className="px-4 py-3 text-right text-xs font-medium text-gray-300 uppercase tracking-wider">
              リターン率
            </th>
          </tr>
        </thead>
        <tbody className="bg-gray-900 divide-y divide-gray-800">
          {tradeHistory.map((trade, index) => (
            <tr key={index} className="hover:bg-gray-800 transition-colors">
              <td className="px-4 py-3 text-sm text-gray-300">
                {formatDateTime(trade.entry_time)}
              </td>
              <td className="px-4 py-3 text-sm text-gray-300">
                {formatDateTime(trade.exit_time)}
              </td>
              <td className="px-4 py-3 text-sm text-gray-300 text-right">
                {formatCurrency(trade.entry_price)}
              </td>
              <td className="px-4 py-3 text-sm text-gray-300 text-right">
                {formatCurrency(trade.exit_price)}
              </td>
              <td className="px-4 py-3 text-sm text-gray-300 text-right">
                {formatNumber(trade.size, 4)}
              </td>
              <td
                className={`px-4 py-3 text-sm text-right font-medium ${getPnlColor(
                  trade.pnl
                )}`}
              >
                {formatCurrency(trade.pnl)}
              </td>
              <td
                className={`px-4 py-3 text-sm text-right font-medium ${getPnlColor(
                  trade.return_pct
                )}`}
              >
                {formatPercentage(trade.return_pct)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* 統計情報 */}
      <div className="mt-4 p-4 bg-gray-800 rounded-lg">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-400">総取引数:</span>
            <span className="ml-2 text-white font-medium">
              {tradeHistory.length}
            </span>
          </div>
          <div>
            <span className="text-gray-400">勝ちトレード:</span>
            <span className="ml-2 text-green-400 font-medium">
              {tradeHistory.filter((t) => t.pnl > 0).length}
            </span>
          </div>
          <div>
            <span className="text-gray-400">負けトレード:</span>
            <span className="ml-2 text-red-400 font-medium">
              {tradeHistory.filter((t) => t.pnl < 0).length}
            </span>
          </div>
          <div>
            <span className="text-gray-400">総損益:</span>
            <span
              className={`ml-2 font-medium ${getPnlColor(
                tradeHistory.reduce((sum, t) => sum + t.pnl, 0)
              )}`}
            >
              {formatCurrency(tradeHistory.reduce((sum, t) => sum + t.pnl, 0))}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradeHistoryTable;
