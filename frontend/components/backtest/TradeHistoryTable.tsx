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
      currency: "JPY",
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
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-secondary-700">
        <thead className="bg-secondary-800">
          <tr>
            <th className="px-3 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider">
              エントリー
              <br />
              日時
            </th>
            <th className="px-3 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider">
              イグジット
              <br />
              日時
            </th>
            <th className="px-3 py-3 text-right text-xs font-medium text-secondary-300 uppercase tracking-wider">
              エントリー
              <br />
              価格
            </th>
            <th className="px-3 py-3 text-right text-xs font-medium text-secondary-300 uppercase tracking-wider">
              イグジット
              <br />
              価格
            </th>
            <th className="px-3 py-3 text-right text-xs font-medium text-secondary-300 uppercase tracking-wider">
              数量
            </th>
            <th className="px-3 py-3 text-right text-xs font-medium text-secondary-300 uppercase tracking-wider">
              損益
            </th>
            <th className="px-3 py-3 text-right text-xs font-medium text-secondary-300 uppercase tracking-wider">
              リターン率
            </th>
          </tr>
        </thead>
        <tbody className="bg-black divide-y divide-secondary-800">
          {tradeHistory.map((trade, index) => {
            const entryDateTime = formatDateTime(trade.entry_time);
            const exitDateTime = formatDateTime(trade.exit_time);

            return (
              <tr
                key={index}
                className="hover:bg-secondary-800 transition-colors"
              >
                <td className="px-3 py-3 text-xs">
                  <div className="text-secondary-300">{entryDateTime.date}</div>
                  <div className="text-secondary-400 text-[10px]">
                    {entryDateTime.time}
                  </div>
                </td>
                <td className="px-3 py-3 text-xs">
                  <div className="text-secondary-300">{exitDateTime.date}</div>
                  <div className="text-secondary-400 text-[10px]">
                    {exitDateTime.time}
                  </div>
                </td>
                <td className="px-3 py-3 text-xs text-secondary-300 text-right">
                  {formatCurrency(trade.entry_price)}
                </td>
                <td className="px-3 py-3 text-xs text-secondary-300 text-right">
                  {formatCurrency(trade.exit_price)}
                </td>
                <td className="px-3 py-3 text-xs text-secondary-300 text-right">
                  {formatNumber(trade.size, 4)}
                </td>
                <td
                  className={`px-3 py-3 text-xs text-right font-medium ${getPnlColor(
                    trade.pnl
                  )}`}
                >
                  {formatCurrency(trade.pnl)}
                </td>
                <td
                  className={`px-3 py-3 text-xs text-right font-medium ${getPnlColor(
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

      {/* 統計情報 */}
      <div className="mt-4 p-4 bg-secondary-800 rounded-lg border border-secondary-700">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-secondary-400">総取引数:</span>
            <span className="ml-2 text-white font-medium">
              {tradeHistory.length}
            </span>
          </div>
          <div>
            <span className="text-secondary-400">勝ちトレード:</span>
            <span className="ml-2 text-green-400 font-medium">
              {tradeHistory.filter((t) => t.pnl > 0).length}
            </span>
          </div>
          <div>
            <span className="text-secondary-400">負けトレード:</span>
            <span className="ml-2 text-red-400 font-medium">
              {tradeHistory.filter((t) => t.pnl < 0).length}
            </span>
          </div>
          <div>
            <span className="text-secondary-400">総損益:</span>
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
