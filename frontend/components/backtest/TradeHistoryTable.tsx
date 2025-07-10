"use client";

import React from "react";
import { Trade } from "@/types/backtest";
import {
  formatDateTime,
  formatCurrency,
  formatPercentage,
  formatNumber,
  getPnlColor,
  getPnlTextColor,
} from "@/utils/formatters";

interface TradeHistoryTableProps {
  tradeHistory: Trade[];
}

const TradeHistoryTable: React.FC<TradeHistoryTableProps> = ({
  tradeHistory,
}) => {
  // ロング/ショートの判定とスタイリング
  const getTradeType = (size: number) => {
    return size > 0 ? "LONG" : "SHORT";
  };

  const getTradeTypeColor = (size: number) => {
    return size > 0 ? "text-green-400" : "text-red-400";
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
      {/* デスクトップ表示 */}
      <div className="hidden lg:block overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-700/50">
          <thead className="bg-black/30">
            <tr>
              <th className="px-2 py-3 text-left text-xs font-mono font-medium text-cyan-400 uppercase tracking-wider">
                Entry Time
              </th>
              <th className="px-2 py-3 text-left text-xs font-mono font-medium text-cyan-400 uppercase tracking-wider">
                Exit Time
              </th>
              <th className="px-2 py-3 text-center text-xs font-mono font-medium text-cyan-400 uppercase tracking-wider">
                Type
              </th>
              <th className="px-2 py-3 text-right text-xs font-mono font-medium text-cyan-400 uppercase tracking-wider">
                Entry
              </th>
              <th className="px-2 py-3 text-right text-xs font-mono font-medium text-cyan-400 uppercase tracking-wider">
                Exit
              </th>
              <th className="px-2 py-3 text-right text-xs font-mono font-medium text-cyan-400 uppercase tracking-wider">
                Size
              </th>
              <th className="px-2 py-3 text-right text-xs font-mono font-medium text-cyan-400 uppercase tracking-wider">
                P/L
              </th>
              <th className="px-2 py-3 text-right text-xs font-mono font-medium text-cyan-400 uppercase tracking-wider">
                Ret%
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/50">
            {tradeHistory.map((trade, index) => {
              const entryDateTime = formatDateTime(trade.entry_time);
              const exitDateTime = formatDateTime(trade.exit_time);

              return (
                <tr
                  key={index}
                  className="hover:bg-gray-800/70 transition-colors duration-200"
                >
                  <td className="px-2 py-3 text-xs font-mono text-gray-300 whitespace-nowrap">
                    <div className="flex flex-col">
                      <span className="text-gray-300">
                        {entryDateTime.date}
                      </span>
                      <span className="text-gray-500 text-xs">
                        {entryDateTime.time}
                      </span>
                    </div>
                  </td>
                  <td className="px-2 py-3 text-xs font-mono text-gray-300 whitespace-nowrap">
                    <div className="flex flex-col">
                      <span className="text-gray-300">{exitDateTime.date}</span>
                      <span className="text-gray-500 text-xs">
                        {exitDateTime.time}
                      </span>
                    </div>
                  </td>
                  <td className={`px-2 py-3 text-xs text-center font-mono font-semibold ${getTradeTypeColor(trade.size)}`}>
                    {getTradeType(trade.size)}
                  </td>
                  <td className="px-2 py-3 text-xs text-gray-300 text-right font-mono">
                    {formatCurrency(trade.entry_price)}
                  </td>
                  <td className="px-2 py-3 text-xs text-gray-300 text-right font-mono">
                    {formatCurrency(trade.exit_price)}
                  </td>
                  <td className="px-2 py-3 text-xs text-gray-300 text-right font-mono">
                    {formatNumber(Math.abs(trade.size), 0, 6)}
                  </td>
                  <td
                    className={`px-2 py-3 text-xs text-right font-mono font-semibold ${getPnlColor(
                      trade.pnl
                    )}`}
                  >
                    {formatCurrency(trade.pnl)}
                  </td>
                  <td
                    className={`px-2 py-3 text-xs text-right font-mono font-semibold ${getPnlColor(
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

      {/* モバイル表示 */}
      <div className="lg:hidden space-y-3">
        {tradeHistory.map((trade, index) => {
          const entryDateTime = formatDateTime(trade.entry_time);
          const exitDateTime = formatDateTime(trade.exit_time);

          return (
            <div
              key={index}
              className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50"
            >
              <div className="flex justify-between items-start mb-3">
                <div className="flex items-center gap-2">
                  <div className="text-xs text-gray-400">取引 #{index + 1}</div>
                  <div className={`text-xs font-mono font-semibold px-2 py-1 rounded ${getTradeTypeColor(trade.size)} bg-gray-700/50`}>
                    {getTradeType(trade.size)}
                  </div>
                </div>
                <div
                  className={`text-sm font-mono font-semibold ${getPnlTextColor(
                    trade.pnl
                  )}`}
                >
                  {formatCurrency(trade.pnl)} (
                  {formatPercentage(trade.return_pct)})
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <div className="text-gray-400 mb-1">エントリー</div>
                  <div className="font-mono text-gray-300">
                    {entryDateTime.date}
                  </div>
                  <div className="font-mono text-gray-500">
                    {entryDateTime.time}
                  </div>
                  <div className="font-mono text-gray-300 mt-1">
                    {formatCurrency(trade.entry_price)}
                  </div>
                </div>

                <div>
                  <div className="text-gray-400 mb-1">エグジット</div>
                  <div className="font-mono text-gray-300">
                    {exitDateTime.date}
                  </div>
                  <div className="font-mono text-gray-500">
                    {exitDateTime.time}
                  </div>
                  <div className="font-mono text-gray-300 mt-1">
                    {formatCurrency(trade.exit_price)}
                  </div>
                </div>
              </div>

              <div className="mt-3 pt-3 border-t border-gray-700/50">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-400">サイズ:</span>
                  <span className="font-mono text-gray-300">
                    {formatNumber(Math.abs(trade.size), 0, 4)}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
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
              className={`block text-lg font-semibold font-mono mt-1 ${getPnlTextColor(
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
