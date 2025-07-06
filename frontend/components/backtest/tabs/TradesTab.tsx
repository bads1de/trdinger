import React from "react";
import { BacktestResult } from "@/types/backtest";
import TradeHistoryTable from "../TradeHistoryTable";

interface TradesTabProps {
  result: BacktestResult;
}

export default function TradesTab({ result }: TradesTabProps) {
  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-br from-gray-900/90 to-black/80 rounded-xl p-6 border border-gray-700/50 shadow-2xl backdrop-blur-sm">
        <div className="flex items-center mb-6">
          <div className="w-4 h-4 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full mr-3 animate-pulse shadow-lg shadow-cyan-500/50"></div>
          <h3 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 font-mono tracking-wide">
            TRADE HISTORY
          </h3>
        </div>
        {result.trade_history && result.trade_history.length > 0 ? (
          <TradeHistoryTable tradeHistory={result.trade_history} />
        ) : (
          <div className="text-center py-8 text-gray-400">
            <div className="text-6xl mb-4">📊</div>
            <p className="text-lg">取引履歴がありません</p>
            <p className="text-sm mt-2">
              このバックテストでは取引は実行されませんでした
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
