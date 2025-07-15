/**
 * バックテスト結果テーブルコンポーネント
 *
 * バックテスト結果の一覧を表形式で表示します。
 */

"use client";

import { formatDateTime } from "@/utils/formatters";
import { BacktestResult } from "@/types/backtest";
import LoadingSpinner from "@/components/common/LoadingSpinner";

interface BacktestResultsTableProps {
  results: BacktestResult[];
  loading?: boolean;
  onResultSelect?: (result: BacktestResult) => void;
  onDelete?: (result: BacktestResult) => Promise<void> | void;
}

export default function BacktestResultsTable({
  results,
  loading = false,
  onResultSelect,
  onDelete,
}: BacktestResultsTableProps) {
  

  const formatPercentage = (value: number | undefined | null) => {
    if (value === undefined || value === null || isNaN(value)) {
      return "N/A";
    }
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

  const getReturnColor = (value: number | undefined | null) => {
    if (value === undefined || value === null || isNaN(value))
      return "text-secondary-400";
    if (value > 0) return "text-green-400";
    if (value < 0) return "text-red-400";
    return "text-gray-400";
  };

  if (loading) {
    return (
      <div className="py-12">
        <LoadingSpinner text="バックテスト結果を読み込んでいます..." />
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-secondary-400 text-lg">
          バックテスト結果がありません
        </p>
        <p className="text-secondary-500 text-sm mt-2">
          バックテストを実行して結果を確認してください
        </p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-secondary-700">
        <thead className="bg-secondary-800">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider">
              戦略名
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider">
              シンボル
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider">
              時間軸
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider">
              総リターン
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider">
              シャープレシオ
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider">
              最大DD
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider">
              勝率
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider">
              取引数
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-400 uppercase tracking-wider">
              実行日時
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider">
              操作
            </th>
          </tr>
        </thead>
        <tbody className="bg-black divide-y divide-secondary-700">
          {results.map((result) => (
            <tr
              key={result.id}
              onClick={() => onResultSelect?.(result)}
              className="hover:bg-secondary-800 cursor-pointer transition-colors"
            >
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm font-medium text-white">
                  {result.strategy_name}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-secondary-300">
                  {result.symbol}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-secondary-300">
                  {result.timeframe}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div
                  className={`text-sm font-medium ${getReturnColor(
                    result.performance_metrics.total_return
                  )}`}
                >
                  {formatPercentage(result.performance_metrics.total_return)}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-secondary-300">
                  {formatNumber(result.performance_metrics.sharpe_ratio)}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-red-400">
                  {formatPercentage(result.performance_metrics.max_drawdown)}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-secondary-300">
                  {formatNumber(result.performance_metrics.win_rate)}%
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-secondary-300">
                  {result.performance_metrics.total_trades}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-secondary-400">
                  {formatDateTime(result.created_at).dateTime}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <button
                  onClick={(e) => {
                    e.stopPropagation(); // 行クリックイベントを防ぐ
                    onDelete?.(result);
                  }}
                  className="text-red-400 hover:text-red-300 transition-colors p-1 rounded hover:bg-red-900/20"
                  title="削除"
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
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                    />
                  </svg>
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
