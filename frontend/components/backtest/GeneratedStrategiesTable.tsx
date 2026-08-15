/**
 * 生成戦略一覧テーブルコンポーネント
 *
 * GAで生成された戦略の一覧を表形式で表示します。
 */

"use client";

import {
  formatNumber,
  formatPercentage,
} from "@/utils/formatters";
import { getValueColorClass } from "@/utils/colorUtils";
import { GeneratedStrategy } from "@/types/generatedStrategy";
import LoadingSpinner from "@/components/common/LoadingSpinner";

interface GeneratedStrategiesTableProps {
  strategies: GeneratedStrategy[];
  loading?: boolean;
  onStrategySelect?: (strategy: GeneratedStrategy) => void;
}

export default function GeneratedStrategiesTable({
  strategies,
  loading = false,
  onStrategySelect,
}: GeneratedStrategiesTableProps) {
  if (loading) {
    return (
      <div className="py-12">
        <LoadingSpinner text="生成戦略を読み込んでいます..." />
      </div>
    );
  }

  if (strategies.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-secondary-400 text-lg">
          生成された戦略がありません
        </p>
        <p className="text-secondary-500 text-sm mt-2">
          オートストラテジーを実行して戦略を生成してください
        </p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-secondary-700">
        <thead className="bg-secondary-800">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider whitespace-nowrap">
              戦略名
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider whitespace-nowrap">
              Fitness
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider whitespace-nowrap">
              検証
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider whitespace-nowrap">
              総リターン
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider whitespace-nowrap">
              SR
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider whitespace-nowrap">
              DD
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider whitespace-nowrap">
              勝率
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-300 uppercase tracking-wider whitespace-nowrap">
              リスク
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-secondary-400 uppercase tracking-wider whitespace-nowrap">
              世代
            </th>
          </tr>
        </thead>
        <tbody className="bg-black divide-y divide-secondary-700">
          {strategies.map((strategy) => (
            <tr
              key={strategy.id}
              onClick={() => onStrategySelect?.(strategy)}
              className="hover:bg-secondary-800 cursor-pointer transition-colors"
            >
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm font-medium text-white">
                  {strategy.name}
                </div>
                {strategy.indicators.length > 0 && (
                  <div className="text-xs text-secondary-500">
                    {strategy.indicators.slice(0, 3).join(" + ")}
                  </div>
                )}
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-secondary-300">
                  {strategy.fitness_score != null
                    ? formatNumber(strategy.fitness_score)
                    : "-"}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                {strategy.validation_passed === null ? (
                  <span className="text-xs text-secondary-500">未検証</span>
                ) : strategy.validation_passed ? (
                  <span className="inline-flex items-center rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800 dark:bg-green-900/30 dark:text-green-300">
                    合格
                  </span>
                ) : (
                  <span className="inline-flex items-center rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-800 dark:bg-red-900/30 dark:text-red-300">
                    不合格
                  </span>
                )}
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div
                  className={`text-sm font-medium ${getValueColorClass(
                    strategy.expected_return
                  )}`}
                >
                  {formatPercentage(strategy.expected_return)}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-secondary-300">
                  {formatNumber(strategy.sharpe_ratio)}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm font-medium text-red-500">
                  {formatPercentage(strategy.max_drawdown)}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-secondary-300">
                  {formatPercentage(strategy.win_rate)}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                {getRiskBadge(strategy.risk_level)}
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-secondary-400">
                  {strategy.generation}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function getRiskBadge(riskLevel: string) {
  switch (riskLevel) {
    case "low":
      return (
        <span className="inline-flex items-center rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800 dark:bg-green-900/30 dark:text-green-300">
          低
        </span>
      );
    case "medium":
      return (
        <span className="inline-flex items-center rounded-full bg-yellow-100 px-2 py-0.5 text-xs font-medium text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300">
          中
        </span>
      );
    case "high":
      return (
        <span className="inline-flex items-center rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-800 dark:bg-red-900/30 dark:text-red-300">
          高
        </span>
      );
    default:
      return (
        <span className="text-xs text-secondary-500">{riskLevel || "-"}</span>
      );
  }
}
