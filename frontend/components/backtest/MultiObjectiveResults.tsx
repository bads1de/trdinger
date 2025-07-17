/**
 * 多目的最適化結果表示コンポーネント
 *
 * NSGA-IIアルゴリズムによる多目的最適化の結果（パレート最適解）を表示します。
 */

"use client";

import React, { useState } from "react";
import { TrendingUp, Target, BarChart3, Info } from "lucide-react";
import { MultiObjectiveGAResult, ParetoSolution } from "@/types/optimization";

interface MultiObjectiveResultsProps {
  result: MultiObjectiveGAResult;
  onClose?: () => void;
}

export default function MultiObjectiveResults({
  result,
  onClose,
}: MultiObjectiveResultsProps) {
  const [selectedSolution, setSelectedSolution] =
    useState<ParetoSolution | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  if (!result.success || !result.result) {
    return (
      <div className="bg-red-900/20 border border-red-500 rounded-lg p-4">
        <h3 className="text-red-400 font-semibold mb-2">エラー</h3>
        <p className="text-red-300">{result.message}</p>
      </div>
    );
  }

  const { pareto_front, objectives, best_strategy } = result.result;

  // 目的関数の表示名マッピング
  const objectiveDisplayNames: Record<string, string> = {
    total_return: "総リターン",
    sharpe_ratio: "シャープレシオ",
    max_drawdown: "最大ドローダウン",
    win_rate: "勝率",
    profit_factor: "プロフィットファクター",
    sortino_ratio: "ソルティーノレシオ",
  };

  // 値のフォーマット関数
  const formatValue = (value: number, objective: string): string => {
    if (objective === "total_return" || objective === "max_drawdown") {
      return `${(value * 100).toFixed(2)}%`;
    } else if (objective === "win_rate") {
      return `${(value * 100).toFixed(1)}%`;
    } else {
      return value.toFixed(3);
    }
  };

  // 目的の色を取得
  const getObjectiveColor = (objective: string): string => {
    const colors: Record<string, string> = {
      total_return: "text-green-400",
      sharpe_ratio: "text-blue-400",
      max_drawdown: "text-red-400",
      win_rate: "text-purple-400",
      profit_factor: "text-yellow-400",
      sortino_ratio: "text-cyan-400",
    };
    return colors[objective] || "text-gray-400";
  };

  return (
    <div className="bg-secondary-950 rounded-lg border border-secondary-700">
      {/* ヘッダー */}
      <div className="p-6 border-b border-secondary-700">
        <div className="flex justify-between items-start">
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">
              多目的最適化結果
            </h2>
            <p className="text-secondary-400">
              NSGA-IIアルゴリズムによるパレート最適解
            </p>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="text-secondary-400 hover:text-white transition-colors"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* サマリー */}
      <div className="p-6 border-b border-secondary-700">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-secondary-800 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <Target className="w-5 h-5 text-primary-400" />
              <span className="text-secondary-200 font-medium">最適化目的</span>
            </div>
            <div className="space-y-1">
              {objectives.map((obj) => (
                <div key={obj} className={`text-sm ${getObjectiveColor(obj)}`}>
                  {objectiveDisplayNames[obj] || obj}
                </div>
              ))}
            </div>
          </div>

          <div className="bg-secondary-800 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <BarChart3 className="w-5 h-5 text-green-400" />
              <span className="text-secondary-200 font-medium">
                パレート解数
              </span>
            </div>
            <div className="text-2xl font-bold text-green-400">
              {pareto_front.length}
            </div>
          </div>

          <div className="bg-secondary-800 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <TrendingUp className="w-5 h-5 text-blue-400" />
              <span className="text-secondary-200 font-medium">実行時間</span>
            </div>
            <div className="text-2xl font-bold text-blue-400">
              {result.result.execution_time.toFixed(1)}s
            </div>
          </div>
        </div>
      </div>

      {/* パレート最適解一覧 */}
      <div className="p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-white">
            パレート最適解 ({pareto_front.length}個)
          </h3>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center space-x-2 text-primary-400 hover:text-primary-300 transition-colors"
          >
            <Info className="w-4 h-4" />
            <span>{showDetails ? "詳細を隠す" : "詳細を表示"}</span>
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-secondary-700">
                <th className="text-left p-3 text-secondary-300">解番号</th>
                {objectives.map((obj) => (
                  <th key={obj} className="text-left p-3 text-secondary-300">
                    {objectiveDisplayNames[obj] || obj}
                  </th>
                ))}
                <th className="text-left p-3 text-secondary-300">アクション</th>
              </tr>
            </thead>
            <tbody>
              {pareto_front.map((solution, index) => (
                <tr
                  key={index}
                  className="border-b border-secondary-800 hover:bg-secondary-800/50 transition-colors"
                >
                  <td className="p-3">
                    <span className="font-mono text-primary-400">
                      #{index + 1}
                    </span>
                  </td>
                  {solution.fitness_values.map((value, objIndex) => (
                    <td key={objIndex} className="p-3">
                      <span className={getObjectiveColor(objectives[objIndex])}>
                        {formatValue(value, objectives[objIndex])}
                      </span>
                    </td>
                  ))}
                  <td className="p-3">
                    <button
                      onClick={() =>
                        setSelectedSolution({
                          strategy: solution.strategy,
                          fitness_values: solution.fitness_values,
                          objectives: objectives,
                        })
                      }
                      className="text-primary-400 hover:text-primary-300 transition-colors text-sm"
                    >
                      詳細表示
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* 詳細情報 */}
        {showDetails && (
          <div className="mt-6 bg-secondary-800 rounded-lg p-4">
            <h4 className="text-white font-medium mb-3">
              多目的最適化について
            </h4>
            <div className="text-sm text-secondary-300 space-y-2">
              <p>
                • <strong>パレート最適解</strong>:
                他の解を犠牲にすることなく改善できない解の集合
              </p>
              <p>
                • <strong>NSGA-II</strong>:
                非支配ソートと混雑距離を用いた多目的遺伝的アルゴリズム
              </p>
              <p>
                • <strong>トレードオフ</strong>:
                各解は異なる目的間のバランスを表現
              </p>
            </div>
          </div>
        )}
      </div>

      {/* 選択された解の詳細モーダル */}
      {selectedSolution && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-secondary-900 rounded-lg border border-secondary-700 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
            <div className="p-6 border-b border-secondary-700">
              <h3 className="text-xl font-bold text-white">戦略詳細</h3>
            </div>
            <div className="p-6">
              <div className="space-y-4">
                <div>
                  <h4 className="text-white font-medium mb-2">
                    フィットネス値
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {selectedSolution.fitness_values.map((value, index) => (
                      <div key={index} className="bg-secondary-800 rounded p-3">
                        <div className="text-secondary-300 text-sm">
                          {
                            objectiveDisplayNames[
                              selectedSolution.objectives[index]
                            ]
                          }
                        </div>
                        <div
                          className={`text-lg font-bold ${getObjectiveColor(
                            selectedSolution.objectives[index]
                          )}`}
                        >
                          {formatValue(
                            value,
                            selectedSolution.objectives[index]
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h4 className="text-white font-medium mb-2">戦略設定</h4>
                  <div className="bg-secondary-800 rounded p-3">
                    <pre className="text-xs text-secondary-300 overflow-x-auto">
                      {JSON.stringify(selectedSolution.strategy, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            </div>
            <div className="p-6 border-t border-secondary-700 flex justify-end">
              <button
                onClick={() => setSelectedSolution(null)}
                className="px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 transition-colors"
              >
                閉じる
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
