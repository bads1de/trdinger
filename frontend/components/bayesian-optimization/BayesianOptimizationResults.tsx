"use client";

import React, { useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import ActionButton from "@/components/common/ActionButton";
import { BayesianOptimizationResult } from "@/types/bayesian-optimization";
import ProfileSaveDialog from "./ProfileSaveDialog";
import { formatDuration, formatScore } from "@/utils/formatters";

interface BayesianOptimizationResultsProps {
  result: BayesianOptimizationResult;
  onSaveAsProfile?: (profileData: {
    name: string;
    description?: string;
    isDefault?: boolean;
  }) => void;
}

const BayesianOptimizationResults: React.FC<
  BayesianOptimizationResultsProps
> = ({ result, onSaveAsProfile }) => {
  const [showProfileDialog, setShowProfileDialog] = useState(false);

  return (
    <div className="space-y-6">
      {/* サマリー */}
      <Card className="p-6">
        <h3 className="text-xl font-bold mb-4">最適化結果サマリー</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {formatScore(result.best_score)}
            </div>
            <div className="text-sm text-gray-600">ベストスコア</div>
          </div>

          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {result.total_evaluations}
            </div>
            <div className="text-sm text-gray-600">総評価回数</div>
          </div>

          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {formatDuration(result.optimization_time)}
            </div>
            <div className="text-sm text-gray-600">最適化時間</div>
          </div>

          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">
              {result.convergence_info.best_iteration}
            </div>
            <div className="text-sm text-gray-600">ベスト反復</div>
          </div>
        </div>

        <div className="mt-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Badge
              variant={
                result.convergence_info.converged ? "success" : "warning"
              }
            >
              {result.convergence_info.converged ? "収束済み" : "未収束"}
            </Badge>
            <Badge variant="outline">
              {result.optimization_type === "bayesian_ga"
                ? "GAパラメータ"
                : "MLハイパーパラメータ"}
            </Badge>
          </div>

          {onSaveAsProfile && (
            <ActionButton
              onClick={() => setShowProfileDialog(true)}
              variant="secondary"
              size="sm"
            >
              プロファイルとして保存
            </ActionButton>
          )}
        </div>
      </Card>

      {/* プロファイル保存ダイアログ */}
      {onSaveAsProfile && (
        <ProfileSaveDialog
          isOpen={showProfileDialog}
          onClose={() => setShowProfileDialog(false)}
          onSave={onSaveAsProfile}
          optimizationResult={result}
        />
      )}

      {/* 最適パラメータ */}
      <Card className="p-6">
        <h3 className="text-xl font-bold mb-4">最適パラメータ</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(result.best_params).map(([key, value]) => (
            <div
              key={key}
              className="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
            >
              <span className="font-medium">{key}:</span>
              <span className="font-mono text-blue-600 dark:text-blue-400">
                {typeof value === "number" ? value.toFixed(4) : String(value)}
              </span>
            </div>
          ))}
        </div>
      </Card>

      {/* 最適化履歴 */}
      <Card className="p-6">
        <h3 className="text-xl font-bold mb-4">最適化履歴（上位10件）</h3>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="text-left p-2">反復</th>
                <th className="text-left p-2">スコア</th>
                <th className="text-left p-2">パラメータ</th>
              </tr>
            </thead>
            <tbody>
              {result.optimization_history
                .sort((a, b) => b.score - a.score)
                .slice(0, 10)
                .map((entry, index) => (
                  <tr
                    key={entry.iteration}
                    className="border-b hover:bg-gray-50 dark:hover:bg-gray-800"
                  >
                    <td className="p-2">
                      <div className="flex items-center space-x-2">
                        <span>{entry.iteration}</span>
                        {index === 0 && <Badge variant="success">BEST</Badge>}
                      </div>
                    </td>
                    <td className="p-2">
                      <span className="font-mono text-green-600 dark:text-green-400">
                        {formatScore(entry.score)}
                      </span>
                    </td>
                    <td className="p-2">
                      <div className="text-xs space-y-1">
                        {Object.entries(entry.params).map(([key, value]) => (
                          <div key={key} className="flex justify-between">
                            <span className="text-gray-500">{key}:</span>
                            <span className="font-mono">
                              {typeof value === "number"
                                ? value.toFixed(3)
                                : String(value)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* 収束情報 */}
      <Card className="p-6">
        <h3 className="text-xl font-bold mb-4">収束情報</h3>

        <div className="space-y-2">
          <div className="flex justify-between">
            <span>収束状態:</span>
            <Badge
              variant={
                result.convergence_info.converged ? "success" : "warning"
              }
            >
              {result.convergence_info.converged ? "収束済み" : "未収束"}
            </Badge>
          </div>

          <div className="flex justify-between">
            <span>ベスト反復:</span>
            <span className="font-mono">
              {result.convergence_info.best_iteration}
            </span>
          </div>

          <div className="flex justify-between">
            <span>総評価回数:</span>
            <span className="font-mono">{result.total_evaluations}</span>
          </div>

          <div className="flex justify-between">
            <span>最適化効率:</span>
            <span className="font-mono">
              {(
                (result.convergence_info.best_iteration /
                  result.total_evaluations) *
                100
              ).toFixed(1)}
              %
            </span>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default BayesianOptimizationResults;
