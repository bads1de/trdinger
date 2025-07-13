"use client";

import React from "react";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { BayesianOptimizationResult } from "@/types/bayesian-optimization";

interface BayesianOptimizationResultsProps {
  result: BayesianOptimizationResult;
}

const BayesianOptimizationResults: React.FC<BayesianOptimizationResultsProps> = ({
  result,
}) => {
  const formatDuration = (seconds: number): string => {
    if (seconds < 60) {
      return `${seconds.toFixed(1)}秒`;
    } else if (seconds < 3600) {
      return `${(seconds / 60).toFixed(1)}分`;
    } else {
      return `${(seconds / 3600).toFixed(1)}時間`;
    }
  };

  const formatScore = (score: number): string => {
    return score.toFixed(4);
  };

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
            <Badge variant={result.convergence_info.converged ? "success" : "warning"}>
              {result.convergence_info.converged ? "収束済み" : "未収束"}
            </Badge>
            <Badge variant="outline">
              {result.optimization_type === "bayesian_ga" ? "GAパラメータ" : "MLハイパーパラメータ"}
            </Badge>
          </div>
          
          {result.experiment_name && (
            <div className="text-sm text-gray-600">
              実験: {result.experiment_name}
            </div>
          )}
          
          {result.model_type && (
            <div className="text-sm text-gray-600">
              モデル: {result.model_type}
            </div>
          )}
        </div>
      </Card>

      {/* ベストパラメータ */}
      <Card className="p-6">
        <h3 className="text-lg font-bold mb-4">ベストパラメータ</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(result.best_params).map(([key, value]) => (
            <div key={key} className="flex justify-between items-center p-3 bg-gray-50 rounded">
              <span className="font-medium">{key}</span>
              <span className="text-blue-600 font-mono">
                {typeof value === "number" ? value.toFixed(4) : String(value)}
              </span>
            </div>
          ))}
        </div>
      </Card>

      {/* 最適化履歴 */}
      <Card className="p-6">
        <h3 className="text-lg font-bold mb-4">最適化履歴</h3>
        
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
                <tr key={entry.iteration} className="border-b hover:bg-gray-50">
                  <td className="p-2">
                    <div className="flex items-center space-x-2">
                      <span>{entry.iteration}</span>
                      {index === 0 && (
                        <Badge variant="success" size="sm">BEST</Badge>
                      )}
                    </div>
                  </td>
                  <td className="p-2 font-mono">
                    {formatScore(entry.score)}
                  </td>
                  <td className="p-2">
                    <details className="cursor-pointer">
                      <summary className="text-blue-600 hover:text-blue-800">
                        パラメータを表示
                      </summary>
                      <div className="mt-2 p-2 bg-gray-100 rounded text-xs font-mono">
                        {JSON.stringify(entry.params, null, 2)}
                      </div>
                    </details>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          {result.optimization_history.length > 10 && (
            <div className="text-center mt-4 text-sm text-gray-600">
              上位10件を表示中（全{result.optimization_history.length}件）
            </div>
          )}
        </div>
      </Card>

      {/* 収束情報 */}
      <Card className="p-6">
        <h3 className="text-lg font-bold mb-4">収束情報</h3>
        
        <div className="space-y-2">
          <div className="flex justify-between">
            <span>収束状態:</span>
            <Badge variant={result.convergence_info.converged ? "success" : "warning"}>
              {result.convergence_info.converged ? "収束済み" : "未収束"}
            </Badge>
          </div>
          
          <div className="flex justify-between">
            <span>ベスト反復:</span>
            <span className="font-mono">{result.convergence_info.best_iteration}</span>
          </div>
          
          <div className="flex justify-between">
            <span>総評価回数:</span>
            <span className="font-mono">{result.total_evaluations}</span>
          </div>
          
          <div className="flex justify-between">
            <span>最適化効率:</span>
            <span className="font-mono">
              {((result.convergence_info.best_iteration / result.total_evaluations) * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default BayesianOptimizationResults;
