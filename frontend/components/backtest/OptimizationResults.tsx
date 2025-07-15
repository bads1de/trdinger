/**
 * 最適化結果表示コンポーネント
 *
 * 拡張バックテスト最適化の結果を表示します。
 * ヒートマップ、パラメータ分析、ロバストネス評価などを含みます。
 */

"use client";

import React, { useState } from "react";
import TabButton from "../common/TabButton";
import {
  formatNumber,
  formatPercentage,
  getValueColorClass,
} from "@/utils/formatters";

interface OptimizationResult {
  strategy_name: string;
  symbol: string;
  timeframe: string;
  initial_capital: number;
  performance_metrics: {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    profit_factor: number;
    total_trades: number;
  };
  optimized_parameters?: Record<string, any>;
  heatmap_summary?: {
    best_combination: any;
    best_value: number;
    worst_combination: any;
    worst_value: number;
    mean_value: number;
    std_value: number;
    total_combinations: number;
  };
  optimization_details?: {
    method: string;
    n_calls: number;
    best_value: number;
    convergence?: {
      initial_value: number;
      final_value: number;
      improvement: number;
      convergence_rate: number;
      plateau_detection: boolean;
    };
  };
  optimization_metadata?: {
    method: string;
    maximize: string;
    parameter_space_size: number;
    optimization_timestamp: string;
  };
  multi_objective_details?: {
    objectives: string[];
    weights: number[];
    individual_scores: Record<string, number>;
  };
  robustness_analysis?: {
    robustness_score: number;
    successful_periods: number;
    failed_periods: number;
    performance_statistics: Record<
      string,
      {
        mean: number;
        std: number;
        min: number;
        max: number;
        consistency_score: number;
      }
    >;
    parameter_stability: Record<
      string,
      {
        mean: number;
        std: number;
        coefficient_of_variation: number;
      }
    >;
  };
  individual_results?: Record<string, any>;
  total_periods?: number;
}

interface OptimizationResultsProps {
  result: OptimizationResult | null;
  resultType: "enhanced" | "multi" | "robustness";
}

export default function OptimizationResults({
  result,
  resultType,
}: OptimizationResultsProps) {
  const [activeTab, setActiveTab] = useState<
    "overview" | "details" | "analysis"
  >("overview");

  if (!result) {
    return (
      <div className="bg-secondary-950 rounded-lg p-6 border border-secondary-700">
        <p className="text-secondary-400 text-center">最適化結果がありません</p>
      </div>
    );
  }

  const MetricCard = ({
    title,
    value,
    color,
    description,
  }: {
    title: string;
    value: string;
    color?: string;
    description?: string;
  }) => (
    <div className="bg-secondary-700 rounded-lg p-4 border border-secondary-700">
      <h4 className="text-sm font-medium text-secondary-300 mb-1">{title}</h4>
      <p className={`text-2xl font-bold ${color || "text-white"}`}>{value}</p>
      {description && (
        <p className="text-xs text-secondary-400 mt-1">{description}</p>
      )}
    </div>
  );

  return (
    <div className="bg-secondary-950 rounded-lg p-6 border border-secondary-700">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">
          {resultType === "enhanced" && "拡張最適化結果"}
          {resultType === "multi" && "マルチ目的最適化結果"}
          {resultType === "robustness" && "ロバストネステスト結果"}
        </h2>
        <div className="text-sm text-secondary-400">
          {result.strategy_name} | {result.symbol} | {result.timeframe}
        </div>
      </div>

      {/* タブナビゲーション */}
      <div className="flex space-x-2 mb-6">
        <TabButton
          label="概要"
          isActive={activeTab === "overview"}
          onClick={() => setActiveTab("overview")}
        />
        <TabButton
          label="詳細"
          isActive={activeTab === "details"}
          onClick={() => setActiveTab("details")}
        />
        <TabButton
          label="分析"
          isActive={activeTab === "analysis"}
          onClick={() => setActiveTab("analysis")}
        />
      </div>

      {/* 概要タブ */}
      {activeTab === "overview" && (
        <div className="space-y-6">
          {/* パフォーマンス指標 */}
          <div>
            <h3 className="text-lg font-medium mb-4">パフォーマンス指標</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              <MetricCard
                title="総リターン"
                value={formatPercentage(
                  result.performance_metrics.total_return / 100
                )}
                color={getValueColorClass(
                  result.performance_metrics.total_return
                )}
              />
              <MetricCard
                title="シャープレシオ"
                value={formatNumber(result.performance_metrics.sharpe_ratio, 3)}
                color={getValueColorClass(
                  result.performance_metrics.sharpe_ratio
                )}
              />
              <MetricCard
                title="最大ドローダウン"
                value={formatPercentage(
                  result.performance_metrics.max_drawdown / 100
                )}
                color={getValueColorClass(
                  result.performance_metrics.max_drawdown,
                  { invert: true }
                )}
              />
              <MetricCard
                title="勝率"
                value={formatPercentage(
                  result.performance_metrics.win_rate / 100
                )}
                color={getValueColorClass(result.performance_metrics.win_rate)}
              />
              <MetricCard
                title="プロフィットファクター"
                value={formatNumber(
                  result.performance_metrics.profit_factor,
                  3
                )}
                color={getValueColorClass(
                  result.performance_metrics.profit_factor,
                  { threshold: 1 }
                )}
              />
              <MetricCard
                title="総取引数"
                value={result.performance_metrics.total_trades.toString()}
              />
            </div>
          </div>

          {/* 最適化されたパラメータ */}
          {result.optimized_parameters && (
            <div>
              <h3 className="text-lg font-medium mb-4">
                最適化されたパラメータ
              </h3>
              <div className="bg-secondary-700 rounded-lg p-4 border border-secondary-700">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(result.optimized_parameters).map(
                    ([key, value]) => (
                      <div key={key} className="text-center">
                        <p className="text-sm text-secondary-300">{key}</p>
                        <p className="text-xl font-bold text-blue-400">
                          {value}
                        </p>
                      </div>
                    )
                  )}
                </div>
              </div>
            </div>
          )}

          {/* マルチ目的最適化の詳細 */}
          {result.multi_objective_details && (
            <div>
              <h3 className="text-lg font-medium mb-4">マルチ目的最適化詳細</h3>
              <div className="bg-secondary-700 rounded-lg p-4 border border-secondary-700">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-medium text-secondary-300 mb-2">
                      目的関数と重み
                    </h4>
                    <div className="space-y-1">
                      {result.multi_objective_details.objectives.map(
                        (obj, index) => (
                          <div key={obj} className="flex justify-between">
                            <span className="text-sm text-secondary-300">
                              {obj}
                            </span>
                            <span className="text-sm text-blue-400">
                              {formatNumber(
                                result.multi_objective_details!.weights[index],
                                1
                              )}
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-secondary-300 mb-2">
                      個別スコア
                    </h4>
                    <div className="space-y-1">
                      {Object.entries(
                        result.multi_objective_details.individual_scores
                      ).map(([obj, score]) => (
                        <div key={obj} className="flex justify-between">
                          <span className="text-sm text-secondary-300">
                            {obj}
                          </span>
                          <span className="text-sm text-green-400">
                            {formatNumber(score, 3)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ロバストネス分析 */}
          {result.robustness_analysis && (
            <div>
              <h3 className="text-lg font-medium mb-4">ロバストネス分析</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <MetricCard
                  title="ロバストネススコア"
                  value={formatNumber(
                    result.robustness_analysis.robustness_score,
                    3
                  )}
                  color={getValueColorClass(
                    result.robustness_analysis.robustness_score
                  )}
                  description="0-1の範囲で、1に近いほど安定"
                />
                <MetricCard
                  title="成功期間"
                  value={`${result.robustness_analysis.successful_periods}/${
                    result.total_periods || 0
                  }`}
                  color="text-green-400"
                />
                <MetricCard
                  title="失敗期間"
                  value={result.robustness_analysis.failed_periods.toString()}
                  color={
                    result.robustness_analysis.failed_periods > 0
                      ? "text-red-400"
                      : "text-green-400"
                  }
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* 詳細タブ */}
      {activeTab === "details" && (
        <div className="space-y-6">
          {/* ヒートマップサマリー */}
          {result.heatmap_summary && (
            <div>
              <h3 className="text-lg font-medium mb-4">ヒートマップサマリー</h3>
              <div className="bg-secondary-700 rounded-lg p-4 border border-secondary-700">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-medium text-secondary-300 mb-2">
                      最適な組み合わせ
                    </h4>
                    <p className="text-lg font-bold text-green-400">
                      {JSON.stringify(result.heatmap_summary.best_combination)}
                    </p>
                    <p className="text-sm text-secondary-400">
                      値: {formatNumber(result.heatmap_summary.best_value, 3)}
                    </p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-secondary-300 mb-2">
                      最悪な組み合わせ
                    </h4>
                    <p className="text-lg font-bold text-red-400">
                      {JSON.stringify(result.heatmap_summary.worst_combination)}
                    </p>
                    <p className="text-sm text-secondary-400">
                      値: {formatNumber(result.heatmap_summary.worst_value, 3)}
                    </p>
                  </div>
                </div>
                <div className="mt-4 grid grid-cols-3 gap-4">
                  <MetricCard
                    title="平均値"
                    value={formatNumber(result.heatmap_summary.mean_value, 3)}
                  />
                  <MetricCard
                    title="標準偏差"
                    value={formatNumber(result.heatmap_summary.std_value, 3)}
                  />
                  <MetricCard
                    title="組み合わせ数"
                    value={result.heatmap_summary.total_combinations.toString()}
                  />
                </div>
              </div>
            </div>
          )}

          {/* 最適化詳細 */}
          {result.optimization_details && (
            <div>
              <h3 className="text-lg font-medium mb-4">最適化詳細</h3>
              <div className="bg-secondary-700 rounded-lg p-4 border border-secondary-700">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <MetricCard
                    title="最適化手法"
                    value={result.optimization_details.method.toUpperCase()}
                  />
                  <MetricCard
                    title="関数評価回数"
                    value={result.optimization_details.n_calls.toString()}
                  />
                  <MetricCard
                    title="最終値"
                    value={formatNumber(
                      result.optimization_details.best_value,
                      3
                    )}
                    color="text-green-400"
                  />
                </div>

                {/* 収束分析 */}
                {result.optimization_details.convergence && (
                  <div className="mt-4">
                    <h4 className="text-sm font-medium text-secondary-300 mb-2">
                      収束分析
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <MetricCard
                        title="初期値"
                        value={formatNumber(
                          result.optimization_details.convergence.initial_value,
                          3
                        )}
                      />
                      <MetricCard
                        title="改善度"
                        value={formatNumber(
                          result.optimization_details.convergence.improvement,
                          3
                        )}
                        color={getValueColorClass(
                          result.optimization_details.convergence.improvement
                        )}
                      />
                      <MetricCard
                        title="収束率"
                        value={formatNumber(
                          result.optimization_details.convergence
                            .convergence_rate,
                          6
                        )}
                      />
                      <MetricCard
                        title="プラトー検出"
                        value={
                          result.optimization_details.convergence
                            .plateau_detection
                            ? "はい"
                            : "いいえ"
                        }
                        color={
                          result.optimization_details.convergence
                            .plateau_detection
                            ? "text-yellow-400"
                            : "text-green-400"
                        }
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 最適化メタデータ */}
          {result.optimization_metadata && (
            <div>
              <h3 className="text-lg font-medium mb-4">最適化メタデータ</h3>
              <div className="bg-secondary-700 rounded-lg p-4 border border-secondary-700">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-medium text-secondary-300 mb-2">
                      設定情報
                    </h4>
                    <div className="space-y-1">
                      <div className="flex justify-between">
                        <span className="text-sm text-secondary-300">
                          最大化指標:
                        </span>
                        <span className="text-sm text-blue-400">
                          {result.optimization_metadata.maximize}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-secondary-300">
                          パラメータ空間サイズ:
                        </span>
                        <span className="text-sm text-blue-400">
                          {result.optimization_metadata.parameter_space_size}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-secondary-300">
                          実行時刻:
                        </span>
                        <span className="text-sm text-blue-400">
                          {new Date(
                            result.optimization_metadata.optimization_timestamp
                          ).toLocaleString("ja-JP")}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 個別期間結果（ロバストネステスト用） */}
          {result.individual_results && (
            <div>
              <h3 className="text-lg font-medium mb-4">期間別結果</h3>
              <div className="bg-secondary-700 rounded-lg p-4 border border-secondary-700">
                <div className="space-y-3">
                  {Object.entries(result.individual_results).map(
                    ([periodName, periodResult]: [string, any]) => (
                      <div
                        key={periodName}
                        className="border-b border-secondary-700 pb-3 last:border-b-0"
                      >
                        <h4 className="text-sm font-medium text-secondary-300 mb-2">
                          {periodName}
                        </h4>
                        {periodResult.error ? (
                          <p className="text-sm text-red-400">
                            エラー: {periodResult.error}
                          </p>
                        ) : (
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                            {periodResult.optimized_parameters &&
                              Object.entries(
                                periodResult.optimized_parameters
                              ).map(([param, value]: [string, any]) => (
                                <div key={param} className="text-center">
                                  <p className="text-xs text-secondary-400">
                                    {param}
                                  </p>
                                  <p className="text-sm font-bold text-blue-400">
                                    {value}
                                  </p>
                                </div>
                              ))}
                            {periodResult.performance_metrics && (
                              <>
                                <div className="text-center">
                                  <p className="text-xs text-secondary-400">
                                    シャープレシオ
                                  </p>
                                  <p className="text-sm font-bold text-green-400">
                                    {formatNumber(
                                      periodResult.performance_metrics
                                        .sharpe_ratio,
                                      3
                                    )}
                                  </p>
                                </div>
                                <div className="text-center">
                                  <p className="text-xs text-secondary-400">
                                    リターン
                                  </p>
                                  <p className="text-sm font-bold text-green-400">
                                    {formatPercentage(
                                      periodResult.performance_metrics
                                        .total_return / 100
                                    )}
                                  </p>
                                </div>
                              </>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 分析タブ */}
      {activeTab === "analysis" && (
        <div className="space-y-6">
          {/* パフォーマンス統計（ロバストネステスト用） */}
          {result.robustness_analysis?.performance_statistics && (
            <div>
              <h3 className="text-lg font-medium mb-4">パフォーマンス統計</h3>
              <div className="bg-secondary-700 rounded-lg p-4 border border-secondary-700">
                <div className="space-y-4">
                  {Object.entries(
                    result.robustness_analysis.performance_statistics
                  ).map(([metric, stats]) => (
                    <div key={metric}>
                      <h4 className="text-sm font-medium text-secondary-300 mb-2">
                        {metric}
                      </h4>
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                        <MetricCard
                          title="平均"
                          value={formatNumber(stats.mean, 3)}
                        />
                        <MetricCard
                          title="標準偏差"
                          value={formatNumber(stats.std, 3)}
                        />
                        <MetricCard
                          title="最小"
                          value={formatNumber(stats.min, 3)}
                          color="text-red-400"
                        />
                        <MetricCard
                          title="最大"
                          value={formatNumber(stats.max, 3)}
                          color="text-green-400"
                        />
                        <MetricCard
                          title="一貫性スコア"
                          value={formatNumber(stats.consistency_score, 3)}
                          color={getValueColorClass(stats.consistency_score)}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* パラメータ安定性（ロバストネステスト用） */}
          {result.robustness_analysis?.parameter_stability && (
            <div>
              <h3 className="text-lg font-medium mb-4">パラメータ安定性</h3>
              <div className="bg-secondary-700 rounded-lg p-4">
                <div className="space-y-4">
                  {Object.entries(
                    result.robustness_analysis.parameter_stability
                  ).map(([param, stats]) => (
                    <div key={param}>
                      <h4 className="text-sm font-medium text-secondary-300 mb-2">
                        {param}
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-4 gap-2">
                        <MetricCard
                          title="平均"
                          value={formatNumber(stats.mean, 1)}
                        />
                        <MetricCard
                          title="標準偏差"
                          value={formatNumber(stats.std, 3)}
                        />
                        <MetricCard
                          title="変動係数"
                          value={formatNumber(
                            stats.coefficient_of_variation,
                            3
                          )}
                          color={
                            stats.coefficient_of_variation < 0.2
                              ? "text-green-400"
                              : "text-yellow-400"
                          }
                          description="0.2未満が理想的"
                        />
                        <MetricCard
                          title="安定性評価"
                          value={
                            stats.coefficient_of_variation < 0.1
                              ? "非常に安定"
                              : stats.coefficient_of_variation < 0.2
                              ? "安定"
                              : stats.coefficient_of_variation < 0.3
                              ? "やや不安定"
                              : "不安定"
                          }
                          color={
                            stats.coefficient_of_variation < 0.1
                              ? "text-green-400"
                              : stats.coefficient_of_variation < 0.2
                              ? "text-blue-400"
                              : stats.coefficient_of_variation < 0.3
                              ? "text-yellow-400"
                              : "text-red-400"
                          }
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* 推奨事項 */}
          <div>
            <h3 className="text-lg font-medium mb-4">推奨事項</h3>
            <div className="bg-secondary-700 rounded-lg p-4">
              <div className="space-y-3">
                {result.performance_metrics.sharpe_ratio > 1.5 && (
                  <div className="flex items-start space-x-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full mt-2"></div>
                    <p className="text-sm text-secondary-300">
                      シャープレシオが1.5を超えており、優秀な戦略です。
                    </p>
                  </div>
                )}
                {result.performance_metrics.max_drawdown < -20 && (
                  <div className="flex items-start space-x-2">
                    <div className="w-2 h-2 bg-yellow-400 rounded-full mt-2"></div>
                    <p className="text-sm text-secondary-300">
                      最大ドローダウンが20%を超えています。リスク管理の見直しを検討してください。
                    </p>
                  </div>
                )}
                {result.robustness_analysis &&
                  result.robustness_analysis.robustness_score > 0.8 && (
                    <div className="flex items-start space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full mt-2"></div>
                      <p className="text-sm text-secondary-300">
                        ロバストネススコアが高く、安定した戦略です。
                      </p>
                    </div>
                  )}
                {result.robustness_analysis &&
                  result.robustness_analysis.failed_periods > 0 && (
                    <div className="flex items-start space-x-2">
                      <div className="w-2 h-2 bg-red-400 rounded-full mt-2"></div>
                      <p className="text-sm text-secondary-300">
                        一部の期間で失敗しています。パラメータの調整や制約条件の見直しを検討してください。
                      </p>
                    </div>
                  )}
                {result.optimization_details?.convergence
                  ?.plateau_detection && (
                  <div className="flex items-start space-x-2">
                    <div className="w-2 h-2 bg-yellow-400 rounded-full mt-2"></div>
                    <p className="text-sm text-secondary-300">
                      最適化がプラトーに達しました。より多くの試行回数や異なるパラメータ範囲を試してみてください。
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
