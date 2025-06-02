/**
 * バックテストページ
 *
 * バックテストの設定、実行、結果表示を行うメインページです。
 */

"use client";

import React, { useState, useEffect } from "react";
import BacktestForm from "@/components/backtest/BacktestForm";
import BacktestResultsTable from "@/components/backtest/BacktestResultsTable";
import PerformanceMetrics from "@/components/backtest/PerformanceMetrics";
import OptimizationForm from "@/components/backtest/OptimizationForm";
import OptimizationResults from "@/components/backtest/OptimizationResults";
import { useApiCall } from "@/hooks/useApiCall";

interface BacktestConfig {
  strategy_name: string;
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  commission_rate: number;
  strategy_config: {
    strategy_type: string;
    parameters: Record<string, number>;
  };
}

interface BacktestResult {
  id: string;
  strategy_name: string;
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  commission_rate: number;
  performance_metrics: {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    profit_factor: number;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    avg_win: number;
    avg_loss: number;
  };
  equity_curve: Array<{
    timestamp: string;
    equity: number;
  }>;
  trade_history: Array<{
    timestamp: string;
    type: "buy" | "sell";
    price: number;
    quantity: number;
    pnl?: number;
  }>;
  created_at: string;
}

export default function BacktestPage() {
  const [activeTab, setActiveTab] = useState<"backtest" | "optimization">(
    "backtest"
  );
  const [latestResult, setLatestResult] = useState<BacktestResult | null>(null);
  const [results, setResults] = useState<BacktestResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<BacktestResult | null>(
    null
  );
  const [optimizationResult, setOptimizationResult] = useState<any>(null);
  const [optimizationType, setOptimizationType] = useState<
    "enhanced" | "multi" | "robustness"
  >("enhanced");

  const { execute: runBacktest, loading: backtestLoading } = useApiCall<{
    result: BacktestResult;
  }>();
  const { execute: fetchResults, loading: resultsLoading } = useApiCall<{
    results: BacktestResult[];
    total: number;
  }>();
  const { execute: deleteResult, loading: deleteLoading } = useApiCall();
  const {
    execute: runEnhancedOptimization,
    loading: enhancedOptimizationLoading,
  } = useApiCall();
  const {
    execute: runMultiObjectiveOptimization,
    loading: multiOptimizationLoading,
  } = useApiCall();
  const { execute: runRobustnessTest, loading: robustnessTestLoading } =
    useApiCall();

  // 結果一覧を取得
  const loadResults = async () => {
    const response = await fetchResults("/api/backtest/results?limit=20");
    if (response?.success) {
      setResults(response.results);
    }
  };

  // ページ読み込み時に結果一覧を取得
  useEffect(() => {
    loadResults();
  }, []);

  // バックテスト実行
  const handleRunBacktest = async (config: BacktestConfig) => {
    const response = await runBacktest("/api/backtest/run", {
      method: "POST",
      body: config,
      onSuccess: (data) => {
        setLatestResult(data.result);
        loadResults(); // 結果一覧を更新
      },
      onError: (error) => {
        console.error("Backtest failed:", error);
      },
    });
  };

  // 結果選択
  const handleResultSelect = (result: BacktestResult) => {
    setSelectedResult(result);
  };

  // 結果削除
  const handleDeleteResult = async (result: BacktestResult) => {
    const response = await deleteResult(`/api/backtest/results/${result.id}`, {
      method: "DELETE",
      confirmMessage: `バックテスト結果「${result.strategy_name}」を削除しますか？\nこの操作は取り消せません。`,
      onSuccess: () => {
        // 削除成功時は一覧を更新
        loadResults();
        // 選択中の結果が削除された場合はクリア
        if (selectedResult?.id === result.id) {
          setSelectedResult(null);
        }
      },
      onError: (error) => {
        console.error("Delete failed:", error);
      },
    });
  };

  // 拡張最適化実行
  const handleEnhancedOptimization = async (config: any) => {
    setOptimizationType("enhanced");
    const response = await runEnhancedOptimization(
      "/api/backtest/optimize-enhanced",
      {
        method: "POST",
        body: config,
        onSuccess: (data) => {
          setOptimizationResult(data.result);
        },
        onError: (error) => {
          console.error("Enhanced optimization failed:", error);
        },
      }
    );
  };

  // マルチ目的最適化実行
  const handleMultiObjectiveOptimization = async (config: any) => {
    setOptimizationType("multi");
    const response = await runMultiObjectiveOptimization(
      "/api/backtest/multi-objective-optimization",
      {
        method: "POST",
        body: config,
        onSuccess: (data) => {
          setOptimizationResult(data.result);
        },
        onError: (error) => {
          console.error("Multi-objective optimization failed:", error);
        },
      }
    );
  };

  // ロバストネステスト実行
  const handleRobustnessTest = async (config: any) => {
    setOptimizationType("robustness");
    const response = await runRobustnessTest("/api/backtest/robustness-test", {
      method: "POST",
      body: config,
      onSuccess: (data) => {
        setOptimizationResult(data.result);
      },
      onError: (error) => {
        console.error("Robustness test failed:", error);
      },
    });
  };

  const isOptimizationLoading =
    enhancedOptimizationLoading ||
    multiOptimizationLoading ||
    robustnessTestLoading;

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* ヘッダー */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">バックテスト</h1>
          <p className="text-gray-400">
            過去データを使用して戦略の有効性を検証します
          </p>

          {/* タブナビゲーション */}
          <div className="flex space-x-1 mt-6">
            <button
              onClick={() => setActiveTab("backtest")}
              className={`px-6 py-3 text-sm font-medium rounded-t-lg transition-colors ${
                activeTab === "backtest"
                  ? "bg-blue-600 text-white border-b-2 border-blue-600"
                  : "bg-gray-700 text-gray-300 hover:bg-gray-600"
              }`}
            >
              基本バックテスト
            </button>
            <button
              onClick={() => setActiveTab("optimization")}
              className={`px-6 py-3 text-sm font-medium rounded-t-lg transition-colors ${
                activeTab === "optimization"
                  ? "bg-blue-600 text-white border-b-2 border-blue-600"
                  : "bg-gray-700 text-gray-300 hover:bg-gray-600"
              }`}
            >
              最適化
            </button>
          </div>
        </div>

        {/* 基本バックテストタブ */}
        {activeTab === "backtest" && (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            {/* 左側: バックテスト設定フォーム */}
            <div className="space-y-6">
              <div className="bg-gray-800 rounded-lg p-6">
                <h2 className="text-xl font-semibold mb-4">バックテスト設定</h2>
                <BacktestForm
                  onSubmit={handleRunBacktest}
                  isLoading={backtestLoading}
                />
              </div>

              {/* 最新結果のプレビュー */}
              {latestResult && (
                <div className="bg-gray-800 rounded-lg p-6">
                  <h2 className="text-xl font-semibold mb-4">最新結果</h2>
                  <PerformanceMetrics result={latestResult} />
                </div>
              )}
            </div>

            {/* 右側: 結果一覧と詳細 */}
            <div className="space-y-6">
              {/* 結果一覧テーブル */}
              <div className="bg-gray-800 rounded-lg p-6">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-semibold">
                    バックテスト結果一覧
                  </h2>
                  <button
                    onClick={loadResults}
                    disabled={resultsLoading}
                    className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                  >
                    {resultsLoading ? "読み込み中..." : "更新"}
                  </button>
                </div>
                <BacktestResultsTable
                  results={results}
                  loading={resultsLoading}
                  onResultSelect={handleResultSelect}
                  onDelete={handleDeleteResult}
                />
              </div>

              {/* 選択された結果の詳細 */}
              {selectedResult && (
                <div className="bg-gray-800 rounded-lg p-6">
                  <h2 className="text-xl font-semibold mb-4">
                    結果詳細 - {selectedResult.strategy_name}
                  </h2>
                  <PerformanceMetrics result={selectedResult} />
                </div>
              )}
            </div>
          </div>
        )}

        {/* 最適化タブ */}
        {activeTab === "optimization" && (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            {/* 左側: 最適化設定フォーム */}
            <div className="space-y-6">
              <OptimizationForm
                onEnhancedOptimization={handleEnhancedOptimization}
                onMultiObjectiveOptimization={handleMultiObjectiveOptimization}
                onRobustnessTest={handleRobustnessTest}
                isLoading={isOptimizationLoading}
              />
            </div>

            {/* 右側: 最適化結果 */}
            <div className="space-y-6">
              {optimizationResult && (
                <OptimizationResults
                  result={optimizationResult}
                  resultType={optimizationType}
                />
              )}

              {!optimizationResult && (
                <div className="bg-gray-800 rounded-lg p-6">
                  <h2 className="text-xl font-semibold mb-4">最適化結果</h2>
                  <p className="text-gray-400 text-center py-8">
                    最適化を実行すると結果がここに表示されます
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ローディング状態 */}
        {backtestLoading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg p-6 text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
              <p className="text-lg font-medium">バックテスト実行中...</p>
              <p className="text-gray-400 text-sm mt-2">
                データ量によっては数分かかる場合があります
              </p>
            </div>
          </div>
        )}

        {/* 最適化ローディング状態 */}
        {isOptimizationLoading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg p-6 text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500 mx-auto mb-4"></div>
              <p className="text-lg font-medium">
                {enhancedOptimizationLoading && "拡張最適化実行中..."}
                {multiOptimizationLoading && "マルチ目的最適化実行中..."}
                {robustnessTestLoading && "ロバストネステスト実行中..."}
              </p>
              <p className="text-gray-400 text-sm mt-2">
                最適化には時間がかかる場合があります
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
