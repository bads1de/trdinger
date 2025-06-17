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
import OptimizationResults from "@/components/backtest/OptimizationResults";
import OptimizationModal from "@/components/backtest/OptimizationModal";
import AutoStrategyModal from "@/components/backtest/AutoStrategyModal";
import { useApiCall } from "@/hooks/useApiCall";
import { BacktestConfig, BacktestResult } from "@/types/backtest";
import { GAConfig } from "@/types/optimization";
import { useBacktestResults } from "@/hooks/useBacktestResults";
import { useBacktestOptimizations } from "@/hooks/useBacktestOptimizations";
import { useAutoStrategy } from "@/hooks/useAutoStrategy";

export default function BacktestPage() {
  const {
    results,
    selectedResult,
    resultsLoading,
    deleteLoading,
    loadResults,
    handleResultSelect,
    handleDeleteResult,
    setSelectedResult,
  } = useBacktestResults();

  const {
    optimizationResult,
    optimizationType,
    isOptimizationModalOpen,
    currentBacktestConfig,
    enhancedOptimizationLoading,
    multiOptimizationLoading,
    robustnessTestLoading,
    isOptimizationLoading,
    setOptimizationResult,
    setOptimizationType,
    setIsOptimizationModalOpen,
    setCurrentBacktestConfig,
    handleEnhancedOptimization,
    handleMultiObjectiveOptimization,
    handleRobustnessTest,
  } = useBacktestOptimizations();

  const {
    showAutoStrategyModal,
    autoStrategyLoading,
    handleAutoStrategy,
    openAutoStrategyModal,
    setShowAutoStrategyModal,
  } = useAutoStrategy(loadResults);

  const { execute: runBacktest, loading: backtestLoading } = useApiCall<{
    result: BacktestResult;
  }>();

  // バックテスト実行
  const handleRunBacktest = async (config: BacktestConfig) => {
    const response = await runBacktest("/api/backtest/run", {
      method: "POST",
      body: config,
      onSuccess: (data) => {
        loadResults(); // 結果一覧を更新
      },
      onError: (error) => {
        console.error("Backtest failed:", error);
      },
    });
  };

  // GA戦略生成実行
  const handleGAGeneration = async (config: any) => {
    console.log("GA戦略生成開始:", config);
    // GA実行は別途進捗表示で管理されるため、ここでは設定のログ出力のみ
  };

  return (
    <div className="min-h-screen bg-black text-white">
      <div className="container mx-auto px-4 py-8">
        {/* ヘッダー */}
        <div className="mb-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold mb-2">バックテスト</h1>
              <p className="text-secondary-400">
                過去データを使用して戦略の有効性を検証します
              </p>
            </div>
            {/* オートストラテジーへのリンクボタン */}
            <div className="flex items-center gap-3">
              <button onClick={openAutoStrategyModal} className="btn-primary">
                🚀 オートストラテジーで生成
              </button>
            </div>
          </div>
        </div>

        {/* メインコンテンツ */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* 左側: バックテスト設定フォーム */}
          <div className="space-y-6">
            <div className="bg-black rounded-lg p-6 border border-secondary-700">
              <h2 className="text-xl font-semibold mb-4">バックテスト設定</h2>
              <BacktestForm
                onSubmit={handleRunBacktest}
                onConfigChange={setCurrentBacktestConfig}
                isLoading={backtestLoading}
              />
            </div>

            {/* 最適化結果 */}
            {optimizationResult && (
              <div className="bg-black rounded-lg p-6 border border-secondary-700">
                <h2 className="text-xl font-semibold mb-4">
                  {optimizationType === "enhanced" && "拡張最適化結果"}
                  {optimizationType === "multi" && "マルチ目的最適化結果"}
                  {optimizationType === "robustness" &&
                    "ロバストネステスト結果"}
                </h2>
                <OptimizationResults
                  result={optimizationResult}
                  resultType={optimizationType}
                />
              </div>
            )}
          </div>

          {/* 右側: 結果一覧と詳細 */}
          <div className="space-y-6">
            {/* 結果一覧テーブル */}
            <div className="bg-secondary-950 rounded-lg p-6 border border-secondary-700">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold">バックテスト結果一覧</h2>
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
              <div className="bg-secondary-950 rounded-lg p-6 border border-secondary-700">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-semibold">
                    結果詳細 - {selectedResult.strategy_name}
                  </h2>
                  <button
                    onClick={() => setIsOptimizationModalOpen(true)}
                    className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors font-medium text-sm"
                  >
                    🔧 最適化
                  </button>
                </div>
                <PerformanceMetrics result={selectedResult} />
              </div>
            )}
          </div>
        </div>

        {/* 最適化設定モーダル */}
        <OptimizationModal
          isOpen={isOptimizationModalOpen}
          onClose={() => setIsOptimizationModalOpen(false)}
          onEnhancedOptimization={handleEnhancedOptimization}
          onMultiObjectiveOptimization={handleMultiObjectiveOptimization}
          onRobustnessTest={handleRobustnessTest}
          onGAGeneration={handleGAGeneration}
          isLoading={isOptimizationLoading}
          selectedResult={selectedResult}
          currentBacktestConfig={currentBacktestConfig}
        />

        {/* オートストラテジーモーダル */}
        <AutoStrategyModal
          isOpen={showAutoStrategyModal}
          onClose={() => setShowAutoStrategyModal(false)}
          onSubmit={handleAutoStrategy}
          isLoading={autoStrategyLoading}
          currentBacktestConfig={currentBacktestConfig}
        />

        {/* ローディング状態 */}
        {backtestLoading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-secondary-950 rounded-lg p-6 text-center border border-secondary-700">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
              <p className="text-lg font-medium">バックテスト実行中...</p>
              <p className="text-secondary-400 text-sm mt-2">
                データ量によっては数分かかる場合があります
              </p>
            </div>
          </div>
        )}

        {/* 最適化ローディング状態 */}
        {isOptimizationLoading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-secondary-950 rounded-lg p-6 text-center border border-secondary-700">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500 mx-auto mb-4"></div>
              <p className="text-lg font-medium">
                {enhancedOptimizationLoading && "拡張最適化実行中..."}
                {multiOptimizationLoading && "マルチ目的最適化実行中..."}
                {robustnessTestLoading && "ロバストネステスト実行中..."}
              </p>
              <p className="text-secondary-400 text-sm mt-2">
                最適化には時間がかかる場合があります
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
