/**
 * バックテストページ
 *
 * バックテストの設定、実行、結果表示を行うメインページです。
 */

"use client";

import React, { useState } from "react";
import { Info } from "lucide-react";
import ActionButton from "@/components/common/ActionButton";

import BacktestResultsTable from "@/components/backtest/BacktestResultsTable";
import PerformanceMetrics from "@/components/backtest/PerformanceMetrics";
import AutoStrategyModal from "@/components/backtest/AutoStrategyModal";
import AutoStrategyExplanationModal from "@/components/backtest/AutoStrategyExplanationModal";
import ConfirmDialog from "@/components/common/ConfirmDialog";

import LoadingSpinner from "@/components/common/LoadingSpinner";

import { useBacktestResults } from "@/hooks/useBacktestResults";
import { useAutoStrategy } from "@/hooks/useAutoStrategy";
import { ExperimentProgressList } from "@/components/common/ExperimentProgress";
import { BacktestResult } from "@/types/backtest";

function BacktestPageContent() {
  const {
    results,
    selectedResult,
    resultsLoading,
    deleteAllLoading,
    loadResults,
    handleResultSelect,
    handleDeleteResult,
    handleDeleteAllResults,
  } = useBacktestResults();

  const {
    showAutoStrategyModal,
    autoStrategyLoading,
    handleAutoStrategy,
    openAutoStrategyModal,
    setShowAutoStrategyModal,
    runningExperiments,
  } = useAutoStrategy(loadResults);

  const [isExplanationModalOpen, setIsExplanationModalOpen] = useState(false);
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);
  const [isDeleteAllConfirmOpen, setIsDeleteAllConfirmOpen] = useState(false);
  const [resultToDelete, setResultToDelete] = useState<BacktestResult | null>(null);

  const onDeleteClick = (result: BacktestResult) => {
    setResultToDelete(result);
    setIsDeleteConfirmOpen(true);
  };

  const onDeleteAllClick = () => {
    setIsDeleteAllConfirmOpen(true);
  };

  return (
    <div className="min-h-screen from-gray-900  text-white">
      <div className="container mx-auto px-4 py-8">
        {/* ヘッダー */}
        <div className="mb-8 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold mb-2">バックテスト</h1>
              <p className="text-secondary-400">
                過去データを使用して戦略の有効性を検証します
              </p>
            </div>
            <div className="flex items-center gap-3">
              <ActionButton
                onClick={openAutoStrategyModal}
                variant="secondary"
                icon={<span className="text-lg">🚀</span>}
              >
                オートストラテジーで生成
              </ActionButton>
              <Info
                className="h-6 w-6 text-gray-400 cursor-pointer hover:text-white"
                onClick={() => setIsExplanationModalOpen(true)}
              />
            </div>
          </div>

          {/* 実行中の実験進捗 */}
          {runningExperiments.size > 0 && (
            <ExperimentProgressList experiments={runningExperiments} />
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-secondary-950 rounded-lg p-6 border border-secondary-700 lg:col-span-1">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">バックテスト結果一覧</h2>
              <div className="flex gap-2">
                <ActionButton
                  onClick={() => loadResults()}
                  loading={resultsLoading}
                  loadingText="読み込み中..."
                  variant="primary"
                >
                  更新
                </ActionButton>
                <ActionButton
                  onClick={onDeleteAllClick}
                  disabled={deleteAllLoading || results.length === 0}
                  loading={deleteAllLoading}
                  loadingText="削除中..."
                  variant="danger"
                >
                  すべて削除
                </ActionButton>
              </div>
            </div>
            <BacktestResultsTable
              results={results}
              loading={resultsLoading}
              onResultSelect={handleResultSelect}
              onDelete={onDeleteClick}
            />
          </div>

          <div className="space-y-6 lg:col-span-1">
            {selectedResult ? (
              <div className="bg-secondary-950 rounded-lg p-6 border border-secondary-700">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-semibold">
                    結果詳細 - {selectedResult.strategy_name}
                  </h2>
                </div>
                <PerformanceMetrics result={selectedResult} />
              </div>
            ) : (
              <div className="flex items-center justify-center h-full bg-secondary-950 rounded-lg p-6 border border-secondary-700 border-dashed">
                <p className="text-secondary-400">
                  結果を一覧から選択して詳細を表示
                </p>
              </div>
            )}
          </div>
        </div>

        <AutoStrategyModal
          isOpen={showAutoStrategyModal}
          onClose={() => setShowAutoStrategyModal(false)}
          onSubmit={handleAutoStrategy}
          isLoading={autoStrategyLoading}
          currentBacktestConfig={null}
        />

        <AutoStrategyExplanationModal
          isOpen={isExplanationModalOpen}
          onClose={() => setIsExplanationModalOpen(false)}
        />

        {/* 削除確認ダイアログ */}
        <ConfirmDialog
          open={isDeleteConfirmOpen}
          onOpenChange={setIsDeleteConfirmOpen}
          title="結果削除の確認"
          description={`バックテスト結果「${resultToDelete?.strategy_name}」を削除しますか？この操作は取り消せません。`}
          confirmText="削除"
          cancelText="キャンセル"
          variant="destructive"
          onConfirm={() => resultToDelete && handleDeleteResult(resultToDelete)}
        />

        {/* すべて削除確認ダイアログ */}
        <ConfirmDialog
          open={isDeleteAllConfirmOpen}
          onOpenChange={setIsDeleteAllConfirmOpen}
          title="すべての結果を削除"
          description={`すべてのバックテスト結果を削除しますか？現在${results.length}件の結果があります。この操作は取り消せません。`}
          confirmText="すべて削除"
          cancelText="キャンセル"
          variant="destructive"
          onConfirm={handleDeleteAllResults}
        />
      </div>
    </div>
  );
}

export default function BacktestPage() {
  return (
    <React.Suspense fallback={<LoadingSpinner />}>
      <BacktestPageContent />
    </React.Suspense>
  );
}