/**
 * データリセットパネルコンポーネント
 *
 * データリセット機能をまとめたパネルコンポーネントです。
 * 各種データのリセットボタンと現在のデータ状況を表示します。
 *
 */

"use client";

import React from "react";
import DataResetButton, {
  DataResetResult,
} from "@/components/button/DataResetButton";
import { useDataReset } from "@/hooks/useDataReset";

/**
 * データリセットパネルのプロパティ
 */
interface DataResetPanelProps {
  /** 選択中のシンボル */
  selectedSymbol?: string;
  /** リセット完了時のコールバック */
  onResetComplete?: (result: DataResetResult) => void;
  /** パネルの表示/非表示 */
  isVisible?: boolean;
  /** パネルを閉じる関数 */
  onClose?: () => void;
}

/**
 * データリセットパネルコンポーネント
 */
const DataResetPanel: React.FC<DataResetPanelProps> = ({
  selectedSymbol,
  onResetComplete,
  isVisible = true,
  onClose,
}) => {
  const {
    dataStatus,
    resetMessage,
    isLoading,
    fetchDataStatus,
    handleResetComplete,
    handleResetError,
  } = useDataReset(isVisible);

  const onResetCompleted = (result: DataResetResult) => {
    handleResetComplete(result);
    onResetComplete?.(result);
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div className="enterprise-card border-warning-200 dark:border-warning-800 bg-warning-50 dark:bg-warning-900/20">
      <div className="p-6">
        {/* ヘッダー */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center">
            <span className="text-2xl mr-3">🗑️</span>
            <div>
              <h3 className="text-lg font-semibold text-warning-800 dark:text-warning-200">
                データリセット
              </h3>
              <p className="text-sm text-warning-600 dark:text-warning-400">
                データベース内のデータを削除します（取り消し不可）
              </p>
            </div>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="text-warning-600 hover:text-warning-800 dark:text-warning-400 dark:hover:text-warning-200"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          )}
        </div>

        {/* データ状況表示 */}
        {dataStatus && (
          <div className="mb-6 p-4 bg-white dark:bg-secondary-800 rounded-lg border border-warning-200 dark:border-warning-700">
            <h4 className="text-sm font-medium text-secondary-900 dark:text-secondary-100 mb-3">
              📊 現在のデータ状況
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="text-center">
                <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
                  {(dataStatus.data_counts?.ohlcv ?? 0).toLocaleString()}
                </div>
                <div className="text-secondary-600 dark:text-secondary-400">
                  OHLCV
                </div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-green-600 dark:text-green-400">
                  {(
                    dataStatus.data_counts?.funding_rates ?? 0
                  ).toLocaleString()}
                </div>
                <div className="text-secondary-600 dark:text-secondary-400">
                  FR
                </div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-purple-600 dark:text-purple-400">
                  {(
                    dataStatus.data_counts?.open_interest ?? 0
                  ).toLocaleString()}
                </div>
                <div className="text-secondary-600 dark:text-secondary-400">
                  OI
                </div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-secondary-900 dark:text-secondary-100">
                  {(dataStatus.total_records ?? 0).toLocaleString()}
                </div>
                <div className="text-secondary-600 dark:text-secondary-400">
                  合計
                </div>
              </div>
            </div>
          </div>
        )}

        {/* リセットボタン群 */}
        <div className="space-y-4">
          {/* 個別リセットボタン */}
          <div>
            <label className="block text-sm font-medium text-warning-800 dark:text-warning-200 mb-2">
              個別データリセット
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <DataResetButton
                resetType="ohlcv"
                onResetComplete={onResetCompleted}
                onResetError={handleResetError}
                disabled={isLoading}
                size="sm"
              />
              <DataResetButton
                resetType="funding-rates"
                onResetComplete={onResetCompleted}
                onResetError={handleResetError}
                disabled={isLoading}
                size="sm"
              />
              <DataResetButton
                resetType="open-interest"
                onResetComplete={onResetCompleted}
                onResetError={handleResetError}
                disabled={isLoading}
                size="sm"
              />
            </div>
          </div>

          {/* シンボル別リセット */}
          {selectedSymbol && (
            <div>
              <label className="block text-sm font-medium text-warning-800 dark:text-warning-200 mb-2">
                シンボル別リセット
              </label>
              <DataResetButton
                resetType="symbol"
                symbol={selectedSymbol}
                onResetComplete={onResetCompleted}
                onResetError={handleResetError}
                disabled={isLoading}
                size="sm"
                className="w-full sm:w-auto"
              />
            </div>
          )}

          {/* 全データリセット */}
          <div className="pt-4 border-t border-warning-200 dark:border-warning-700">
            <label className="block text-sm font-medium text-error-800 dark:text-error-200 mb-2">
              ⚠️ 危険な操作
            </label>
            <DataResetButton
              resetType="all"
              onResetComplete={onResetCompleted}
              onResetError={handleResetError}
              disabled={isLoading}
              size="sm"
              className="w-full sm:w-auto"
            />
          </div>
        </div>

        {/* ステータスメッセージ */}
        {resetMessage && (
          <div className="mt-4 p-3 bg-white dark:bg-secondary-800 rounded-lg border border-warning-200 dark:border-warning-700">
            <div className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
              {resetMessage}
            </div>
          </div>
        )}

        {/* 更新ボタン */}
        <div className="mt-4 flex justify-end">
          <button
            onClick={fetchDataStatus}
            disabled={isLoading}
            className="text-sm text-warning-600 hover:text-warning-800 dark:text-warning-400 dark:hover:text-warning-200 disabled:opacity-50"
          >
            {isLoading ? "更新中..." : "🔄 状況を更新"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default DataResetPanel;
