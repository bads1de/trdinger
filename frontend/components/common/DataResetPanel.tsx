/**
 * データリセットパネルコンポーネント
 *
 * データリセット機能をまとめたパネルコンポーネントです。
 * 各種データのリセットボタンと現在のデータ状況を表示します。
 *
 */

"use client";

import React, { useState, useEffect } from "react";
import DataResetButton, {
  DataResetResult,
} from "@/components/button/DataResetButton";
import { useApiCall } from "@/hooks/useApiCall";

/**
 * データ状況の型
 */
interface DataStatus {
  data_counts: {
    ohlcv: number;
    funding_rates: number;
    open_interest: number;
  };
  total_records: number;
  timestamp: string;
}

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
  const [dataStatus, setDataStatus] = useState<DataStatus | null>(null);
  const [resetMessage, setResetMessage] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const apiCall = useApiCall<DataStatus>();

  /**
   * データ状況を取得
   */
  const fetchDataStatus = async () => {
    try {
      setIsLoading(true);
      const result = await apiCall.execute("/api/data-reset/status", {
        method: "GET",
      });

      if (result) {
        setDataStatus(result);
      }
    } catch (error) {
      console.error("データ状況取得エラー:", error);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * リセット完了時のハンドラー
   */
  const handleResetComplete = (result: DataResetResult) => {
    // 成功メッセージを表示
    if (result.success) {
      if (result.total_deleted !== undefined) {
        setResetMessage(
          `✅ ${
            result.message
          } (${result.total_deleted.toLocaleString()}件削除)`
        );
      } else if (result.deleted_count !== undefined) {
        setResetMessage(
          `✅ ${
            result.message
          } (${result.deleted_count.toLocaleString()}件削除)`
        );
      } else {
        setResetMessage(`✅ ${result.message}`);
      }
    } else {
      setResetMessage(`❌ ${result.message}`);
    }

    // データ状況を再取得
    setTimeout(() => {
      fetchDataStatus();
    }, 1000);

    // 親コンポーネントに通知
    onResetComplete?.(result);

    // 10秒後にメッセージをクリア
    setTimeout(() => {
      setResetMessage("");
    }, 10000);
  };

  /**
   * リセットエラー時のハンドラー
   */
  const handleResetError = (error: string) => {
    setResetMessage(`❌ ${error}`);

    // 10秒後にメッセージをクリア
    setTimeout(() => {
      setResetMessage("");
    }, 10000);
  };

  // 初期データ取得
  useEffect(() => {
    if (isVisible) {
      fetchDataStatus();
    }
  }, [isVisible]);

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
                  {dataStatus.data_counts.ohlcv.toLocaleString()}
                </div>
                <div className="text-secondary-600 dark:text-secondary-400">
                  OHLCV
                </div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-green-600 dark:text-green-400">
                  {dataStatus.data_counts.funding_rates.toLocaleString()}
                </div>
                <div className="text-secondary-600 dark:text-secondary-400">
                  FR
                </div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-purple-600 dark:text-purple-400">
                  {dataStatus.data_counts.open_interest.toLocaleString()}
                </div>
                <div className="text-secondary-600 dark:text-secondary-400">
                  OI
                </div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-secondary-900 dark:text-secondary-100">
                  {dataStatus.total_records.toLocaleString()}
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
                onResetComplete={handleResetComplete}
                onResetError={handleResetError}
                disabled={isLoading}
                size="sm"
              />
              <DataResetButton
                resetType="funding-rates"
                onResetComplete={handleResetComplete}
                onResetError={handleResetError}
                disabled={isLoading}
                size="sm"
              />
              <DataResetButton
                resetType="open-interest"
                onResetComplete={handleResetComplete}
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
                onResetComplete={handleResetComplete}
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
              onResetComplete={handleResetComplete}
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
