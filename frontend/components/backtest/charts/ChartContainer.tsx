/**
 * チャート共通コンテナコンポーネント
 *
 * 全てのチャートで共通して使用するコンテナとローディング・エラー処理
 */

"use client";

import React from "react";
import { ChartContainerProps } from "@/types/backtest";
import LoadingSpinner from "@/components/common/LoadingSpinner";
import ErrorDisplay from "@/components/common/ErrorDisplay";

/**
 * ローディングスケルトンコンポーネント
 */
const ChartSkeleton: React.FC<{ height?: number }> = ({ height = 400 }) => (
  <div className="animate-pulse">
    <div className="h-6 bg-secondary-700 rounded mb-4 w-1/3"></div>
    <div className="h-4 bg-secondary-800 rounded mb-6 w-1/2"></div>
    <div
      className="bg-secondary-800 rounded-lg flex items-center justify-center"
      style={{ height: `${height}px` }}
    >
      <LoadingSpinner text="チャートを読み込み中..." />
    </div>
  </div>
);

/**
 * エラー表示コンポーネント
 */
const ChartError: React.FC<{ error: string; height?: number }> = ({
  error,
  height = 400,
}) => (
  <div
    className="flex items-center justify-center"
    style={{ height: `${height}px` }}
  >
    <ErrorDisplay message={error} />
  </div>
);

/**
 * 空データ表示コンポーネント
 */
const ChartEmpty: React.FC<{ height?: number }> = ({ height = 400 }) => (
  <div className="text-center">
    <div className="h-6 text-secondary-400 font-semibold mb-2">
      データがありません
    </div>
    <div
      className="bg-secondary-800/30 border border-secondary-700/30 rounded-lg flex items-center justify-center"
      style={{ height: `${height}px` }}
    >
      <div className="text-secondary-400 text-sm">
        <svg
          className="w-12 h-12 mx-auto mb-2"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1}
            d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
          />
        </svg>
        <div>表示するデータがありません</div>
      </div>
    </div>
  </div>
);

/**
 * チャートヘッダーコンポーネント
 */
const ChartHeader: React.FC<{
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
}> = ({ title, subtitle, actions }) => (
  <div className="flex items-start justify-between mb-4">
    <div>
      <h3 className="text-lg font-semibold text-white mb-1">{title}</h3>
      {subtitle && <p className="text-sm text-secondary-400">{subtitle}</p>}
    </div>
    {actions && <div className="flex items-center space-x-2">{actions}</div>}
  </div>
);

/**
 * チャートコンテナメインコンポーネント
 */
const ChartContainer: React.FC<ChartContainerProps> = ({
  title,
  subtitle,
  children,
  actions,
  data,
  loading = false,
  error,
  height = 400,
  className = "",
  theme = "dark",
}) => {
  // ローディング状態
  if (loading) {
    return (
      <div className={`bg-secondary-800/30 rounded-lg p-6 ${className}`}>
        <ChartHeader title={title} subtitle={subtitle} actions={actions} />
        <ChartSkeleton height={height} />
      </div>
    );
  }

  // エラー状態
  if (error) {
    return (
      <div className={`bg-secondary-800/30 rounded-lg p-6 ${className}`}>
        <ChartHeader title={title} subtitle={subtitle} actions={actions} />
        <ChartError error={error} height={height} />
      </div>
    );
  }

  // 空データ状態
  if (!data || data.length === 0) {
    return (
      <div className={`bg-secondary-800/30 rounded-lg p-6 ${className}`}>
        <ChartHeader title={title} subtitle={subtitle} actions={actions} />
        <ChartEmpty height={height} />
      </div>
    );
  }

  // 正常状態
  return (
    <div className={`bg-gray-800/30 rounded-lg p-6 ${className}`}>
      <ChartHeader title={title} subtitle={subtitle} actions={actions} />
      <div className="relative" style={{ height: `${height}px` }}>
        {children}
      </div>
    </div>
  );
};

export default ChartContainer;
