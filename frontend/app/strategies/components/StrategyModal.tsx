/**
 * 戦略詳細モーダルコンポーネント
 *
 * 選択された戦略の詳細情報を表示するモーダル
 *
 */

import React from "react";
import { UnifiedStrategy } from "@/types/auto-strategy";

interface StrategyModalProps {
  strategy: UnifiedStrategy | undefined;
  isOpen: boolean;
  onClose: () => void;
}

/**
 * 戦略詳細モーダルコンポーネント
 */
const StrategyModal: React.FC<StrategyModalProps> = ({
  strategy,
  isOpen,
  onClose,
}) => {
  if (!isOpen || !strategy) return null;

  /**
   * カテゴリのアイコンを取得
   */
  const getCategoryIcon = () => {
    switch (strategy.category) {
      case "trend_following":
        return "📈";
      case "mean_reversion":
        return "🔄";
      case "breakout":
        return "💥";
      case "range_trading":
        return "📊";
      case "momentum":
        return "⚡";
      default:
        return "🎯";
    }
  };

  /**
   * カテゴリの日本語名を取得
   */
  const getCategoryName = () => {
    switch (strategy.category) {
      case "trend_following":
        return "トレンドフォロー";
      case "mean_reversion":
        return "逆張り";
      case "breakout":
        return "ブレイクアウト";
      case "range_trading":
        return "レンジ取引";
      case "momentum":
        return "モメンタム";
      default:
        return "その他";
    }
  };

  /**
   * リスクレベルの日本語名を取得
   */
  const getRiskLevelName = () => {
    switch (strategy.risk_level) {
      case "low":
        return "低リスク";
      case "medium":
        return "中リスク";
      case "high":
        return "高リスク";
      default:
        return "不明";
    }
  };

  /**
   * リスクレベルのスタイルを取得
   */
  const getRiskLevelStyle = () => {
    switch (strategy.risk_level) {
      case "low":
        return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
      case "medium":
        return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
      case "high":
        return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200";
    }
  };

  /**
   * パフォーマンス指標の色を取得
   */
  const getPerformanceColor = (
    value: number,
    type: "return" | "sharpe" | "drawdown"
  ) => {
    switch (type) {
      case "return":
        return value >= 10
          ? "text-green-500"
          : value >= 0
          ? "text-secondary-600 dark:text-secondary-400"
          : "text-red-500";
      case "sharpe":
        return value >= 2.0
          ? "text-green-500"
          : value >= 1.0
          ? "text-blue-500"
          : value >= 0.5
          ? "text-yellow-500"
          : "text-red-500";
      case "drawdown":
        return value <= 5
          ? "text-green-500"
          : value <= 15
          ? "text-yellow-500"
          : "text-red-500";
      default:
        return "text-secondary-600 dark:text-secondary-400";
    }
  };

  /**
   * モーダル外クリックでクローズ
   */
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black bg-opacity-50 backdrop-blur-sm"
      onClick={handleBackdropClick}
    >
      <div className="enterprise-card max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          {/* ヘッダー */}
          <div className="flex items-start justify-between mb-6">
            <div className="flex items-center gap-4">
              <span className="text-3xl">{getCategoryIcon()}</span>
              <div>
                <h2 className="text-2xl font-bold text-secondary-900 dark:text-secondary-100">
                  {strategy.name}
                </h2>
                <p className="text-secondary-600 dark:text-secondary-400">
                  {getCategoryName()}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span
                className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getRiskLevelStyle()}`}
              >
                {getRiskLevelName()}
              </span>
              <button
                onClick={onClose}
                className="text-secondary-400 hover:text-secondary-600 dark:hover:text-secondary-300 transition-colors"
              >
                <svg
                  className="w-6 h-6"
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
            </div>
          </div>

          {/* 説明 */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100 mb-2">
              戦略概要
            </h3>
            <p className="text-secondary-600 dark:text-secondary-400">
              {strategy.description}
            </p>
          </div>

          {/* パフォーマンス指標 */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100 mb-4">
              📊 パフォーマンス指標
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-secondary-50 dark:bg-secondary-800 rounded-lg p-4">
                <p className="text-sm text-secondary-600 dark:text-secondary-400 mb-1">
                  期待リターン
                </p>
                <p
                  className={`text-2xl font-bold ${getPerformanceColor(
                    strategy.expected_return,
                    "return"
                  )}`}
                >
                  {strategy.expected_return > 0 ? "+" : ""}
                  {strategy.expected_return.toFixed(1)}%
                </p>
              </div>
              <div className="bg-secondary-50 dark:bg-secondary-800 rounded-lg p-4">
                <p className="text-sm text-secondary-600 dark:text-secondary-400 mb-1">
                  シャープレシオ
                </p>
                <p
                  className={`text-2xl font-bold ${getPerformanceColor(
                    strategy.sharpe_ratio,
                    "sharpe"
                  )}`}
                >
                  {strategy.sharpe_ratio.toFixed(2)}
                </p>
              </div>
              <div className="bg-secondary-50 dark:bg-secondary-800 rounded-lg p-4">
                <p className="text-sm text-secondary-600 dark:text-secondary-400 mb-1">
                  最大ドローダウン
                </p>
                <p
                  className={`text-2xl font-bold ${getPerformanceColor(
                    strategy.max_drawdown,
                    "drawdown"
                  )}`}
                >
                  -{strategy.max_drawdown.toFixed(1)}%
                </p>
              </div>
              <div className="bg-secondary-50 dark:bg-secondary-800 rounded-lg p-4">
                <p className="text-sm text-secondary-600 dark:text-secondary-400 mb-1">
                  勝率
                </p>
                <p className="text-2xl font-bold text-secondary-900 dark:text-secondary-100">
                  {strategy.win_rate.toFixed(1)}%
                </p>
              </div>
            </div>
          </div>

          {/* 使用指標 */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100 mb-4">
              🔧 使用指標
            </h3>
            <div className="flex flex-wrap gap-2">
              {strategy.indicators.map((indicator, index) => (
                <span
                  key={index}
                  className="inline-flex items-center px-3 py-2 rounded-lg text-sm font-medium bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200"
                >
                  {indicator}
                </span>
              ))}
            </div>
          </div>

          {/* パラメータ詳細 */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100 mb-4">
              ⚙️ パラメータ設定
            </h3>
            <div className="bg-secondary-50 dark:bg-secondary-800 rounded-lg p-4">
              <pre className="text-sm text-secondary-700 dark:text-secondary-300 overflow-x-auto">
                {JSON.stringify(strategy.parameters, null, 2)}
              </pre>
            </div>
          </div>

          {/* 戦略情報 */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100 mb-4">
                📋 戦略情報
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-secondary-600 dark:text-secondary-400">
                    推奨時間軸
                  </span>
                  <span className="font-medium text-secondary-900 dark:text-secondary-100">
                    {strategy.recommended_timeframe}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-secondary-600 dark:text-secondary-400">
                    作成日時
                  </span>
                  <span className="font-medium text-secondary-900 dark:text-secondary-100">
                    {new Date(strategy.created_at).toLocaleDateString("ja-JP")}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-secondary-600 dark:text-secondary-400">
                    更新日時
                  </span>
                  <span className="font-medium text-secondary-900 dark:text-secondary-100">
                    {new Date(strategy.updated_at).toLocaleDateString("ja-JP")}
                  </span>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100 mb-4">
                🎯 戦略の特徴
              </h3>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-primary-500 rounded-full"></span>
                  <span className="text-sm text-secondary-600 dark:text-secondary-400">
                    {strategy.indicators.length}個の指標を使用
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-primary-500 rounded-full"></span>
                  <span className="text-sm text-secondary-600 dark:text-secondary-400">
                    {getRiskLevelName()}の投資戦略
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-primary-500 rounded-full"></span>
                  <span className="text-sm text-secondary-600 dark:text-secondary-400">
                    {getCategoryName()}タイプ
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* アクションボタン */}
          <div className="mt-8 flex justify-end gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-secondary-700 dark:text-secondary-300 bg-secondary-100 dark:bg-secondary-700 rounded-md hover:bg-secondary-200 dark:hover:bg-secondary-600 transition-colors"
            >
              閉じる
            </button>
            <button className="btn-primary">この戦略でバックテスト</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StrategyModal;
