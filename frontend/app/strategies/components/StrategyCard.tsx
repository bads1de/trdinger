/**
 * 戦略カードコンポーネント
 *
 * 個別の投資戦略の要約情報を表示するカードコンポーネント
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

import React from "react";
import { StrategyCardProps, PerformanceColors } from "@/types/strategy-showcase";

/**
 * 戦略カードコンポーネント
 */
const StrategyCard: React.FC<StrategyCardProps> = ({
  strategy,
  onViewDetail,
  className = "",
}) => {
  /**
   * パフォーマンス指標の色分けを決定
   */
  const getPerformanceColors = (): PerformanceColors => {
    return {
      return: strategy.expected_return >= 10 
        ? "positive" 
        : strategy.expected_return >= 0 
        ? "neutral" 
        : "negative",
      sharpe: strategy.sharpe_ratio >= 2.0
        ? "excellent"
        : strategy.sharpe_ratio >= 1.0
        ? "good"
        : strategy.sharpe_ratio >= 0.5
        ? "fair"
        : "poor",
      drawdown: strategy.max_drawdown <= 5
        ? "low"
        : strategy.max_drawdown <= 15
        ? "medium"
        : "high",
    };
  };

  /**
   * リスクレベルのスタイルを取得
   */
  const getRiskLevelStyle = () => {
    switch (strategy.risk_level) {
      case "low":
        return "badge-success";
      case "medium":
        return "badge-warning";
      case "high":
        return "badge-danger";
      default:
        return "badge-secondary";
    }
  };

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

  const colors = getPerformanceColors();

  return (
    <div
      className={`enterprise-card hover:shadow-enterprise-lg transition-all duration-300 cursor-pointer group ${className}`}
      onClick={() => onViewDetail(strategy)}
    >
      <div className="p-6">
        {/* ヘッダー */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <span className="text-2xl">{getCategoryIcon()}</span>
            <div>
              <h3 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100 group-hover:text-primary-500 transition-colors">
                {strategy.name}
              </h3>
              <p className="text-sm text-secondary-600 dark:text-secondary-400">
                {getCategoryName()}
              </p>
            </div>
          </div>
          <span className={`badge ${getRiskLevelStyle()}`}>
            {getRiskLevelName()}
          </span>
        </div>

        {/* 説明 */}
        <p className="text-sm text-secondary-600 dark:text-secondary-400 mb-4 line-clamp-2">
          {strategy.description}
        </p>

        {/* 使用指標 */}
        <div className="mb-4">
          <p className="text-xs text-secondary-500 dark:text-secondary-500 mb-2">
            使用指標
          </p>
          <div className="flex flex-wrap gap-1">
            {strategy.indicators.slice(0, 3).map((indicator, index) => (
              <span
                key={index}
                className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200"
              >
                {indicator}
              </span>
            ))}
            {strategy.indicators.length > 3 && (
              <span className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-secondary-100 text-secondary-800 dark:bg-secondary-800 dark:text-secondary-200">
                +{strategy.indicators.length - 3}
              </span>
            )}
          </div>
        </div>

        {/* パフォーマンス指標 */}
        <div className="space-y-3">
          {/* 期待リターン */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-secondary-600 dark:text-secondary-400">
              期待リターン
            </span>
            <span
              className={`text-sm font-semibold ${
                colors.return === "positive"
                  ? "text-green-500"
                  : colors.return === "negative"
                  ? "text-red-500"
                  : "text-secondary-600 dark:text-secondary-400"
              }`}
            >
              {strategy.expected_return > 0 ? "+" : ""}{strategy.expected_return.toFixed(1)}%
            </span>
          </div>

          {/* シャープレシオ */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-secondary-600 dark:text-secondary-400">
              シャープレシオ
            </span>
            <span
              className={`text-sm font-semibold ${
                colors.sharpe === "excellent"
                  ? "text-green-500"
                  : colors.sharpe === "good"
                  ? "text-blue-500"
                  : colors.sharpe === "fair"
                  ? "text-yellow-500"
                  : "text-red-500"
              }`}
            >
              {strategy.sharpe_ratio.toFixed(2)}
            </span>
          </div>

          {/* 最大ドローダウン */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-secondary-600 dark:text-secondary-400">
              最大DD
            </span>
            <span
              className={`text-sm font-semibold ${
                colors.drawdown === "low"
                  ? "text-green-500"
                  : colors.drawdown === "medium"
                  ? "text-yellow-500"
                  : "text-red-500"
              }`}
            >
              -{strategy.max_drawdown.toFixed(1)}%
            </span>
          </div>

          {/* 勝率 */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-secondary-600 dark:text-secondary-400">
              勝率
            </span>
            <span className="text-sm font-semibold text-secondary-900 dark:text-secondary-100">
              {strategy.win_rate.toFixed(1)}%
            </span>
          </div>
        </div>

        {/* 推奨時間軸 */}
        <div className="mt-4 pt-4 border-t border-secondary-200 dark:border-secondary-700">
          <div className="flex items-center justify-between">
            <span className="text-xs text-secondary-500 dark:text-secondary-500">
              推奨時間軸
            </span>
            <span className="text-xs font-medium text-secondary-700 dark:text-secondary-300">
              {strategy.recommended_timeframe}
            </span>
          </div>
        </div>

        {/* ホバーエフェクト用のボタン */}
        <div className="mt-4 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
          <button className="w-full btn-primary text-sm py-2">
            詳細を見る
          </button>
        </div>
      </div>
    </div>
  );
};

export default StrategyCard;
